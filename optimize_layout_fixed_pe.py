#!/usr/bin/env python3
import os, sys
_LIBDIR = os.environ["CIMEROOT"]
sys.path.append(_LIBDIR)

import argparse
import optuna
import re
import random
from CIME.case import Case
import CIME.build as build

# Fixed PE budget and known component names
TOTAL_PES = 1024
COMPONENTS = ["ATM", "LND", "ICE", "CPL", "ROF", "GLC", "WAV", "OCN"]
TASK_MAX_LIMITS = {
    "ATM": 5400  # Upper bound known 
}
TASK_MIN_LIMITS = {
    "ATM": 64  ,
    "OCN": 512 # Lower bound known from experience
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caseroot", default=os.getcwd(), help="Path to the CESM case directory")
    parser.add_argument("--num-trials", type=int, default=20)
    parser.add_argument("--metric", choices=["throughput", "cost"], default="cost")
    parser.add_argument("--stop-option", default="ndays")
    parser.add_argument("--stop-n", type=int, default=5)
    return parser.parse_args()

def assign_rootpes(task_counts, stub_comps, overlap_groups):
    """
    Assigns ROOTPE values based on:
      - OCN gets exclusive PE range starting at 0
      - Each group of overlapping components gets packed sequentially
    """
    assigned = {}
    current_rootpe = 0

    # Assign OCN its own exclusive block
    ocn_ntasks = task_counts["OCN"]
    if ocn_ntasks > TOTAL_PES:
        raise optuna.TrialPruned()
    assigned["OCN"] = 0
    current_rootpe += ocn_ntasks

    # For each overlap group (can be just one component), assign shared ROOTPE
    for group in overlap_groups:
        max_ntasks = max(task_counts[comp] for comp in group if comp not in stub_comps)
        if current_rootpe + max_ntasks > TOTAL_PES:
            raise optuna.TrialPruned()

        for comp in group:
            if comp in stub_comps or comp == "OCN":
                continue
            assigned[comp] = current_rootpe
        current_rootpe += max_ntasks  # Increment by maximum task span

    return assigned


def configure_case(case: Case, config, stop_option, stop_n):
    for comp, vals in config.items():
        case.set_value(f"NTASKS_{comp}", vals["ntasks"])
        case.set_value(f"NTHRDS_{comp}", vals["nthrds"])
        case.set_value(f"ROOTPE_{comp}", vals["rootpe"])
    case.set_value("STOP_OPTION", stop_option)
    case.set_value("STOP_N", stop_n)
    case.case_setup(reset=True)
    build.case_build(case._caseroot, case=case)
    case.submit()

def parse_timing(case_dir):
    timing_dir = os.path.join(case_dir, "timing")
    files = sorted(f for f in os.listdir(timing_dir) if f.startswith("cesm_timing"))
    if not files:
        raise RuntimeError("No timing files found.")

    last_file = os.path.join(timing_dir, files[-1])
    result = {
        "total_cost": None,
        "throughput": None,
        "run_time": None,
        "component_times": {}
    }

    component_time_re = re.compile(r"^\s+([A-Z]{3}) Run Time:\s+([\d\.]+) seconds")
    overall_cost_re = re.compile(r"Model Cost:\s+([\d\.]+)")
    throughput_re = re.compile(r"Model Throughput:\s+([\d\.]+)")
    total_runtime_re = re.compile(r"TOT Run Time:\s+([\d\.]+) seconds")

    with open(last_file) as f:
        for line in f:
            line = line.strip()
            if m := overall_cost_re.search(line):
                result["total_cost"] = float(m.group(1))
            elif m := throughput_re.search(line):
                result["throughput"] = float(m.group(1))
            elif m := component_time_re.search(line):
                comp, time_sec = m.groups()
                result["component_times"][comp] = float(time_sec)
            elif m := total_runtime_re.search(line):
                result["run_time"] = float(m.group(1))

    if result["total_cost"] is None or result["throughput"] is None or result["run_time"] is None:
        raise RuntimeError("Missing timing metrics in timing file.")

    return result

def legal_ntasks(value, max_mpi):
    """
    Return the closest integer to `value` that is either a multiple or a divisor of `max_mpi`.

    The result is chosen from the union of:
    - all divisors of `max_mpi`
    - all multiples of `max_mpi` up to TOTAL_PES

    >>> legal_ntasks(563, 128)
    512
    >>> legal_ntasks(600, 128)
    640
    >>> legal_ntasks(64, 128)
    64
    >>> legal_ntasks(7, 128)
    8
    >>> legal_ntasks(130, 128)
    128
    >>> legal_ntasks(127, 128)
    128
    >>> legal_ntasks(1, 128)
    1
    """
    # Generate both divisors and multiples of MAX_MPITASKS_PER_NODE
    divisors = [i for i in range(1, max_mpi + 1) if max_mpi % i == 0]
    multiples = [i for i in range(max_mpi, TOTAL_PES + 1, max_mpi)]
    candidates = sorted(set(divisors + multiples))
    return min(candidates, key=lambda x: abs(x - value))


def objective(trial, caseroot, stop_option, stop_n, metric):
    with Case(caseroot, read_only=False) as case:
        comp_types = {comp: case.get_value(f"COMP_{comp}") for comp in COMPONENTS}
        max_mpitasks_per_node = case.get_value("MAX_MPITASKS_PER_NODE")
    stub_comps = {comp for comp, val in comp_types.items() if val.lower().startswith("s")}

    # Sample overlap strategy: a partition of non-OCN components into groups
    components_to_group = [comp for comp in COMPONENTS if comp != "OCN" and comp not in stub_comps]
    num_groups = trial.suggest_int("num_groups", 1, len(components_to_group))

    # Assign components to groups
    shuffled = components_to_group.copy()

    random.shuffle(shuffled)
    overlap_groups = [[] for _ in range(num_groups)]
    for i, comp in enumerate(shuffled):
        overlap_groups[i % num_groups].append(comp)

    # Sample NTASK weights
    raw_weights = {
        comp: trial.suggest_float(f"weight_{comp}", 0.01, 1.0)
        for comp in COMPONENTS if comp not in stub_comps
    }
    weight_sum = sum(raw_weights.values())

    # Assign NTASKS aligned to node hardware
    task_counts = {}
    for comp in COMPONENTS:
        if comp in stub_comps:
            task_counts[comp] = 1
        else:
            raw = raw_weights[comp] / weight_sum * TOTAL_PES
            bounded = min(max(TASK_MIN_LIMITS.get(comp, 1), int(raw)), TASK_MAX_LIMITS.get(comp, TOTAL_PES))
            snapped = max(1, round(bounded / max_mpitasks_per_node)) * max_mpitasks_per_node
            task_counts[comp] = snapped
            print(f"{comp} raw={int(raw)}, bounded={bounded}, snapped={snapped}")

    try:
        rootpes = assign_rootpes(task_counts, stub_comps, overlap_groups)
    except Exception:
        raise optuna.TrialPruned()

    # Build configuration dictionary
    config = {}
    for comp in COMPONENTS:
        if comp in stub_comps:
            config[comp] = {"ntasks": 1, "nthrds": 1, "rootpe": 0}
        else:
            config[comp] = {
                "ntasks": task_counts[comp],
                "nthrds": 1,
                "rootpe": rootpes[comp]
            }

    with Case(caseroot, read_only=False) as case:
        try:
            configure_case(case, config, stop_option, stop_n)
        except Exception as e:
            print(f"⚠️ Trial pruned due to model failure: {e}")
            raise optuna.TrialPruned()

        try:
            results = parse_timing(caseroot)
        except (FileNotFoundError, RuntimeError):
            print("⚠️ Timing file not found or incomplete — pruning trial.")
            raise optuna.TrialPruned()

    return results["throughput"] if metric == "throughput" else results["total_cost"]


def main():
    args = parse_args()
    study = optuna.create_study(direction="maximize" if args.metric == "throughput" else "minimize")
    study.optimize(
        lambda trial: objective(trial, args.caseroot, args.stop_option, args.stop_n, args.metric),
        n_trials=args.num_trials
    )
    print("Best trial:")
    print(study.best_trial)

if __name__ == "__main__":
    main()
