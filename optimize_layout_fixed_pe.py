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

TOTAL_PES = 1024
COMPONENTS = ["ATM", "LND", "ICE", "CPL", "ROF", "GLC", "WAV", "OCN"]
TASK_MAX_LIMITS = {"ATM": 5400}
TASK_MIN_LIMITS = {"ATM": 64, "OCN": 512}

def parse_args():
    '''
    >>> import sys
    >>> sys.argv = ['script_name', '--caseroot', '/test/case', '--num-trials', '10']
    >>> args = parse_args()
    >>> args.caseroot == '/test/case'
    True
    >>> args.num_trials == 10
    True
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--caseroot", default=os.getcwd())
    parser.add_argument("--num-trials", type=int, default=20)
    parser.add_argument("--metric", choices=["throughput", "cost"], default="cost")
    parser.add_argument("--stop-option", default="ndays")
    parser.add_argument("--stop-n", type=int, default=5)
    return parser.parse_args()

def snap_to_nearest(value):
    candidates = [i for i in range(1, TOTAL_PES + 1) if MAX_MPITASKS_PER_NODE % i == 0 or i % MAX_MPITASKS_PER_NODE == 0]
    return min(candidates, key=lambda x: abs(x - value))

def assign_rootpes(task_counts, overlap_map):
    rootpes = {}
    ocn_end = task_counts["OCN"]
    rootpes["OCN"] = 0

    assigned = set(["OCN"])
    remaining = [c for c in COMPONENTS if c != "OCN"]

    random.shuffle(remaining)
    while remaining:
        a = remaining.pop(0)
        if a in assigned:
            continue
        overlap_group = [a] + [b for b in remaining if overlap_map[a][b]]
        overlap_group.sort(key=lambda c: task_counts[c], reverse=True)

        base = ocn_end
        for i, comp in enumerate(overlap_group):
            ntasks = task_counts[comp]
            rootpes[comp] = base
            if i + 1 < len(overlap_group):
                next_comp = overlap_group[i + 1]
                next_ntasks = task_counts[next_comp]
                if next_ntasks < ntasks:
                    base += next_ntasks
                else:
                    base += ntasks
            assigned.add(comp)
            if comp in remaining:
                remaining.remove(comp)

    return rootpes

def configure_case(case: Case, config, stop_option, stop_n):
    for comp, vals in config.items():
        case.set_value(f"NTASKS_{comp}", vals["ntasks"])
        case.set_value(f"NTHRDS_{comp}", vals["nthrds"])
        case.set_value(f"ROOTPE_{comp}", vals["rootpe"])
    case.set_value("STOP_OPTION", stop_option)
    case.set_value("STOP_N", stop_n)
    case.case_setup(reset=True)
    build.case_build(case._caseroot, case=case)
    case.submit(no_batch=True)

def parse_timing(case_dir):
    '''
    >>> import tempfile, os
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     timing_dir = os.path.join(tmpdir, 'timing')
    ...     os.mkdir(timing_dir)
    ...     with open(os.path.join(timing_dir, 'cesm_timing.testlog'), 'w') as f:
    ...         _ = f.write(' Model Cost:  1000.0\\n')
    ...         _ = f.write(' Model Throughput: 10.0\\n')
    ...         _ = f.write(' TOT Run Time:  500.0 seconds\\n')
    ...         _ = f.write(' ATM Run Time: 400.0 seconds\\n')
    ...     result = parse_timing(tmpdir)
    >>> result['total_cost'] == 1000.0
    True
    >>> result['throughput'] == 10.0
    True
    >>> result['run_time'] == 500.0
    True
    >>> result['component_times']['ATM'] == 400.0
    True
    '''
    timing_dir = os.path.join(case_dir, "timing")
    files = sorted(f for f in os.listdir(timing_dir) if f.startswith("cesm_timing"))
    if not files:
        raise RuntimeError("No timing files found.")
    last_file = os.path.join(timing_dir, files[-1])
    result = {"total_cost": None, "throughput": None, "run_time": None, "component_times": {}}
    with open(last_file) as f:
        for line in f:
            if m := re.search(r"Model Cost:\s+([\d.]+)", line):
                result["total_cost"] = float(m.group(1))
            elif m := re.search(r"Model Throughput:\s+([\d.]+)", line):
                result["throughput"] = float(m.group(1))
            elif m := re.search(r"TOT Run Time:\s+([\d.]+) seconds", line):
                result["run_time"] = float(m.group(1))
            elif m := re.search(r"^\s+([A-Z]{3}) Run Time:\s+([\d.]+) seconds", line):
                result["component_times"][m.group(1)] = float(m.group(2))
    if not all([result["total_cost"], result["throughput"], result["run_time"]]):
        raise RuntimeError("Incomplete timing file")
    return result

def objective(trial, caseroot, stop_option, stop_n, metric):
    with Case(caseroot, read_only=False) as case:
        MAX_MPITASKS_PER_NODE = case.get_value("MAX_MPITASKS_PER_NODE")
        comp_types = {comp: case.get_value(f"COMP_{comp}") for comp in COMPONENTS}
    stub_comps = {comp for comp, val in comp_types.items() if val.lower().startswith("s")}

    raw_weights = {comp: trial.suggest_float(f"weight_{comp}", 0.01, 1.0)
                   for comp in COMPONENTS if comp not in stub_comps}
    weight_sum = sum(raw_weights.values())

    task_counts = {}
    for comp in COMPONENTS:
        if comp in stub_comps:
            task_counts[comp] = 1
        else:
            raw = raw_weights[comp] / weight_sum * TOTAL_PES
            bounded = min(max(TASK_MIN_LIMITS.get(comp, 1), int(raw)), TASK_MAX_LIMITS.get(comp, TOTAL_PES))
            snapped = snap_to_nearest(bounded)
            task_counts[comp] = snapped
            print(f"{comp} raw={int(raw)}, bounded={bounded}, snapped={snapped}")

    overlap_map = {a: {b: trial.suggest_categorical(f"{a}_overlaps_{b}", [True, False])
                       if a != b and b != "OCN" and a != "OCN" else False for b in COMPONENTS}
                   for a in COMPONENTS}

    try:
        rootpes = assign_rootpes(task_counts, overlap_map)
    except Exception as e:
        print(f"❌ Failed to assign rootpes: {e}")
        raise optuna.TrialPruned()

    config = {comp: {"ntasks": 1, "nthrds": 1, "rootpe": 0} if comp in stub_comps else {
              "ntasks": task_counts[comp], "nthrds": 1, "rootpe": rootpes[comp]} for comp in COMPONENTS}

    with Case(caseroot, read_only=False) as case:
        try:
            configure_case(case, config, stop_option, stop_n)
        except Exception as e:
            print(f"⚠️ Model config/build failed: {e}")
            raise optuna.TrialPruned()
        try:
            results = parse_timing(caseroot)
        except Exception as e:
            print(f"⚠️ Timing error: {e}")
            raise optuna.TrialPruned()
    return results["throughput"] if metric == "throughput" else results["total_cost"]

def main():
    args = parse_args()
    study = optuna.create_study(direction="maximize" if args.metric == "throughput" else "minimize")
    study.optimize(lambda trial: objective(trial, args.caseroot, args.stop_option, args.stop_n, args.metric),
                   n_trials=args.num_trials)
    print("Best trial:")
    print(study.best_trial)

if __name__ == "__main__":
    main()
