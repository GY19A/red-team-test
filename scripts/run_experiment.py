#!/usr/bin/env python3
"""Prepare fixed test sets and run one or all methods on both HarmBench and JailbreakBench separately."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--method", choices=["Crescendo", "TAP", "CKA-Agent", "x-teaming", "all"], default="all")
    ap.add_argument("--run-id", default="default")
    ap.add_argument("--max-prompts", type=int, default=None, help="Cap prompts per dataset (cost control).")
    ap.add_argument(
        "--parallel",
        action="store_true",
        help="Run all selected methods as separate processes in parallel (one OS process per method).",
    )
    ap.add_argument(
        "--dataset",
        choices=["harmbench", "jailbreakbench", "both"],
        default="both",
        help="Which dataset(s) to run on. Default: both (runs separately on each)."
    )
    args = ap.parse_args()

    py = ROOT / ".venv" / "bin" / "python"
    if not py.exists():
        py = Path(sys.executable)

    subprocess.check_call([str(py), str(ROOT / "scripts" / "prepare_test_sets.py")], cwd=str(ROOT))
    if args.prepare_only:
        return

    test_sets = ROOT / "data" / "test_sets"
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    # Build runners for each method and dataset combination
    all_runners: list[tuple[str, list[str]]] = []

    if args.dataset in ("harmbench", "both"):
        harmbench_prompts = test_sets / "harmbench_50.jsonl"
    else:
        harmbench_prompts = None

    if args.dataset in ("jailbreakbench", "both"):
        jailbreakbench_prompts = test_sets / "jailbreakbench_50.jsonl"
    else:
        jailbreakbench_prompts = None

    if args.method in ("Crescendo", "all"):
        if harmbench_prompts:
            all_runners.append(("Crescendo-HarmBench", [
                str(py), str(ROOT / "methods" / "Crescendo" / "run.py"),
                "--prompts", str(harmbench_prompts),
                "--run-id", args.run_id,
                "--dataset", "harmbench",
            ]))
        if jailbreakbench_prompts:
            all_runners.append(("Crescendo-JailbreakBench", [
                str(py), str(ROOT / "methods" / "Crescendo" / "run.py"),
                "--prompts", str(jailbreakbench_prompts),
                "--run-id", args.run_id,
                "--dataset", "jailbreakbench",
            ]))
        if args.max_prompts is not None:
            for name, runner in all_runners:
                if name.startswith("Crescendo-"):
                    runner.extend(["--max-prompts", str(args.max_prompts)])

    if args.method in ("TAP", "all"):
        if harmbench_prompts:
            all_runners.append(("TAP-HarmBench", [
                str(py), str(ROOT / "methods" / "TAP" / "run.py"),
                "--prompts", str(harmbench_prompts),
                "--run-id", args.run_id,
                "--dataset", "harmbench",
            ]))
        if jailbreakbench_prompts:
            all_runners.append(("TAP-JailbreakBench", [
                str(py), str(ROOT / "methods" / "TAP" / "run.py"),
                "--prompts", str(jailbreakbench_prompts),
                "--run-id", args.run_id,
                "--dataset", "jailbreakbench",
            ]))
        if args.max_prompts is not None:
            for name, runner in all_runners:
                if name.startswith("TAP-"):
                    runner.extend(["--max-prompts", str(args.max_prompts)])

    if args.method in ("CKA-Agent", "all"):
        if args.dataset in ("harmbench", "both"):
            all_runners.append(("CKA-Agent-HarmBench", [
                str(py), str(ROOT / "methods" / "CKA-Agent" / "run.py"),
                "--run-id", args.run_id,
                "--dataset", "harmbench",
            ]))
        if args.dataset in ("jailbreakbench", "both"):
            all_runners.append(("CKA-Agent-JailbreakBench", [
                str(py), str(ROOT / "methods" / "CKA-Agent" / "run.py"),
                "--run-id", args.run_id,
                "--dataset", "jailbreakbench",
            ]))
        if args.max_prompts is not None:
            for name, runner in all_runners:
                if name.startswith("CKA-Agent-"):
                    runner.extend(["--max-prompts", str(args.max_prompts)])

    if args.method in ("x-teaming", "all"):
        if args.dataset in ("harmbench", "both"):
            all_runners.append(("x-teaming-HarmBench", [
                str(py), str(ROOT / "methods" / "x-teaming" / "run.py"),
                "--run-id", args.run_id,
                "--dataset", "harmbench",
            ]))
        if args.dataset in ("jailbreakbench", "both"):
            all_runners.append(("x-teaming-JailbreakBench", [
                str(py), str(ROOT / "methods" / "x-teaming" / "run.py"),
                "--run-id", args.run_id,
                "--dataset", "jailbreakbench",
            ]))
        if args.max_prompts is not None:
            for name, runner in all_runners:
                if name.startswith("x-teaming-"):
                    runner.extend(["--max-prompts", str(args.max_prompts)])

    # Extract unique method names for parallel execution grouping
    method_order = []
    if args.method != "all":
        method_order = [args.method]
    else:
        method_order = ["Crescendo", "TAP", "CKA-Agent", "x-teaming"]

    use_parallel = args.parallel and len(all_runners) > 1
    if use_parallel:
        print(f"Parallel run: {len(all_runners)} processes", flush=True)
        procs: list[tuple[str, subprocess.Popen]] = []
        for name, runner in all_runners:
            print(f"[spawn] {name}", flush=True)
            procs.append((name, subprocess.Popen(runner, cwd=str(ROOT), env=env)))
        failed: list[tuple[str, int]] = []
        for name, p in procs:
            code = p.wait()
            print(f"[exit {code}] {name}", flush=True)
            if code != 0:
                failed.append((name, code))
        if failed:
            for name, code in failed:
                print(f"Failed: {name} (exit {code})", file=sys.stderr)
            sys.exit(1)
    else:
        for name, runner in all_runners:
            print(f"=== {name} ===")
            try:
                subprocess.check_call(runner, cwd=str(ROOT), env=env)
            except subprocess.CalledProcessError as e:
                print(f"Failed: {name}", e, file=sys.stderr)


if __name__ == "__main__":
    main()
