#!/usr/bin/env python3
"""Build deterministic 50+50 evaluation slices (HarmBench JSON + JailbreakBench harmful)."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_from_disk

from rtbench.paths import DATA_TEST_SETS, DATASETS, ROOT

DEFAULT_HB = DATASETS / "HarmBench" / "harmbench_full.json"
DEFAULT_JBB = DATASETS / "JailbreakBench"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-each", type=int, default=50)
    ap.add_argument("--harmbench-json", type=Path, default=DEFAULT_HB)
    ap.add_argument("--jailbreakbench-dir", type=Path, default=DEFAULT_JBB)
    ap.add_argument("--out-dir", type=Path, default=DATA_TEST_SETS)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    hb_raw = json.loads(Path(args.harmbench_json).read_text(encoding="utf-8"))
    hb_items = []
    for i, row in enumerate(hb_raw):
        goal = row.get("goal") or row.get("Goal") or ""
        if not goal.strip():
            continue
        hb_items.append(
            {
                "id": f"harmbench_{i}",
                "source": "harmbench",
                "goal": goal,
                "meta": {k: v for k, v in row.items() if k != "goal"},
            }
        )
    if len(hb_items) < args.n_each:
        raise RuntimeError(f"Not enough HarmBench rows: {len(hb_items)} < {args.n_each}")
    hb_pick = rng.sample(hb_items, args.n_each)

    jbb = load_from_disk(str(args.jailbreakbench_dir))
    harmful = jbb["harmful"]
    rows = harmful.to_list()
    if len(rows) < args.n_each:
        raise RuntimeError(f"Not enough JBB harmful rows: {len(rows)} < {args.n_each}")
    jbb_pick = rng.sample(rows, args.n_each)
    jbb_out = []
    for r in jbb_pick:
        goal = r.get("Goal") or r.get("goal") or ""
        jbb_out.append(
            {
                "id": f"jbb_harmful_{r.get('Index', r.get('index'))}",
                "source": "jailbreakbench",
                "goal": goal,
                "meta": {k: v for k, v in r.items() if k not in ("Goal", "goal")},
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": args.seed,
        "n_each": args.n_each,
        "harmbench_path": str(args.harmbench_json.resolve()),
        "jailbreakbench_dir": str(args.jailbreakbench_dir.resolve()),
        "harmbench_ids": [x["id"] for x in hb_pick],
        "jailbreakbench_ids": [x["id"] for x in jbb_out],
        "combined_100": "combined_100.jsonl",
        "project_root": str(ROOT.resolve()),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, data in [("harmbench_50.jsonl", hb_pick), ("jailbreakbench_50.jsonl", jbb_out)]:
        p = args.out_dir / name
        p.write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in data) + "\n", encoding="utf-8"
        )
    combined = hb_pick + jbb_out
    (args.out_dir / "combined_100.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in combined) + "\n", encoding="utf-8"
    )
    print("Wrote", args.out_dir / "manifest.json")
    print("Wrote", args.out_dir / "harmbench_50.jsonl", "and", args.out_dir / "jailbreakbench_50.jsonl")
    print("Wrote", args.out_dir / "combined_100.jsonl")


if __name__ == "__main__":
    main()
