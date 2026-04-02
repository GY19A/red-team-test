from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
from typing import Any
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
LOCAL_PYRIT = ROOT / "references" / "PyRIT"
if LOCAL_PYRIT.exists():
    sys.path.insert(0, str(LOCAL_PYRIT))

from pyrit.executor.attack.core import AttackAdversarialConfig, AttackScoringConfig
from pyrit.executor.attack.multi_turn.crescendo import CrescendoAttack
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.setup import initialize_pyrit_async

from rtbench.models_config import load_attacker_defender, require_token
from rtbench.paths import DATA_TEST_SETS
from rtbench.results_io import append_jsonl, method_results_dir, write_json
from rtbench.results_schema import PromptResult, utc_now


def _piece_text(piece: Any) -> str | None:
    if piece is None:
        return None
    cv = getattr(piece, "converted_value", None)
    if cv is not None:
        return str(cv)
    return str(piece)


def _summarize_attack_result(result: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "outcome": result.outcome.value if hasattr(result.outcome, "value") else str(result.outcome),
        "executed_turns": getattr(result, "executed_turns", None),
        "execution_time_ms": getattr(result, "execution_time_ms", None),
        "last_response": _piece_text(getattr(result, "last_response", None)),
        "backtrack_count": getattr(result, "backtrack_count", None),
    }
    ls = getattr(result, "last_score", None)
    if ls is not None:
        out["last_score"] = getattr(ls, "score_value", str(ls))
    return out


async def run_one_crescendo(
    objective: str,
    *,
    attacker: OpenAIChatTarget,
    defender: OpenAIChatTarget,
    max_turns: int = 8,
    max_backtracks: int = 6,
) -> Any:
    adversarial_cfg = AttackAdversarialConfig(target=attacker)
    scoring_cfg = AttackScoringConfig()
    attack = CrescendoAttack(
        objective_target=defender,
        attack_adversarial_config=adversarial_cfg,
        attack_scoring_config=scoring_cfg,
        max_turns=max_turns,
        max_backtracks=max_backtracks,
    )
    return await attack.execute_async(objective=objective)


async def run_batch(
    prompts_path: Path,
    run_id: str,
    *,
    max_prompts: int | None = None,
    max_turns: int = 8,
    max_backtracks: int = 6,
) -> None:
    attacker_ep, defender_ep = load_attacker_defender()
    require_token(attacker_ep.base_url, attacker_ep.api_key)

    await initialize_pyrit_async(memory_db_type="InMemory", silent=True)

    attacker = OpenAIChatTarget(
        endpoint=attacker_ep.base_url,
        api_key=attacker_ep.api_key,
        model_name=attacker_ep.model,
    )
    defender = OpenAIChatTarget(
        endpoint=defender_ep.base_url,
        api_key=defender_ep.api_key,
        model_name=defender_ep.model,
    )

    # Determine dataset name from path for output directory
    dataset_name = prompts_path.stem  # e.g., "harmbench_50" or "jailbreakbench_50"
    out_dir = method_results_dir("Crescendo", run_id) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.jsonl"

    lines = prompts_path.read_text(encoding="utf-8").strip().splitlines()
    if max_prompts is not None:
        lines = lines[:max_prompts]

    agg: list[dict] = []
    progress = tqdm(
        lines,
        desc=f"Crescendo {dataset_name}",
        unit="prompt",
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    )
    for line in progress:
        row = json.loads(line)
        pr = PromptResult(
            prompt_id=row["id"],
            source=row["source"],
            method="Crescendo",
            objective=row["goal"],
            attacker_model=attacker_ep.model,
            target_model=defender_ep.model,
            success=None,
            error=None,
            started_at=utc_now(),
        )
        try:
            result = await run_one_crescendo(
                row["goal"],
                attacker=attacker,
                defender=defender,
                max_turns=max_turns,
                max_backtracks=max_backtracks,
            )
            summ = _summarize_attack_result(result)
            pr.raw = summ
            pr.success = summ.get("outcome") == "success"
            pr.finished_at = utc_now()
        except Exception as e:
            pr.error = str(e)
            pr.finished_at = utc_now()
        append_jsonl(summary_path, pr.to_json())
        agg.append(pr.to_json())
        progress.set_postfix(done=len(agg), errors=sum(1 for item in agg if item.get("error")), refresh=False)

    write_json(out_dir / "aggregate.json", {"run_id": run_id, "dataset": dataset_name, "n": len(agg), "results": agg})


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="PyRIT Crescendo (microsoft PyRIT package).")
    ap.add_argument("--prompts", type=Path, default=None, help="Path to prompts file (jsonl). If not specified, runs on both harmbench_50.jsonl and jailbreakbench_50.jsonl.")
    ap.add_argument("--run-id", type=str, default="smoke")
    ap.add_argument("--max-prompts", type=int, default=None)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--max-backtracks", type=int, default=6)
    ap.add_argument("--dataset", choices=["harmbench", "jailbreakbench", "both"], default="both",
                    help="Which dataset(s) to run on. Default: both")
    args = ap.parse_args()

    if args.prompts:
        # Single file mode
        asyncio.run(
            run_batch(
                args.prompts,
                args.run_id,
                max_prompts=args.max_prompts,
                max_turns=args.max_turns,
                max_backtracks=args.max_backtracks,
            )
        )
    else:
        # Run on both datasets separately
        if args.dataset in ("harmbench", "both"):
            p = DATA_TEST_SETS / "harmbench_50.jsonl"
            print(f"[Crescendo] Running on HarmBench: {p}")
            asyncio.run(
                run_batch(
                    p,
                    args.run_id,
                    max_prompts=args.max_prompts,
                    max_turns=args.max_turns,
                    max_backtracks=args.max_backtracks,
                )
            )
        if args.dataset in ("jailbreakbench", "both"):
            p = DATA_TEST_SETS / "jailbreakbench_50.jsonl"
            print(f"[Crescendo] Running on JailbreakBench: {p}")
            asyncio.run(
                run_batch(
                    p,
                    args.run_id,
                    max_prompts=args.max_prompts,
                    max_turns=args.max_turns,
                    max_backtracks=args.max_backtracks,
                )
            )


if __name__ == "__main__":
    main()
