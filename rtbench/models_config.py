from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import yaml

from rtbench.paths import CONFIGS


ROOT = Path(__file__).resolve().parents[1]
# Load local secrets file (if present). This is process-local and works with parallel runs
# because each subprocess imports this module.
load_dotenv(ROOT / ".env", override=False)


@dataclass
class ModelEndpoint:
    name: str
    api_key: str
    base_url: str
    model: str


def load_models_yaml(path: str | Path | None = None) -> dict[str, Any]:
    p = CONFIGS / "models.yaml" if path is None else Path(path)
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _merge_defaults(cfg: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    defaults = cfg.get("defaults") or {}
    if not isinstance(defaults, dict):
        defaults = {}
    return {**defaults, **entry}


def endpoint_from_merged(entry: dict[str, Any], role_name: str) -> ModelEndpoint:
    # Prefer env-based secrets so tokens are not stored in this repo.
    key_env = str(entry.get("api_key_env") or "DEEPINFRA_TOKEN").strip()
    api_key = str((os.environ.get(key_env) or "")).strip()
    base = str((entry.get("base_url") or "")).strip()
    model = entry["model"]
    return ModelEndpoint(name=role_name, api_key=api_key, base_url=base, model=model)


def load_attacker_defender(cfg: dict[str, Any] | None = None) -> tuple[ModelEndpoint, ModelEndpoint]:
    cfg = cfg or load_models_yaml()
    atk_key = next(iter(cfg["attackers"]))
    def_key = next(iter(cfg["defenders"]))
    atk_entry = _merge_defaults(cfg, cfg["attackers"][atk_key])
    def_entry = _merge_defaults(cfg, cfg["defenders"][def_key])
    return (
        endpoint_from_merged(atk_entry, atk_key),
        endpoint_from_merged(def_entry, def_key),
    )


def require_token(base_url: str, api_key: str) -> None:
    if not api_key:
        raise RuntimeError(
            "Missing API key. Put it into `.env` (see .env.example) as DEEPINFRA_TOKEN (or set defaults.api_key_env in configs/models.yaml)."
        )
    if not base_url:
        raise RuntimeError("Missing base URL: set defaults.base_url in configs/models.yaml.")
