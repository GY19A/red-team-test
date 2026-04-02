from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
 

@dataclass
class TurnRecord:
    role: str
    content: str


@dataclass
class PromptResult:
    prompt_id: str
    source: str
    method: str
    objective: str
    attacker_model: str
    target_model: str
    success: bool | None
    error: str | None
    raw: dict = field(default_factory=dict)
    turns: list[dict] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""

    def to_json(self) -> dict:
        d = asdict(self)
        return d


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
