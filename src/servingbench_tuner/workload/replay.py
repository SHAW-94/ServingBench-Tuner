from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ArrivalEvent:
    """
    Unified event format for arrival generation.

    timestamp_s: seconds since start (float)
    request_id: unique id
    session_id / turn_id: optional (for multi-turn)
    input_tokens / output_tokens: token lengths (can be used directly or as hints)
    """

    timestamp_s: float
    request_id: str
    session_id: str | None = None
    turn_id: int | None = None
    category: str = "replay"
    input_tokens: int | None = None
    output_tokens: int | None = None
    raw: dict[str, Any] | None = None


def read_jsonl_trace(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"trace jsonl not found: {p}")
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def parse_trace_rows(rows: Iterable[dict[str, Any]]) -> list[ArrivalEvent]:
    events: list[ArrivalEvent] = []
    for i, r in enumerate(rows):
        ts = float(r.get("timestamp_s", r.get("t", 0.0)))
        rid = str(r.get("request_id", r.get("id", f"req_{i:06d}")))
        ev = ArrivalEvent(
            timestamp_s=ts,
            request_id=rid,
            session_id=r.get("session_id", None),
            turn_id=r.get("turn_id", None),
            category=str(r.get("category", "replay")),
            input_tokens=(int(r["input_tokens"]) if "input_tokens" in r else None),
            output_tokens=(int(r["output_tokens"]) if "output_tokens" in r else None),
            raw=dict(r),
        )
        events.append(ev)
    # Ensure stable order by timestamp then request_id
    events.sort(key=lambda e: (e.timestamp_s, e.request_id))
    return events


def load_replay_events(trace_path: str | Path) -> list[ArrivalEvent]:
    """
    Read JSONL trace and parse into ArrivalEvent list.
    """
    rows = read_jsonl_trace(trace_path)
    return parse_trace_rows(rows)
