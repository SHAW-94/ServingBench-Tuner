from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalExample:
    """
    One evaluation example from eval_pack JSONL.

    Minimal required fields:
      - id: unique string
      - type: evaluator routing key (e.g. "closed_qa", "json_schema", "code_unittest", "llm_judge")
      - prompt: input prompt or instruction

    Optional fields vary by evaluator:
      exact_match:
        - answer: str
        - answers: [str] (aliases)
        - regex: str
        - numeric: {target: number, tolerance: float}
      json_schema:
        - schema: JSON Schema dict
      code_unittest:
        - tests: str (pytest content)
        - filename: str (defaults to "solution.py")
      llm_judge:
        - rubric: str
        - judge_prompt: str (optional override)
        - scale: {min:1, max:5} (optional)
    """

    id: str
    type: str
    prompt: str

    # the rest is raw payload
    fields: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.fields.get(key, default)

    @property
    def category(self) -> str:
        return str(self.fields.get("category", self.type))

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "prompt": self.prompt, **self.fields}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"eval_pack not found: {p}")
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            if not isinstance(obj, dict):
                raise ValueError(f"eval_pack row must be JSON object: {p}")
            rows.append(obj)
    return rows


def load_eval_pack(path: str | Path) -> list[EvalExample]:
    """
    Load eval_pack JSONL into EvalExample list with minimal validation.
    """
    rows = read_jsonl(path)
    out: list[EvalExample] = []
    seen = set()

    for i, r in enumerate(rows):
        ex_id = str(r.get("id", "")).strip()
        ex_type = str(r.get("type", "")).strip()
        prompt = str(r.get("prompt", "")).strip()

        if not ex_id:
            ex_id = f"ex_{i:04d}"
        if ex_id in seen:
            raise ValueError(f"duplicate eval example id: {ex_id}")
        seen.add(ex_id)

        if not ex_type:
            raise ValueError(f"eval example {ex_id} missing 'type'")
        if not prompt:
            # allow empty prompt for some special test cases, but usually this is a bug
            prompt = ""

        # keep extra fields
        fields = {k: v for k, v in r.items() if k not in ("id", "type", "prompt")}
        out.append(EvalExample(id=ex_id, type=ex_type, prompt=prompt, fields=fields))

    return out


def group_by_type(examples: Iterable[EvalExample]) -> dict[str, list[EvalExample]]:
    m: dict[str, list[EvalExample]] = {}
    for ex in examples:
        m.setdefault(ex.type, []).append(ex)
    return m


def dump_eval_pack(path: str | Path, examples: list[EvalExample]) -> None:
    """
    Dump examples back to JSONL (useful for generated public packs).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
