from __future__ import annotations

import re
import string
from collections.abc import Sequence
from dataclasses import dataclass

from servingbench_tuner.quality.dataset import EvalExample

from .base import EvalResult, Evaluator


def _normalize(s: str) -> str:
    s = s.strip().lower()
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    # strip punctuation around tokens
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = s.strip()
    return s


def _try_parse_float(s: str) -> float | None:
    # extract first number-like token
    m = re.search(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


@dataclass
class ExactMatchEvaluator(Evaluator):
    """
    Supports:
      - closed_qa / exact_match: answer or answers (aliases)
      - regex_match: regex pattern must match output
      - numeric: compare number with tolerance

    Example JSONL:
      {"id":"1","type":"closed_qa","prompt":"2+2?","answer":"4"}
      {"id":"2","type":"regex_match","prompt":"...","regex":"^hello"}
      {"id":"3","type":"numeric","prompt":"...","numeric":{"target":3.14,"tolerance":0.01}}
    """

    name: str = "exact_match"
    supported_types: Sequence[str] = ("closed_qa", "exact_match", "regex_match", "numeric")

    async def evaluate(self, example: EvalExample, output_text: str) -> EvalResult:
        et = example.type
        out_raw = output_text or ""
        out_norm = _normalize(out_raw)

        if et in ("closed_qa", "exact_match"):
            ans = example.get("answer", None)
            aliases = example.get("answers", None)
            golds: list[str] = []
            if isinstance(ans, str) and ans.strip():
                golds.append(ans)
            if isinstance(aliases, list):
                golds.extend([str(x) for x in aliases if str(x).strip()])

            golds_norm = [_normalize(g) for g in golds]
            passed = out_norm in set(golds_norm)
            score = 1.0 if passed else 0.0
            return EvalResult(
                example_id=example.id,
                example_type=et,
                score=score,
                passed=passed,
                details={
                    "output_norm": out_norm,
                    "gold_norm": golds_norm,
                    "match": passed,
                },
            )

        if et == "regex_match":
            pattern = str(example.get("regex", "") or "")
            if not pattern:
                return EvalResult(example.id, et, 0.0, False, {"error": "missing regex"})
            try:
                ok = re.search(pattern, out_raw, flags=re.MULTILINE) is not None
            except re.error as e:
                return EvalResult(example.id, et, 0.0, False, {"error": f"bad regex: {e}"})
            return EvalResult(example.id, et, 1.0 if ok else 0.0, ok, {"regex": pattern})

        if et == "numeric":
            spec = example.get("numeric", {}) or {}
            target = spec.get("target", None)
            tol = float(spec.get("tolerance", 0.0) or 0.0)
            try:
                target_f = float(target)
            except Exception:
                return EvalResult(
                    example.id, et, 0.0, False, {"error": "numeric.target missing/invalid"}
                )
            out_f = _try_parse_float(out_raw)
            if out_f is None:
                return EvalResult(
                    example.id,
                    et,
                    0.0,
                    False,
                    {"target": target_f, "tolerance": tol, "parsed": None},
                )
            ok = abs(out_f - target_f) <= tol
            return EvalResult(
                example.id,
                et,
                1.0 if ok else 0.0,
                ok,
                {"target": target_f, "tolerance": tol, "parsed": out_f},
            )

        return EvalResult(example.id, et, 0.0, False, {"error": f"unsupported type {et}"})
