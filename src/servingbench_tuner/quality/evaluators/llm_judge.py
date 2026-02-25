from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from servingbench_tuner.backends.base import BackendAdapter, GenerationRequest
from servingbench_tuner.quality.dataset import EvalExample

from .base import EvalResult, Evaluator

DEFAULT_JUDGE_PROMPT = """You are a strict evaluator for LLM outputs.
You must score the candidate output according to the rubric.
Return ONLY a JSON object with keys:
  - score: integer in [1, 5]
  - rationale: short string (<= 80 words)

Rubric:
{rubric}

Task prompt:
{prompt}

Candidate output:
{output}
"""


def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()
    # if already json
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass
    # find first { ... } block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


@dataclass
class LLMJudgeEvaluator(Evaluator):
    """
    LLM-as-judge evaluator.

    - Calls a judge model via BackendAdapter
    - Fixed prompt template + temperature=0
    - Attempts to pass 'seed' through extra (some servers support it)
    - Expects JSON output: {"score": 1..5, "rationale": "..."}
    """

    name: str = "llm_judge"
    supported_types: Sequence[str] = ("llm_judge", "judge")

    judge_backend: BackendAdapter | None = None
    judge_model: str | None = None
    temperature: float = 0.0
    max_new_tokens: int = 256
    seed: int = 42

    # default scale
    score_min: int = 1
    score_max: int = 5
    pass_threshold: float = 0.6  # normalized [0,1]

    async def evaluate(self, example: EvalExample, output_text: str) -> EvalResult:
        if self.judge_backend is None:
            return EvalResult(
                example.id, example.type, 0.0, False, {"error": "judge_backend not configured"}
            )

        rubric = str(example.get("rubric", "") or "")
        # allow override
        judge_prompt = str(example.get("judge_prompt", "") or "").strip()
        if not judge_prompt:
            judge_prompt = DEFAULT_JUDGE_PROMPT

        # allow custom scale per example
        scale = example.get("scale", None) or {}
        smin = int(scale.get("min", self.score_min))
        smax = int(scale.get("max", self.score_max))
        smin = min(smin, smax)
        smax = max(smin, smax)

        prompt = judge_prompt.format(
            rubric=rubric,
            prompt=example.prompt,
            output=(output_text or ""),
        )

        req = GenerationRequest(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            max_new_tokens=int(example.get("judge_max_new_tokens", self.max_new_tokens)),
            temperature=float(example.get("judge_temperature", self.temperature)),
            top_p=1.0,
            stream=False,
            timeout_s=float(example.get("judge_timeout_s", 60.0)),
            extra={
                # Some servers support seed; harmless if ignored
                "seed": int(example.get("judge_seed", self.seed)),
                # Encourage JSON-only output
                "response_format": {"type": "json_object"},
            },
        )

        res = await self.judge_backend.generate(req)
        obj = _extract_json(res.text)

        if not obj or "score" not in obj:
            return EvalResult(
                example.id,
                example.type,
                0.0,
                False,
                {"error": "judge output not parseable json", "raw": res.text[:2000]},
            )

        try:
            score_raw = int(obj["score"])
        except Exception:
            return EvalResult(
                example.id, example.type, 0.0, False, {"error": "judge score not int", "obj": obj}
            )

        score_clamped = max(smin, min(smax, score_raw))
        # normalize to 0..1
        denom = max(1, (smax - smin))
        score_norm = float((score_clamped - smin) / denom)
        passed = score_norm >= float(example.get("pass_threshold", self.pass_threshold))

        return EvalResult(
            example_id=example.id,
            example_type=example.type,
            score=score_norm,
            passed=passed,
            details={
                "score_raw": score_raw,
                "score_clamped": score_clamped,
                "score_norm": score_norm,
                "rationale": obj.get("rationale", ""),
                "judge_meta": res.meta,
            },
        )
