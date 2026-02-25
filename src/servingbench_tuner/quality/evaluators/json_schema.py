from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from servingbench_tuner.quality.dataset import EvalExample

from .base import EvalResult, Evaluator


def _extract_first_json(text: str) -> str | None:
    """
    Best-effort extract a JSON object/array substring from a text blob.
    """
    if not text:
        return None
    text = text.strip()
    # quick path
    if text.startswith("{") or text.startswith("["):
        return text
    # search for first { ... } block
    start = text.find("{")
    if start == -1:
        start = text.find("[")
    if start == -1:
        return None
    return text[start:]


@dataclass
class JSONSchemaEvaluator(Evaluator):
    """
    Validate that output is JSON and conforms to provided JSON Schema.

    Example JSONL:
      {
        "id":"x",
        "type":"json_schema",
        "prompt":"Return JSON with keys a,b",
        "schema": {"type":"object","required":["a","b"],"properties":{"a":{"type":"number"},"b":{"type":"string"}}}
      }
    """

    name: str = "json_schema"
    supported_types: Sequence[str] = ("json_schema",)

    async def evaluate(self, example: EvalExample, output_text: str) -> EvalResult:
        schema = example.get("schema", None)
        if not isinstance(schema, dict):
            return EvalResult(
                example.id, example.type, 0.0, False, {"error": "missing schema dict"}
            )

        candidate_text = _extract_first_json(output_text or "")
        if not candidate_text:
            return EvalResult(example.id, example.type, 0.0, False, {"error": "no json found"})

        try:
            obj = json.loads(candidate_text)
        except Exception as e:
            return EvalResult(
                example.id, example.type, 0.0, False, {"error": f"json parse error: {e}"}
            )

        # validate schema
        try:
            import jsonschema  # type: ignore
        except Exception as e:
            return EvalResult(
                example.id,
                example.type,
                0.0,
                False,
                {"error": "jsonschema not installed. install jsonschema", "exc": str(e)},
            )

        try:
            jsonschema.validate(instance=obj, schema=schema)
            return EvalResult(example.id, example.type, 1.0, True, {"parsed": obj})
        except Exception as e:
            return EvalResult(
                example.id, example.type, 0.0, False, {"parsed": obj, "validation_error": str(e)}
            )
