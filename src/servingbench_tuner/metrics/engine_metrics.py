from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class EngineMetricsConfig:
    """
    Optional engine metrics fetcher.

    Typical cases:
    - vLLM metrics endpoint (often Prometheus text format)
      e.g. http://127.0.0.1:8000/metrics
    - A standalone Prometheus endpoint that scrapes the server
    """

    metrics_url: str = "http://127.0.0.1:8000/metrics"
    timeout_s: float = 3.0


def parse_prometheus_text(text: str) -> dict[str, float]:
    """
    Minimal Prometheus text parser:
    - extracts lines like: metric_name{...} 123.45
    - collapses labels (sums by metric name) to keep it simple
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # metric{label="x"} 123
        m = re.match(
            r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+([-+]?\d+(\.\d+)?([eE][-+]?\d+)?)$", line
        )
        if not m:
            continue
        name = m.group(1)
        val = float(m.group(3))
        out[name] = out.get(name, 0.0) + val
    return out


async def fetch_engine_metrics(cfg: EngineMetricsConfig) -> tuple[dict[str, Any], str | None]:
    """
    Fetch engine metrics from cfg.metrics_url.
    Returns: (metrics_dict, error_str)

    - If endpoint returns Prometheus text, parse_prometheus_text is applied.
    - If endpoint returns JSON, it will be loaded directly.
    """
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(cfg.timeout_s)) as client:
            r = await client.get(cfg.metrics_url)
            r.raise_for_status()
            content_type = (r.headers.get("content-type") or "").lower()
            body = r.text

            if "application/json" in content_type or body.strip().startswith("{"):
                try:
                    return (r.json(), None)
                except Exception:
                    # fallback to text parse
                    return (parse_prometheus_text(body), None)

            # assume prometheus text
            return (parse_prometheus_text(body), None)
    except Exception as e:
        return ({}, str(e))


def pick_common_vllm_signals(metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Best-effort extraction of commonly useful vLLM/Prometheus signals.
    Names vary across versions; this function is intentionally tolerant.
    """
    # If JSON dict from custom endpoint: just pass through
    if not isinstance(metrics, dict):
        return {}

    # Prometheus parsed dict: float values by metric name
    if all(isinstance(v, int | float) for v in metrics.values()):
        m = metrics  # type: ignore[assignment]
        keys = [
            # common-ish examples (may differ)
            "vllm:gpu_cache_usage_ratio",
            "vllm:num_requests_running",
            "vllm:num_requests_waiting",
            "vllm:request_queue_time_seconds_sum",
            "vllm:request_queue_time_seconds_count",
            "vllm:prefill_time_seconds_sum",
            "vllm:prefill_time_seconds_count",
            "vllm:decode_time_seconds_sum",
            "vllm:decode_time_seconds_count",
        ]
        out: dict[str, Any] = {}
        for k in keys:
            if k in m:
                out[k] = m[k]
        return out

    return {}
