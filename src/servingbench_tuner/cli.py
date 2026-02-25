from __future__ import annotations

import json
import math
import os
import random
import sqlite3
import statistics
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from servingbench_tuner.core.config import load_yaml, stable_json_dumps
from servingbench_tuner.core.logging import setup_logging
from servingbench_tuner.core.reproducibility import collect_env_signature
from servingbench_tuner.core.types import (
    E2EMetrics,
    GateResult,
    QualitySummary,
    RequestTrace,
    RunRecord,
    ServerMetrics,
    WorkloadSpec,
)

app = typer.Typer(add_completion=False, help="ServingBench-Tuner (no-docker) CLI")
console = Console()


# ---------------------------
# Utilities
# ---------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dirs(out_dir: Path) -> None:
    (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)


def _default_db_path(out_dir: Path) -> Path:
    return out_dir / "runs.sqlite"


def _run_id(prefix: str = "run") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}_{uuid.uuid4().hex[:8]}"


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    k = (len(v) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(v[int(k)])
    d0 = v[f] * (c - k)
    d1 = v[c] * (k - f)
    return float(d0 + d1)


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(values))


def _read_jsonl_trace(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"trace jsonl not found: {p}")
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def _load_workload_spec(workload_yaml: str) -> WorkloadSpec:
    cfg = load_yaml(workload_yaml)
    # Minimal mapping from YAML to WorkloadSpec
    arrival = cfg.get("arrival", {}) or {}
    length_dist = cfg.get("length_dist", {}) or {}
    prompt_dist = length_dist.get("prompt", {}) or {}
    output_dist = length_dist.get("output", {}) or {}
    sessions = cfg.get("sessions", {}) or {}

    spec = WorkloadSpec(
        name=str(cfg.get("name", Path(workload_yaml).stem)),
        seed=int(cfg.get("seed", 42)),
        arrival_mode=str(arrival.get("mode", "replay")),
        trace_path=arrival.get("trace_path", None),
        concurrency_limit=int(cfg.get("concurrency_limit", 64)),
        timeout_s=float(cfg.get("timeout_s", 30.0)),
        retries=int(cfg.get("retries", 0)),
        prompt_len_path=prompt_dist.get("path", None),
        output_len_path=output_dist.get("path", None),
        sessions_enabled=bool(sessions.get("enabled", False)),
        turns=int(sessions.get("turns", 1)),
        context_growth=str(sessions.get("context_growth", "append")),
        max_context_tokens=int(sessions.get("max_context_tokens", 8192)),
        warmup_requests=int(cfg.get("warmup_requests", 50)),
        duration_s=int(cfg.get("duration_s", 300)),
    )
    # Replay requires trace_path
    if spec.arrival_mode == "replay" and not spec.trace_path:
        raise ValueError("workload arrival.mode=replay requires arrival.trace_path")
    return spec


def _load_quality_cfg(quality_yaml: str) -> dict[str, Any]:
    cfg = load_yaml(quality_yaml)
    # expected structure produced by quickstart_cpu.sh
    if "eval_pack" not in cfg:
        raise ValueError("quality config must include eval_pack")
    return cfg


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------
# SQLite storage (stdlib)
# ---------------------------


def _init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY,
              created_at TEXT NOT NULL,
              kind TEXT NOT NULL,
              backend TEXT NOT NULL,
              workload_path TEXT NOT NULL,
              serving_path TEXT NOT NULL,
              quality_path TEXT NOT NULL,
              tuner_path TEXT NOT NULL,
              env_json TEXT NOT NULL,
              metrics_json TEXT NOT NULL,
              server_json TEXT NOT NULL,
              quality_json TEXT NOT NULL,
              artifacts_json TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tuning_candidates (
              run_id TEXT NOT NULL,
              cand_id TEXT NOT NULL,
              params_json TEXT NOT NULL,
              objectives_json TEXT NOT NULL,
              constraints_json TEXT NOT NULL,
              passed_gate INTEGER NOT NULL,
              PRIMARY KEY (run_id, cand_id)
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def _insert_run(db_path: Path, record: RunRecord) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO runs
            (run_id, created_at, kind, backend, workload_path, serving_path, quality_path, tuner_path,
             env_json, metrics_json, server_json, quality_json, artifacts_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                record.run_id,
                record.created_at,
                record.kind,
                record.backend,
                record.workload_path,
                record.serving_path,
                record.quality_path,
                record.tuner_path,
                stable_json_dumps(record.env_json),
                stable_json_dumps(record.metrics_json),
                stable_json_dumps(record.server_json),
                stable_json_dumps(record.quality_json),
                stable_json_dumps(record.artifacts),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _insert_candidate(
    db_path: Path,
    run_id: str,
    cand_id: str,
    params: dict[str, Any],
    objectives: dict[str, float],
    constraints: dict[str, float],
    passed_gate: bool,
) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO tuning_candidates
            (run_id, cand_id, params_json, objectives_json, constraints_json, passed_gate)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (
                run_id,
                cand_id,
                stable_json_dumps(params),
                stable_json_dumps(objectives),
                stable_json_dumps(constraints),
                1 if passed_gate else 0,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _get_run(db_path: Path, run_id: str) -> dict[str, Any] | None:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run_id,))
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _get_latest_run_id(db_path: Path) -> str | None:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute("SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1;")
        row = cur.fetchone()
        return row["run_id"] if row else None
    finally:
        conn.close()


# ---------------------------
# Mock benchmark engine
# ---------------------------


def _mock_serving_effects(params: dict[str, Any]) -> dict[str, float]:
    """
    A tiny param->performance mapping so tuning makes sense even with mock backend.

    Conventions:
    - Higher concurrency -> more queueing (tail latency up)
    - Higher max_batch_tokens -> better throughput but worse TTFT slightly
    - Higher max_new_tokens -> more decode potential; if too low can hurt quality
    - Lower kv_cache_limit_mb -> potential quality drop + tail latency up (simulated)
    """
    concurrency = int(params.get("concurrency", 8))
    max_batch_tokens = int(params.get("max_batch_tokens", 2048))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    kv_cache_limit_mb = int(params.get("kv_cache_limit_mb", 2048))

    # Prefill/decode "speed" multipliers (higher is faster)
    speed = 1.0
    speed *= 1.0 + min(0.35, math.log(max_batch_tokens / 512, 2) * 0.10)  # diminishing returns
    speed *= 1.0 - min(0.25, max(0, concurrency - 8) * 0.02)  # contention
    speed *= 1.0 - min(0.20, max(0, 1024 - kv_cache_limit_mb) / 4096)  # cache pressure
    speed = max(0.35, speed)

    # TTFT base shifts with batching (slightly worse) and cache pressure
    ttft_base = (
        0.08 + (max_batch_tokens / 4096) * 0.08 + max(0, 1024 - kv_cache_limit_mb) / 1024 * 0.04
    )

    # decode time per token
    tpot = 0.010 / speed  # seconds per output token
    prefill_per_tok = 0.00010 / speed  # seconds per input token

    # Quality proxy: aggressive settings reduce it
    quality = 0.98
    if max_new_tokens < 64:
        quality -= 0.05
    if concurrency > 12:
        quality -= 0.03
    if kv_cache_limit_mb < 768:
        quality -= 0.06
    quality = max(0.60, min(0.99, quality))

    return {
        "speed": speed,
        "ttft_base": ttft_base,
        "tpot": tpot,
        "prefill_per_tok": prefill_per_tok,
        "quality_level": quality,
    }


def _simulate_queueing(
    arrivals: list[tuple[float, str, str | None, int | None, int, int]],
    concurrency: int,
    timeout_s: float,
    params: dict[str, Any],
    seed: int,
) -> list[RequestTrace]:
    """
    Simple multi-server queue simulation:
    - Each request has arrival time a
    - Service time = prefill + decode
    - Assign to earliest-available worker
    - Compute queue_time, start, first_token, end
    """
    rng = random.Random(seed)
    fx = _mock_serving_effects(params)

    # workers' next free times
    workers = [0.0 for _ in range(max(1, concurrency))]
    traces: list[RequestTrace] = []

    for a, rid, sid, turn_id, in_tok, out_tok in arrivals:
        # compute service components
        jitter = rng.lognormvariate(0.0, 0.15)  # mild lognormal jitter
        prefill = (fx["ttft_base"] + fx["prefill_per_tok"] * in_tok) * jitter
        decode = (fx["tpot"] * out_tok) * (0.9 + 0.2 * rng.random())  # small variance

        service = prefill + decode

        # assign earliest worker
        i = min(range(len(workers)), key=lambda k: workers[k])
        start = max(a, workers[i])
        queue_t = max(0.0, start - a)
        first_token = start + prefill
        end = start + service
        workers[i] = end

        status = "ok"
        if (end - a) > timeout_s:
            status = "timeout"

        traces.append(
            RequestTrace(
                request_id=rid,
                arrival_s=a,
                start_s=start,
                first_token_s=first_token,
                end_s=end,
                input_tokens=in_tok,
                output_tokens=out_tok,
                status=status,  # type: ignore
                retry_count=0,
                session_id=sid,
                turn_id=turn_id,
                queue_time_s=queue_t,
                prefill_time_s=prefill,
                decode_time_s=decode,
            )
        )

    return traces


def _aggregate_e2e(traces: list[RequestTrace]) -> E2EMetrics:
    ok = [t for t in traces if t.status == "ok"]
    timeouts = [t for t in traces if t.status == "timeout"]
    errors = [t for t in traces if t.status == "error"]

    ttft = [t.ttft_s for t in ok]
    tpot = [t.tpot_s for t in ok if t.output_tokens > 0]
    lat = [t.latency_s for t in ok]

    queue = [t.queue_time_s for t in ok]
    prefill = [t.prefill_time_s for t in ok]
    decode = [t.decode_time_s for t in ok]

    # Throughput measured by total tokens / (makespan)
    if ok:
        makespan = max(t.end_s for t in ok) - min(t.arrival_s for t in ok)
        makespan = max(1e-9, makespan)
        total_tokens = sum(t.output_tokens for t in ok)
        total_reqs = len(ok)
        rps = total_reqs / makespan
        tok_s = total_tokens / makespan
    else:
        rps = 0.0
        tok_s = 0.0

    jitter_std = _std(lat)
    p50 = _percentile(lat, 50) if lat else 0.0
    p99 = _percentile(lat, 99) if lat else 0.0
    tail_amp = (p99 / p50) if p50 > 1e-9 else 0.0

    return E2EMetrics(
        ttft_p50=_percentile(ttft, 50),
        ttft_p95=_percentile(ttft, 95),
        tpot_p50=_percentile(tpot, 50),
        tpot_p95=_percentile(tpot, 95),
        latency_p50=_percentile(lat, 50),
        latency_p95=_percentile(lat, 95),
        latency_p99=_percentile(lat, 99),
        rps=rps,
        tok_s=tok_s,
        timeout_rate=(len(timeouts) / max(1, len(traces))),
        error_rate=(len(errors) / max(1, len(traces))),
        retry_rate=0.0,
        jitter_std=jitter_std,
        tail_amp=tail_amp,
        queue_p95=_percentile(queue, 95),
        prefill_p95=_percentile(prefill, 95),
        decode_p95=_percentile(decode, 95),
    )


def _server_metrics_proxy(params: dict[str, Any]) -> ServerMetrics:
    """
    Proxy server metrics for mock backend: VRAM scales with kv cache and model len.
    Keeps interface consistent with production runs.
    """
    kv_cache_limit_mb = int(params.get("kv_cache_limit_mb", 2048))
    max_model_len = int(params.get("max_model_len", 8192))
    # crude model: base + kv_cache + context overhead
    base = 4500.0  # MB
    vram_peak = base + kv_cache_limit_mb * 0.9 + (max_model_len / 8192) * 800.0
    vram_avg = base + kv_cache_limit_mb * 0.6 + (max_model_len / 8192) * 500.0
    frag = min(1.8, 1.1 + max(0, 2048 - kv_cache_limit_mb) / 4096)
    return ServerMetrics(
        vram_peak_mb=vram_peak,
        vram_avg_mb=vram_avg,
        vram_fragmentation_ratio=frag,
        gpu_util_avg=None,
    )


# ---------------------------
# Quality gate (mock)
# ---------------------------


def _load_eval_pack(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"eval_pack not found: {p}")
    items = []
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            items.append(json.loads(ln))
    return items


def _evaluate_quality_mock(
    eval_pack: list[dict[str, Any]], params: dict[str, Any], seed: int, out_path: Path
) -> QualitySummary:
    """
    Deterministic-ish quality simulation:
    - closed_qa: correct with probability ~ quality_level
    - json_schema: valid if quality_level >= 0.90 else fail
    - judge: score 1-5 mapped from quality_level

    Produces details JSON for transparency.
    """
    rng = random.Random(seed)
    fx = _mock_serving_effects(params)
    q = fx["quality_level"]

    details = []
    by_type_scores: dict[str, list[float]] = {}

    for ex in eval_pack:
        ex_id = ex.get("id", "unknown")
        ex_type = ex.get("type", "unknown")

        # deterministic per-example roll (stable)
        roll = (hash(ex_id) % 10_000) / 10_000.0
        roll = (roll + rng.random() * 0.001) % 1.0  # tiny noise but stable-ish

        passed = True
        score = 1.0

        if ex_type == "closed_qa":
            passed = roll < q
            score = 1.0 if passed else 0.0

        elif ex_type == "json_schema":
            passed = q >= 0.90
            score = 1.0 if passed else 0.0

        elif ex_type == "judge":
            # 1..5 scaled, normalize to 0..1
            judge_score = 1.0 + 4.0 * q
            score = max(0.0, min(1.0, judge_score / 5.0))
            passed = score >= 0.6

        else:
            # unknown types: neutral
            passed = True
            score = 1.0

        by_type_scores.setdefault(ex_type, []).append(score)
        details.append(
            {
                "id": ex_id,
                "type": ex_type,
                "score": score,
                "pass": passed,
                "quality_level": q,
            }
        )

    # aggregate
    all_scores = [d["score"] for d in details]
    overall = float(sum(all_scores) / max(1, len(all_scores)))
    pass_rate = float(sum(1 for d in details if d["pass"]) / max(1, len(details)))

    by_type = {k: float(sum(v) / max(1, len(v))) for k, v in by_type_scores.items()}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"overall": overall, "pass_rate": pass_rate, "by_type": by_type, "details": details},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return QualitySummary(
        overall=overall, pass_rate=pass_rate, by_type=by_type, details_path=str(out_path)
    )


def _quality_gate(
    policy: dict[str, Any], cand: QualitySummary, baseline: QualitySummary
) -> GateResult:
    gate = policy.get("gate", {}) if "gate" in policy else policy
    min_overall = float(gate.get("min_overall", 0.0))
    min_rel = float(gate.get("min_relative_to_baseline", 0.0))
    hard_fail_tasks = set(gate.get("hard_fail_tasks", []) or [])

    passed = True
    reasons: dict[str, Any] = {}

    if cand.overall < min_overall:
        passed = False
        reasons["min_overall"] = {"cand": cand.overall, "min": min_overall}

    # relative check only if baseline is meaningful (>0)
    if baseline.overall > 1e-9:
        rel = cand.overall / baseline.overall
        if rel < min_rel:
            passed = False
            reasons["relative_to_baseline"] = {
                "cand": cand.overall,
                "base": baseline.overall,
                "rel": rel,
                "min_rel": min_rel,
            }

    # hard fail: any specified type score < 1.0 (for schema-like tasks)
    for t in hard_fail_tasks:
        if cand.by_type.get(t, 1.0) < 0.999:
            passed = False
            reasons.setdefault("hard_fail", {})[t] = cand.by_type.get(t)

    return GateResult(passed=passed, reasons=reasons)


# ---------------------------
# Report
# ---------------------------


def _render_report_html(run_row: dict[str, Any], out_path: Path) -> None:
    """
    Minimal HTML report (can be upgraded later with plots + pareto).
    """
    metrics = json.loads(run_row["metrics_json"])
    quality = json.loads(run_row["quality_json"])
    server = json.loads(run_row["server_json"])
    env = json.loads(run_row["env_json"])
    artifacts = json.loads(run_row["artifacts_json"])

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>ServingBench-Tuner Report - {run_row["run_id"]}</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; margin-bottom: 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    .k {{ color: #6b7280; }}
    code, pre {{ background: #f3f4f6; padding: 2px 6px; border-radius: 6px; }}
    pre {{ padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>ServingBench-Tuner Report</h1>
  <p><b>Run ID:</b> <code>{run_row["run_id"]}</code></p>
  <p><b>Kind:</b> {run_row["kind"]} &nbsp;&nbsp; <b>Backend:</b> {run_row["backend"]} &nbsp;&nbsp; <b>Created:</b> {run_row["created_at"]}</p>

  <div class="card">
    <h2>Recommendation Card (Current Run)</h2>
    <div class="grid">
      <div><span class="k">Latency P95</span><br><b>{metrics.get("latency_p95", 0):.3f}s</b></div>
      <div><span class="k">TTFT P95</span><br><b>{metrics.get("ttft_p95", 0):.3f}s</b></div>
      <div><span class="k">tok/s</span><br><b>{metrics.get("tok_s", 0):.2f}</b></div>
      <div><span class="k">timeout rate</span><br><b>{metrics.get("timeout_rate", 0):.3f}</b></div>
      <div><span class="k">Quality overall</span><br><b>{quality.get("overall", 0):.3f}</b></div>
      <div><span class="k">VRAM peak (proxy)</span><br><b>{server.get("vram_peak_mb", 0):.0f} MB</b></div>
    </div>
  </div>

  <div class="card">
    <h2>Tail Latency Breakdown (P95)</h2>
    <div class="grid">
      <div><span class="k">queue_p95</span><br><b>{metrics.get("queue_p95", 0):.3f}s</b></div>
      <div><span class="k">prefill_p95</span><br><b>{metrics.get("prefill_p95", 0):.3f}s</b></div>
      <div><span class="k">decode_p95</span><br><b>{metrics.get("decode_p95", 0):.3f}s</b></div>
      <div><span class="k">tail_amp (p99/p50)</span><br><b>{metrics.get("tail_amp", 0):.2f}x</b></div>
    </div>
    <p style="color:#6b7280;margin-top:12px;">
      Tip: If queue_p95 dominates, focus on concurrency/batching/rate limiting. If decode dominates, focus on max_new_tokens/speculative/quantization.
    </p>
  </div>

  <div class="card">
    <h2>Artifacts</h2>
    <pre>{json.dumps(artifacts, ensure_ascii=False, indent=2)}</pre>
  </div>

  <div class="card">
    <h2>Env Signature</h2>
    <pre>{json.dumps(env, ensure_ascii=False, indent=2)}</pre>
  </div>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


# ---------------------------
# CLI Commands
# ---------------------------


@app.callback()
def _main(log_level: str = typer.Option("INFO", help="Log level (INFO/DEBUG/...)")) -> None:
    setup_logging(log_level)


@app.command()
def benchmark(
    workload: str = typer.Option(..., "--workload", help="Path to workload YAML"),
    serving: str = typer.Option("", "--serving", help="Path to serving YAML (optional)"),
    quality: str = typer.Option("", "--quality", help="Path to quality YAML (optional)"),
    backend: str = typer.Option("mock", "--backend", help="Backend: mock (CPU) or vllm (future)"),
    out: str = typer.Option(
        "results", "--out", help="Output directory (contains runs.sqlite, artifacts/)"
    ),
) -> None:
    """
    Run a single benchmark with mock backend and store results in SQLite + artifacts.
    """
    out_dir = Path(out)
    _ensure_dirs(out_dir)
    db_path = _default_db_path(out_dir)
    _init_db(db_path)

    workload_path = str(Path(workload))
    serving_path = str(Path(serving)) if serving else ""
    quality_path = str(Path(quality)) if quality else ""

    spec = _load_workload_spec(workload_path)
    # rng = random.Random(spec.seed)

    serving_cfg = load_yaml(serving_path) if serving_path and Path(serving_path).exists() else {}
    # For mock backend, we only use a small set of knobs (but pass-through is fine)
    params = {
        "concurrency": int(serving_cfg.get("concurrency", spec.concurrency_limit)),
        "max_new_tokens": int(serving_cfg.get("max_new_tokens", 256)),
        "max_batch_tokens": int(serving_cfg.get("max_batch_tokens", 2048)),
        "kv_cache_limit_mb": int(serving_cfg.get("kv_cache_limit_mb", 2048)),
        "max_model_len": int(serving_cfg.get("max_model_len", spec.max_context_tokens)),
    }

    # Build arrivals from replay trace
    trace_rows = _read_jsonl_trace(spec.trace_path) if spec.trace_path else []
    arrivals = []
    for row in trace_rows:
        arrivals.append(
            (
                float(row.get("timestamp_s", 0.0)),
                str(row.get("request_id", f"req_{uuid.uuid4().hex[:8]}")),
                row.get("session_id", None),
                row.get("turn_id", None),
                int(row.get("input_tokens", 64)),
                int(row.get("output_tokens", 128)),
            )
        )

    # Warmup: ignore first warmup_requests for metrics (still affects worker state)
    warm = min(spec.warmup_requests, len(arrivals))
    all_traces = _simulate_queueing(
        arrivals=arrivals,
        concurrency=int(params["concurrency"]),
        timeout_s=spec.timeout_s,
        params=params,
        seed=spec.seed,
    )
    traces = all_traces[warm:] if warm < len(all_traces) else all_traces

    e2e = _aggregate_e2e(traces)
    server_m = _server_metrics_proxy(params)

    run_id = _run_id("bench")
    art_dir = out_dir / "artifacts" / run_id
    art_dir.mkdir(parents=True, exist_ok=True)

    traces_path = art_dir / "traces.json"
    metrics_path = art_dir / "metrics.json"
    server_path = art_dir / "server.json"

    _write_json(traces_path, [asdict(t) for t in traces])
    _write_json(metrics_path, asdict(e2e))
    _write_json(server_path, asdict(server_m))

    # Quality (optional)
    quality_json: dict[str, Any] = {
        "overall": 1.0,
        "pass_rate": 1.0,
        "by_type": {},
        "details_path": "",
    }
    if quality_path and Path(quality_path).exists():
        qcfg = _load_quality_cfg(quality_path)
        eval_pack = _load_eval_pack(qcfg["eval_pack"])
        q_details_path = art_dir / "quality_details.json"
        qsum = _evaluate_quality_mock(
            eval_pack, params=params, seed=spec.seed, out_path=q_details_path
        )
        # baseline: for single benchmark we assume baseline == candidate (passes relative check)
        gate_res = _quality_gate(qcfg, qsum, qsum)
        quality_json = {
            "overall": qsum.overall,
            "pass_rate": qsum.pass_rate,
            "by_type": qsum.by_type,
            "details_path": qsum.details_path,
            "gate": asdict(gate_res),
        }
        _write_json(art_dir / "quality.json", quality_json)
    else:
        _write_json(art_dir / "quality.json", quality_json)

    env = collect_env_signature().to_json()

    record = RunRecord(
        run_id=run_id,
        created_at=_utc_now(),
        kind="benchmark",
        backend=backend,
        workload_path=workload_path,
        serving_path=serving_path,
        quality_path=quality_path,
        tuner_path="",
        env_json=env,
        metrics_json=asdict(e2e),
        server_json=asdict(server_m),
        quality_json=quality_json,
        artifacts={
            "traces": str(traces_path),
            "metrics": str(metrics_path),
            "server": str(server_path),
            "quality": str(art_dir / "quality.json"),
        },
    )
    _insert_run(db_path, record)

    # Print summary
    table = Table(title=f"Benchmark Result: {run_id}")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("latency_p95", f"{e2e.latency_p95:.3f}s")
    table.add_row("ttft_p95", f"{e2e.ttft_p95:.3f}s")
    table.add_row("tok/s", f"{e2e.tok_s:.2f}")
    table.add_row("timeout_rate", f"{e2e.timeout_rate:.3f}")
    table.add_row("quality_overall", f"{quality_json.get('overall', 1.0):.3f}")
    table.add_row("vram_peak_mb(proxy)", f"{server_m.vram_peak_mb:.0f}")
    console.print(table)
    console.print(f"[green]Saved run:[/green] {run_id}")
    console.print(f"[green]DB:[/green] {db_path}")


@app.command()
def tune(
    workload: str = typer.Option(..., "--workload", help="Path to workload YAML"),
    tuner: str = typer.Option(..., "--tuner", help="Path to tuner YAML"),
    serving: str = typer.Option("", "--serving", help="Path to serving YAML (optional baseline)"),
    quality: str = typer.Option("", "--quality", help="Path to quality YAML (optional)"),
    backend: str = typer.Option("mock", "--backend", help="Backend: mock (CPU) or vllm (future)"),
    out: str = typer.Option("results", "--out", help="Output directory"),
) -> None:
    """
    Multi-objective tuning on mock backend.

    For MVP:
    - Uses random sampling for candidates.
    - Produces a pseudo-Pareto set based on (p95 latency, cost proxy, -tok/s) under constraints and quality gate.
    - Stores candidates in SQLite.
    """
    out_dir = Path(out)
    _ensure_dirs(out_dir)
    db_path = _default_db_path(out_dir)
    _init_db(db_path)

    workload_path = str(Path(workload))
    tuner_path = str(Path(tuner))
    serving_path = str(Path(serving)) if serving else ""
    quality_path = str(Path(quality)) if quality else ""

    spec = _load_workload_spec(workload_path)
    tcfg = load_yaml(tuner_path)

    # Read baseline serving config (optional)
    base_serving = load_yaml(serving_path) if serving_path and Path(serving_path).exists() else {}
    baseline_params = {
        "concurrency": int(base_serving.get("concurrency", spec.concurrency_limit)),
        "max_new_tokens": int(base_serving.get("max_new_tokens", 256)),
        "max_batch_tokens": int(base_serving.get("max_batch_tokens", 2048)),
        "kv_cache_limit_mb": int(base_serving.get("kv_cache_limit_mb", 2048)),
        "max_model_len": int(base_serving.get("max_model_len", spec.max_context_tokens)),
    }

    # Load trace arrivals once
    trace_rows = _read_jsonl_trace(spec.trace_path) if spec.trace_path else []
    arrivals = [
        (
            float(row.get("timestamp_s", 0.0)),
            str(row.get("request_id", f"req_{uuid.uuid4().hex[:8]}")),
            row.get("session_id", None),
            row.get("turn_id", None),
            int(row.get("input_tokens", 64)),
            int(row.get("output_tokens", 128)),
        )
        for row in trace_rows
    ]
    warm = min(spec.warmup_requests, len(arrivals))

    # Quality baseline (optional)
    baseline_quality = QualitySummary(overall=1.0, pass_rate=1.0, by_type={}, details_path="")
    qcfg = None
    eval_pack = None
    if quality_path and Path(quality_path).exists():
        qcfg = _load_quality_cfg(quality_path)
        eval_pack = _load_eval_pack(qcfg["eval_pack"])
        # baseline quality computed from baseline params
        tmp_dir = out_dir / "artifacts" / "_baseline_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        baseline_quality = _evaluate_quality_mock(
            eval_pack, baseline_params, spec.seed, tmp_dir / "baseline_quality_details.json"
        )

    # Candidate budget
    algo = str(tcfg.get("algorithm", "nsga2")).lower()
    seed = int(tcfg.get("seed", spec.seed))
    population = int(tcfg.get("population", 16))
    generations = int(tcfg.get("generations", 5))
    budget = max(1, population * generations)

    constraints_cfg = tcfg.get("constraints", {}) or {}
    max_timeout_rate = float(constraints_cfg.get("max_timeout_rate", 0.05))
    max_vram_mb = float(constraints_cfg.get("max_vram_mb", 1e9))
    min_quality_overall = float(constraints_cfg.get("min_quality_overall", 0.0))

    search_space = tcfg.get("search_space", {}) or {}

    rng = random.Random(seed)

    def sample_param(name: str, spec_: dict[str, Any]) -> Any:
        t = str(spec_.get("type", "int"))
        if t == "int":
            low = int(spec_.get("low"))
            high = int(spec_.get("high"))
            return rng.randint(low, high)
        if t == "float":
            low = float(spec_.get("low"))
            high = float(spec_.get("high"))
            return low + (high - low) * rng.random()
        if t == "choice":
            opts = list(spec_.get("options", []))
            if not opts:
                raise ValueError(f"choice space {name} has no options")
            return rng.choice(opts)
        raise ValueError(f"Unknown param type for {name}: {t}")

    def evaluate_candidate(
        params: dict[str, Any],
    ) -> tuple[E2EMetrics, ServerMetrics, QualitySummary, GateResult]:
        # Ensure required params
        params = dict(params)
        params.setdefault("max_model_len", spec.max_context_tokens)

        all_tr = _simulate_queueing(
            arrivals=arrivals,
            concurrency=int(params.get("concurrency", spec.concurrency_limit)),
            timeout_s=spec.timeout_s,
            params=params,
            seed=seed,
        )
        tr = all_tr[warm:] if warm < len(all_tr) else all_tr
        e2e = _aggregate_e2e(tr)
        srv = _server_metrics_proxy(params)

        if eval_pack is not None:
            # write details per candidate later in orchestrator
            # use placeholder path
            qsum = _evaluate_quality_mock(eval_pack, params, seed, Path(os.devnull))
            gate = _quality_gate(qcfg, qsum, baseline_quality) if qcfg else GateResult(True, {})
        else:
            qsum = QualitySummary(overall=1.0, pass_rate=1.0, by_type={}, details_path="")
            gate = GateResult(True, {})
        return e2e, srv, qsum, gate

    def objectives_from(
        e2e: E2EMetrics, srv: ServerMetrics, qsum: QualitySummary
    ) -> dict[str, float]:
        # cost proxy: inverse throughput + vram penalty
        cost_proxy = (1.0 / max(1e-6, e2e.tok_s)) + (srv.vram_peak_mb / 100_000.0)
        return {
            "p95_latency_s": e2e.latency_p95,
            "cost_proxy": cost_proxy,
            "tok_s": e2e.tok_s,
            "quality_overall": qsum.overall,
        }

    def constraints_from(
        e2e: E2EMetrics, srv: ServerMetrics, qsum: QualitySummary
    ) -> dict[str, float]:
        return {
            "timeout_rate": e2e.timeout_rate,
            "vram_peak_mb": srv.vram_peak_mb,
            "quality_overall": qsum.overall,
        }

    # Run tuning (random sampling for MVP; the YAML can say nsga2 but we still sample)
    run_id = _run_id("tune")
    art_dir = out_dir / "artifacts" / run_id
    art_dir.mkdir(parents=True, exist_ok=True)

    candidates_summary = []
    passed = 0

    for i in range(budget):
        cand_id = f"cand_{i:04d}"
        params = dict(baseline_params)

        # override sampled parameters
        for name, sp in search_space.items():
            params[name] = sample_param(name, sp)

        e2e, srv, qsum, gate = evaluate_candidate(params)

        # constraints
        c = constraints_from(e2e, srv, qsum)
        violates = (
            (c["timeout_rate"] > max_timeout_rate)
            or (c["vram_peak_mb"] > max_vram_mb)
            or (c["quality_overall"] < min_quality_overall)
        )
        passed_gate = gate.passed and (not violates)

        # write per-candidate artifacts
        cand_dir = art_dir / cand_id
        cand_dir.mkdir(parents=True, exist_ok=True)
        _write_json(cand_dir / "params.json", params)
        _write_json(cand_dir / "metrics.json", asdict(e2e))
        _write_json(cand_dir / "server.json", asdict(srv))
        _write_json(cand_dir / "quality.json", asdict(qsum))
        _write_json(cand_dir / "gate.json", asdict(gate))
        _write_json(cand_dir / "constraints.json", c)

        obj = objectives_from(e2e, srv, qsum)
        _insert_candidate(db_path, run_id, cand_id, params, obj, c, passed_gate)

        candidates_summary.append(
            {
                "cand_id": cand_id,
                "passed": passed_gate,
                "objectives": obj,
                "constraints": c,
                "params": params,
            }
        )
        if passed_gate:
            passed += 1

    # Build a simple Pareto set from passed candidates
    passed_cands = [c for c in candidates_summary if c["passed"]]
    pareto = _pareto_front(
        passed_cands, keys_min=["p95_latency_s", "cost_proxy"], keys_max=["tok_s"]
    )

    # pick Top-K recommendation: minimal p95 among pareto
    pareto_sorted = sorted(pareto, key=lambda x: x["objectives"]["p95_latency_s"])
    topk = pareto_sorted[: min(5, len(pareto_sorted))]

    # Save tuning summary
    _write_json(art_dir / "candidates_summary.json", candidates_summary)
    _write_json(art_dir / "pareto.json", pareto_sorted)
    _write_json(art_dir / "topk.json", topk)

    # Save an aggregated run record (use best candidate metrics)
    best = (
        topk[0]
        if topk
        else (
            passed_cands[0]
            if passed_cands
            else (candidates_summary[0] if candidates_summary else None)
        )
    )
    if best is None:
        raise RuntimeError("No candidates evaluated.")

    # best_metrics = best["objectives"]
    # We store as "metrics_json" the best candidate's *E2E metrics-like* fields for convenience
    metrics_json = {
        "latency_p95": best["objectives"]["p95_latency_s"],
        "cost_proxy": best["objectives"]["cost_proxy"],
        "tok_s": best["objectives"]["tok_s"],
        "quality_overall": best["objectives"]["quality_overall"],
        "passed_candidates": passed,
        "budget": budget,
    }
    server_json = {"vram_peak_mb": best["constraints"]["vram_peak_mb"]}
    quality_json = {"overall": best["objectives"]["quality_overall"]}

    env = collect_env_signature().to_json()

    record = RunRecord(
        run_id=run_id,
        created_at=_utc_now(),
        kind="tune",
        backend=backend,
        workload_path=workload_path,
        serving_path=serving_path,
        quality_path=quality_path,
        tuner_path=tuner_path,
        env_json=env,
        metrics_json=metrics_json,
        server_json=server_json,
        quality_json=quality_json,
        artifacts={
            "candidates_summary": str(art_dir / "candidates_summary.json"),
            "pareto": str(art_dir / "pareto.json"),
            "topk": str(art_dir / "topk.json"),
        },
    )
    _insert_run(db_path, record)

    # Print summary
    table = Table(title=f"Tuning Summary: {run_id}")
    table.add_column("field")
    table.add_column("value", justify="right")
    table.add_row("algorithm", algo)
    table.add_row("budget", str(budget))
    table.add_row("passed", str(passed))
    table.add_row("pareto_points", str(len(pareto_sorted)))
    if topk:
        table.add_row("best_p95_latency_s", f"{topk[0]['objectives']['p95_latency_s']:.3f}")
        table.add_row("best_tok_s", f"{topk[0]['objectives']['tok_s']:.2f}")
        table.add_row("best_cost_proxy", f"{topk[0]['objectives']['cost_proxy']:.4f}")
        table.add_row("best_quality", f"{topk[0]['objectives']['quality_overall']:.3f}")
    console.print(table)
    console.print(f"[green]Saved tuning run:[/green] {run_id}")
    console.print(f"[green]DB:[/green] {db_path}")


def _dominates(
    a: dict[str, Any], b: dict[str, Any], keys_min: list[str], keys_max: list[str]
) -> bool:
    """
    a dominates b if:
      - for all min keys: a <= b
      - for all max keys: a >= b
      - and strictly better in at least one objective
    """
    aobj = a["objectives"]
    bobj = b["objectives"]

    not_worse = True
    strictly_better = False

    for k in keys_min:
        if aobj[k] > bobj[k]:
            not_worse = False
            break
        if aobj[k] < bobj[k]:
            strictly_better = True

    if not not_worse:
        return False

    for k in keys_max:
        if aobj[k] < bobj[k]:
            not_worse = False
            break
        if aobj[k] > bobj[k]:
            strictly_better = True

    return not_worse and strictly_better


def _pareto_front(
    cands: list[dict[str, Any]], keys_min: list[str], keys_max: list[str]
) -> list[dict[str, Any]]:
    pareto = []
    for i, a in enumerate(cands):
        dominated = False
        for j, b in enumerate(cands):
            if i == j:
                continue
            if _dominates(b, a, keys_min, keys_max):
                dominated = True
                break
        if not dominated:
            pareto.append(a)
    return pareto


@app.command()
def report(
    run_id: str = typer.Option("", "--run-id", help="Run ID"),
    latest: bool = typer.Option(False, "--latest", help="Use latest run"),
    out: str = typer.Option("reports/report.html", "--out", help="Output HTML path"),
    results_dir: str = typer.Option(
        "results", "--results-dir", help="Results directory that contains runs.sqlite"
    ),
) -> None:
    """
    Generate an HTML report for a run (or latest).
    """
    out_path = Path(out)
    db_path = _default_db_path(Path(results_dir))
    _init_db(db_path)

    rid = run_id
    if latest:
        rid = _get_latest_run_id(db_path) or ""
    if not rid:
        raise typer.BadParameter("Provide --run-id or --latest")

    row = _get_run(db_path, rid)
    if not row:
        raise typer.BadParameter(f"Run not found: {rid}")

    _render_report_html(row, out_path)
    console.print(f"[green]Report written:[/green] {out_path}")


@app.command()
def regress(
    base: str = typer.Option(..., "--base", help="Baseline run id"),
    cand: str = typer.Option(..., "--cand", help="Candidate run id"),
    results_dir: str = typer.Option("results", "--results-dir", help="Results directory"),
    max_p95_regress_pct: float = typer.Option(
        10.0, "--max-p95-regress-pct", help="Fail if P95 worsens by > X%"
    ),
    max_quality_drop: float = typer.Option(
        0.02, "--max-quality-drop", help="Fail if quality drops by > Y (absolute)"
    ),
) -> None:
    """
    Regression check: fail if latency P95 regresses too much or quality drops too much.
    Exits with code 1 on regression.
    """
    db_path = _default_db_path(Path(results_dir))
    _init_db(db_path)

    b = _get_run(db_path, base)
    c = _get_run(db_path, cand)
    if not b or not c:
        raise typer.BadParameter("base/cand run_id not found in DB")

    bm = json.loads(b["metrics_json"])
    cm = json.loads(c["metrics_json"])
    bq = json.loads(b["quality_json"])
    cq = json.loads(c["quality_json"])

    # p95 extraction (benchmark run uses latency_p95; tune run stores p95_latency_s)
    b_p95 = float(bm.get("latency_p95", bm.get("p95_latency_s", bm.get("p95_latency", 0.0))))
    c_p95 = float(cm.get("latency_p95", cm.get("p95_latency_s", cm.get("p95_latency", 0.0))))

    b_qual = float(bq.get("overall", bm.get("quality_overall", 1.0)))
    c_qual = float(cq.get("overall", cm.get("quality_overall", 1.0)))

    regress_pct = 0.0
    if b_p95 > 1e-9:
        regress_pct = (c_p95 - b_p95) / b_p95 * 100.0

    quality_drop = b_qual - c_qual

    passed = True
    reasons = {}

    if regress_pct > max_p95_regress_pct:
        passed = False
        reasons["p95_regress_pct"] = {
            "base": b_p95,
            "cand": c_p95,
            "pct": regress_pct,
            "max": max_p95_regress_pct,
        }

    if quality_drop > max_quality_drop:
        passed = False
        reasons["quality_drop"] = {
            "base": b_qual,
            "cand": c_qual,
            "drop": quality_drop,
            "max": max_quality_drop,
        }

    table = Table(title="Regression Check")
    table.add_column("check")
    table.add_column("value")
    table.add_row("base_run", base)
    table.add_row("cand_run", cand)
    table.add_row("p95_base", f"{b_p95:.3f}s")
    table.add_row("p95_cand", f"{c_p95:.3f}s")
    table.add_row("p95_regress_pct", f"{regress_pct:.2f}%")
    table.add_row("quality_base", f"{b_qual:.3f}")
    table.add_row("quality_cand", f"{c_qual:.3f}")
    table.add_row("quality_drop", f"{quality_drop:.3f}")
    table.add_row("PASS", "YES" if passed else "NO")
    console.print(table)

    if not passed:
        console.print("[red]Regression detected[/red]")
        console.print(json.dumps(reasons, ensure_ascii=False, indent=2))
        raise typer.Exit(code=1)

    console.print("[green]No regression[/green]")


@app.command()
def plan(
    peak_rps: float = typer.Option(..., "--peak-rps", help="Peak requests per second"),
    p95_sla_s: float = typer.Option(2.0, "--p95-sla-s", help="Target P95 SLA (seconds)"),
    avg_output_tokens: int = typer.Option(
        200, "--avg-output-tokens", help="Avg output tokens per request"
    ),
    tok_s_per_gpu: float = typer.Option(
        300.0, "--tok-s-per-gpu", help="Estimated sustainable tok/s per GPU"
    ),
) -> None:
    """
    Simple capacity planner (MVP):
    Given peak RPS and avg output tokens, estimate required GPUs from tok/s capacity.

    This is a placeholder; later you can refine using measured tok/s under target workload and tail risk factors.
    """
    demand_tok_s = peak_rps * avg_output_tokens
    gpus = math.ceil(demand_tok_s / max(1e-9, tok_s_per_gpu))

    table = Table(title="Capacity Planner (MVP)")
    table.add_column("field")
    table.add_column("value", justify="right")
    table.add_row("peak_rps", f"{peak_rps:.2f}")
    table.add_row("avg_output_tokens", str(avg_output_tokens))
    table.add_row("demand_tok_s", f"{demand_tok_s:.2f}")
    table.add_row("tok_s_per_gpu", f"{tok_s_per_gpu:.2f}")
    table.add_row("estimated_gpus", str(gpus))
    table.add_row("p95_sla_s", f"{p95_sla_s:.2f}")
    console.print(table)
