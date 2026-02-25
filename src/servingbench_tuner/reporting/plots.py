from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


@dataclass
class PlotPaths:
    cdf_png: str = ""
    tail_breakdown_png: str = ""
    pareto_png: str = ""
    tail_attribution_png: str = ""
    tail_attribution_json: str = ""  # absolute path for report.py to read


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    v = sorted(float(x) for x in vals)
    if len(v) == 1:
        return v[0]
    q = max(0.0, min(100.0, q))
    idx = (len(v) - 1) * q / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(v) - 1)
    frac = idx - lo
    return v[lo] * (1.0 - frac) + v[hi] * frac


def _rel_plot_path(plots_dir: Path, file_name: str) -> str:
    return f"{plots_dir.name}/{file_name}"


def _make_latency_cdf(traces: list[dict[str, Any]], out_png: Path) -> None:
    lats = [max(0.0, float(t.get("end_s", 0.0)) - float(t.get("arrival_s", 0.0))) for t in traces]
    if not lats:
        return
    xs = sorted(lats)
    n = len(xs)
    ys = [i / n for i in range(1, n + 1)]
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys)
    plt.xlabel("Latency (s)")
    plt.ylabel("CDF")
    plt.title("Request Latency CDF")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def _span(trace: dict[str, Any], key: str, fallback: str) -> float:
    spans = trace.get("spans") or {}
    if key in spans:
        return float(spans.get(key) or 0.0)
    return float(spans.get(fallback) or 0.0)


def _make_tail_breakdown(traces: list[dict[str, Any]], out_png: Path) -> None:
    if not traces:
        return
    q_vals = [_span(t, "server_queue_s", "client_queue_s") for t in traces]
    p_vals = [_span(t, "server_prefill_s", "client_prefill_s") for t in traces]
    d_vals = [_span(t, "server_decode_s", "client_decode_s") for t in traces]

    labels = ["P50", "P95", "P99"]
    q = [_pct(q_vals, 50), _pct(q_vals, 95), _pct(q_vals, 99)]
    p = [_pct(p_vals, 50), _pct(p_vals, 95), _pct(p_vals, 99)]
    d = [_pct(d_vals, 50), _pct(d_vals, 95), _pct(d_vals, 99)]

    x = list(range(len(labels)))
    plt.figure(figsize=(7, 4))
    plt.bar(x, q, label="queue")
    plt.bar(x, p, bottom=q, label="prefill")
    bottoms = [q[i] + p[i] for i in range(len(q))]
    plt.bar(x, d, bottom=bottoms, label="decode")
    plt.xticks(x, labels)
    plt.ylabel("Seconds")
    plt.title("Tail latency breakdown")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def _load_candidate_metrics(artifacts_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fp in sorted(artifacts_dir.glob("candidate_*.json")):
        try:
            c = _read_json(fp)
            ov = c.get("objective_values") or {}
            rows.append(
                {
                    "idx": int(c.get("idx", -1)),
                    "feasible": bool(c.get("feasible", False)),
                    "p95_s": float(ov.get("latency_p95_s", ov.get("p95_s", 0.0)) or 0.0),
                    "tok_s": float(ov.get("tok_s", ov.get("throughput_tok_s", 0.0)) or 0.0),
                    "cost_proxy": float(ov.get("cost_proxy", 0.0) or 0.0),
                }
            )
        except Exception:
            continue
    return rows


def _make_pareto_plot(artifacts_dir: Path, out_png: Path) -> None:
    summ_path = artifacts_dir / "tuning_summary.json"
    if not summ_path.exists():
        return
    summ = _read_json(summ_path)
    pareto_idx = set(int(i) for i in (summ.get("pareto_indices") or []))
    rows = _load_candidate_metrics(artifacts_dir)
    if not rows:
        return

    x_feas = [r["p95_s"] for r in rows if r["feasible"]]
    y_feas = [r["tok_s"] for r in rows if r["feasible"]]
    x_infeas = [r["p95_s"] for r in rows if not r["feasible"]]
    y_infeas = [r["tok_s"] for r in rows if not r["feasible"]]
    x_par = [r["p95_s"] for r in rows if r["idx"] in pareto_idx]
    y_par = [r["tok_s"] for r in rows if r["idx"] in pareto_idx]

    plt.figure(figsize=(6.8, 4.6))
    if x_infeas:
        plt.scatter(x_infeas, y_infeas, marker="x", label="infeasible")
    if x_feas:
        plt.scatter(x_feas, y_feas, label="feasible")
    if x_par:
        plt.scatter(x_par, y_par, s=80, marker="o", facecolors="none", label="pareto")

    if len(x_par) >= 2:
        paired = sorted(zip(x_par, y_par, strict=False), key=lambda t: t[0])
        plt.plot([p[0] for p in paired], [p[1] for p in paired])

    plt.xlabel("P95 latency (s)")
    plt.ylabel("Throughput (tok/s)")
    plt.title("Pareto front")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def _tail_stats_from_traces(traces: list[dict[str, Any]]) -> dict[str, float]:
    if not traces:
        return {"long_ctx_share": 0.0, "multi_turn_share": 0.0, "retry_share": 0.0}
    input_tokens = [float(t.get("input_tokens", 0.0) or 0.0) for t in traces]
    thr = max(1.0, _pct(input_tokens, 75))
    long_share = sum(1 for v in input_tokens if v >= thr) / len(input_tokens)
    multi_turn_share = sum(1 for t in traces if (t.get("turn_id") or 1) > 1) / len(traces)
    retry_share = sum(1 for t in traces if int(t.get("retry_count", 0) or 0) > 0) / len(traces)
    return {
        "long_ctx_share": long_share,
        "multi_turn_share": multi_turn_share,
        "retry_share": retry_share,
    }


def _make_tail_attribution_compare(payload: dict[str, Any], out_png: Path, out_json: Path) -> None:
    rows = payload.get("rows") or []
    if not rows:
        return

    labels = [str(r.get("label", r.get("name", "-"))) for r in rows]
    q = [float(r.get("queue_p95_s", 0.0) or 0.0) for r in rows]
    p = [float(r.get("prefill_p95_s", 0.0) or 0.0) for r in rows]
    d = [float(r.get("decode_p95_s", 0.0) or 0.0) for r in rows]
    retry_share = [100.0 * float(r.get("retry_share", 0.0) or 0.0) for r in rows]
    long_share = [100.0 * float(r.get("long_ctx_share", 0.0) or 0.0) for r in rows]
    multi_share = [100.0 * float(r.get("multi_turn_share", 0.0) or 0.0) for r in rows]

    x = list(range(len(labels)))
    fig, ax1 = plt.subplots(figsize=(8.6, 4.8))
    ax1.bar(x, q, label="queue")
    ax1.bar(x, p, bottom=q, label="prefill")
    bottoms = [q[i] + p[i] for i in range(len(q))]
    ax1.bar(x, d, bottom=bottoms, label="decode")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("P95 component (s)")
    ax1.set_title("Tail latency attribution")
    ax1.grid(True, axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, retry_share, marker="o", label="retry%")
    ax2.plot(x, long_share, marker="s", label="long-ctx%")
    ax2.plot(x, multi_share, marker="^", label="multi-turn%")
    ax2.set_ylabel("Request share (%)")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_all_plots(artifacts_dir: str | Path, out_dir: str | Path) -> PlotPaths:
    artifacts_dir = Path(artifacts_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = PlotPaths()

    # Base traces from repeated benchmark runs (if present)
    trace_files = sorted(artifacts_dir.glob("traces_repeat_*.jsonl"))
    if trace_files:
        traces = _read_jsonl(trace_files[0])
        if traces:
            cdf_png = out_dir / "latency_cdf.png"
            _make_latency_cdf(traces, cdf_png)
            paths.cdf_png = _rel_plot_path(out_dir, cdf_png.name)

            tail_png = out_dir / "tail_breakdown.png"
            _make_tail_breakdown(traces, tail_png)
            paths.tail_breakdown_png = _rel_plot_path(out_dir, tail_png.name)

    # Pareto plot from candidate artifacts
    pareto_png = out_dir / "pareto_front.png"
    _make_pareto_plot(artifacts_dir, pareto_png)
    if pareto_png.exists():
        paths.pareto_png = _rel_plot_path(out_dir, pareto_png.name)

    # Tail attribution compare (preferred) or fallback single-run derivation
    tail_json_in = artifacts_dir / "tail_compare.json"
    if tail_json_in.exists():
        payload = _read_json(tail_json_in)
        tail_png = out_dir / "tail_attribution.png"
        tail_json_out = out_dir / "tail_attribution.json"
        _make_tail_attribution_compare(payload, tail_png, tail_json_out)
        if tail_png.exists():
            paths.tail_attribution_png = _rel_plot_path(out_dir, tail_png.name)
            paths.tail_attribution_json = str(tail_json_out)
    elif trace_files:
        traces = _read_jsonl(trace_files[0])
        s = _tail_stats_from_traces(traces)
        payload = {
            "rows": [
                {
                    "label": "Current run",
                    "queue_p95_s": _pct(
                        [_span(t, "server_queue_s", "client_queue_s") for t in traces], 95
                    ),
                    "prefill_p95_s": _pct(
                        [_span(t, "server_prefill_s", "client_prefill_s") for t in traces], 95
                    ),
                    "decode_p95_s": _pct(
                        [_span(t, "server_decode_s", "client_decode_s") for t in traces], 95
                    ),
                    **s,
                }
            ]
        }
        tail_png = out_dir / "tail_attribution.png"
        tail_json_out = out_dir / "tail_attribution.json"
        _make_tail_attribution_compare(payload, tail_png, tail_json_out)
        if tail_png.exists():
            paths.tail_attribution_png = _rel_plot_path(out_dir, tail_png.name)
            paths.tail_attribution_json = str(tail_json_out)

    return paths
