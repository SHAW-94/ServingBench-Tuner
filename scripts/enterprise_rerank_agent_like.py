#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(p: str | Path) -> dict[str, Any]:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _find_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = summary.get("rows")
    if isinstance(rows, list):
        return [r for r in rows if isinstance(r, dict)]
    return []


def _score_row(r: dict[str, Any], baseline: dict[str, Any]) -> float:
    # Lower is better. Agent-like: P95 / TTFT / tail prioritized.
    p95 = float(r.get("p95_s", 1e9) or 1e9)
    ttft = float(r.get("ttft_p95_s", 1e9) or 1e9)
    tail = float(r.get("tail_amp", 9e9) or 9e9)
    cost = float(r.get("cost_proxy", 1e9) or 1e9)
    toks = float(r.get("tok_s", 0.0) or 0.0)

    bp95 = max(float(baseline.get("p95_s", p95) or p95), 1e-6)
    bttft = max(float(baseline.get("ttft_p95_s", ttft) or ttft), 1e-6)
    btail = max(float(baseline.get("tail_amp", tail) or tail), 1e-6)
    bcost = max(float(baseline.get("cost_proxy", cost) or cost), 1e-6)
    btoks = max(float(baseline.get("tok_s", toks) or max(toks, 1.0)), 1e-6)

    # Primary: p95/ttft/tail. Secondary: cost. Throughput as small bonus (negative term).
    score = (
        0.45 * (p95 / bp95)
        + 0.35 * (ttft / bttft)
        + 0.10 * (tail / btail)
        + 0.08 * (cost / bcost)
        - 0.08 * (toks / btoks)
    )
    return float(score)


def _apply_strict_agent_constraints(
    r: dict[str, Any], baseline: dict[str, Any]
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not bool(r.get("is_feasible", False)):
        reasons.append("base_feasible=false")

    p95 = float(r.get("p95_s", 1e9) or 1e9)
    ttft = float(r.get("ttft_p95_s", 1e9) or 1e9)
    quality = float(r.get("quality", 0.0) or 0.0)
    tail = float(r.get("tail_amp", 9e9) or 9e9)
    timeout_rate = float(r.get("timeout_rate", 1.0) or 1.0)
    vram = float(r.get("vram_peak_mb", 1e9) or 1e9)

    bp95 = float(baseline.get("p95_s", p95) or p95)
    bttft = float(baseline.get("ttft_p95_s", ttft) or ttft)
    bq = float(baseline.get("quality", quality) or quality)
    btail = float(baseline.get("tail_amp", tail) or tail)

    # stricter interaction-friendly constraints
    if p95 > bp95 * 1.05:
        reasons.append(f"p95>{bp95 * 1.05:.3f}")
    if ttft > bttft * 1.05:
        reasons.append(f"ttft>{bttft * 1.05:.3f}")
    if bq > 0 and quality < bq * 0.99:
        reasons.append(f"quality<{bq * 0.99:.4f}")
    if tail > max(2.2, btail * 1.05):
        reasons.append("tail_amp_too_high")
    if timeout_rate > 0.01:
        reasons.append("timeout_rate>1%")
    if vram > 21000:
        reasons.append("vram>21000MB")

    return (len(reasons) == 0), reasons


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Enterprise reranking for agent_like (P95/TTFT first)."
    )
    ap.add_argument("--summary", default="results/formal_comparison/comparison_summary.json")
    ap.add_argument(
        "--out-json", default="results/formal_comparison/enterprise_agent_like_recommendation.json"
    )
    ap.add_argument(
        "--out-md", default="results/formal_comparison/enterprise_agent_like_recommendation.md"
    )
    args = ap.parse_args()

    summary = _load_json(args.summary)
    rows = _find_rows(summary)
    if not rows:
        raise SystemExit("No rows[] found in comparison_summary.json")

    baseline = next(
        (r for r in rows if str(r.get("method", "")).lower().startswith("baseline")), rows[0]
    )

    scored: list[dict[str, Any]] = []
    for r in rows:
        ok, why = _apply_strict_agent_constraints(r, baseline)
        rr = dict(r)
        rr["enterprise_agent_like_feasible"] = ok
        rr["enterprise_reasons"] = why
        rr["enterprise_score"] = _score_row(rr, baseline)
        scored.append(rr)

    feasible = [r for r in scored if r["enterprise_agent_like_feasible"]]
    feasible.sort(key=lambda x: x["enterprise_score"])
    scored.sort(key=lambda x: x["enterprise_score"])

    top = feasible[0] if feasible else scored[0]
    payload = {
        "workload": summary.get("workload", "agent_like"),
        "policy": "enterprise_agent_like_v1",
        "baseline_method": baseline.get("method"),
        "strict_feasible_count": len(feasible),
        "total_rows": len(rows),
        "top_recommendation": top,
        "all_ranked": scored,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md = []
    md.append("# Enterprise Agent-like Re-ranking (interaction-first)")
    md.append("")
    md.append(f"- workload: `{payload['workload']}`")
    md.append(f"- strict feasible: `{len(feasible)}/{len(rows)}`")
    md.append("")
    md.append("## Top recommendation")
    md.append("")
    md.append(f"- method: **{top.get('method')}**")
    md.append(f"- enterprise_score: `{top.get('enterprise_score'):.4f}` (lower is better)")
    md.append(
        f"- P95: `{top.get('p95_s'):.3f}s` | TTFT P95: `{top.get('ttft_p95_s'):.3f}s` | tok/s: `{top.get('tok_s'):.1f}`"
    )
    md.append(
        f"- VRAM: `{top.get('vram_peak_mb'):.0f}MB` | quality: `{top.get('quality'):.4f}` | tail_amp: `{top.get('tail_amp'):.3f}`"
    )
    if top.get("enterprise_reasons"):
        md.append(f"- notes: `{'; '.join(top['enterprise_reasons'])}`")
    md.append("")
    md.append("## Ranked rows")
    md.append("")
    md.append(
        "| rank | method | strict_feasible | P95 | TTFT | tok/s | VRAM MB | quality | tail_amp | score |"
    )
    md.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(scored, 1):
        md.append(
            f"| {i} | {r.get('method')} | {'Y' if r.get('enterprise_agent_like_feasible') else 'N'} | "
            f"{float(r.get('p95_s', 0)):.3f} | {float(r.get('ttft_p95_s', 0)):.3f} | "
            f"{float(r.get('tok_s', 0)):.1f} | {float(r.get('vram_peak_mb', 0)):.0f} | "
            f"{float(r.get('quality', 0)):.4f} | {float(r.get('tail_amp', 0)):.3f} | {float(r.get('enterprise_score', 0)):.4f} |"
        )
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"✅ wrote {out_json}")
    print(f"✅ wrote {out_md}")


if __name__ == "__main__":
    main()
