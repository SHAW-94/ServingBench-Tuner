#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load_json(p: str | Path) -> dict[str, Any]:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _guess_avg_output_tokens(workload_name: str) -> int:
    # conservative defaults; can be overridden with CLI
    wl = (workload_name or "").lower()
    if "short" in wl:
        return 96
    if "agent" in wl:
        return 224
    if "long" in wl:
        return 512
    return 192


def _rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = summary.get("rows")
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


def main() -> None:
    ap = argparse.ArgumentParser(description="Capacity planner from formal comparison metrics")
    ap.add_argument("--summary", default="results/formal_comparison/comparison_summary.json")
    ap.add_argument("--peak-rps", type=float, default=20.0, help="Target peak requests/sec")
    ap.add_argument("--sla-p95", type=float, default=3.0, help="P95 SLA in seconds")
    ap.add_argument("--gpu-vram-mb", type=float, default=24576.0)
    ap.add_argument("--gpu-hourly-cost", type=float, default=1.5)
    ap.add_argument("--safety-util", type=float, default=0.75, help="Capacity safety utilization")
    ap.add_argument(
        "--avg-output-tokens", type=int, default=0, help="Override avg output tokens per request"
    )
    ap.add_argument("--out-json", default="results/formal_comparison/capacity_plan.json")
    ap.add_argument("--out-md", default="results/formal_comparison/capacity_plan.md")
    args = ap.parse_args()

    summary = _load_json(args.summary)
    workload = str(summary.get("workload", "unknown"))
    avg_out = args.avg_output_tokens or _guess_avg_output_tokens(workload)

    plans = []
    for r in _rows(summary):
        tok_s = float(r.get("tok_s", 0.0) or 0.0)
        p95_s = float(r.get("p95_s", 1e9) or 1e9)
        vram = float(r.get("vram_peak_mb", 1e9) or 1e9)
        quality = float(r.get("quality", 0.0) or 0.0)
        timeout_rate = float(r.get("timeout_rate", 1.0) or 1.0)
        if tok_s <= 0:
            continue

        est_rps_per_gpu = (tok_s / max(avg_out, 1)) * args.safety_util
        meets_sla = p95_s <= args.sla_p95
        fits_vram = vram <= args.gpu_vram_mb * 0.95
        n_gpu = math.ceil(args.peak_rps / max(est_rps_per_gpu, 1e-9))
        hourly = n_gpu * args.gpu_hourly_cost
        monthly = hourly * 24 * 30
        params = r.get("params") if isinstance(r.get("params"), dict) else {}
        rec = {
            "method": r.get("method"),
            "algo": r.get("algo"),
            "meets_sla": meets_sla,
            "fits_vram_95pct": fits_vram,
            "quality": quality,
            "timeout_rate": timeout_rate,
            "p95_s": p95_s,
            "ttft_p95_s": float(r.get("ttft_p95_s", 0.0) or 0.0),
            "tok_s": tok_s,
            "vram_peak_mb": vram,
            "est_rps_per_gpu": est_rps_per_gpu,
            "target_peak_rps": args.peak_rps,
            "recommended_gpu_count": n_gpu,
            "recommended_concurrency_limit": params.get("concurrency_limit"),
            "hourly_cost_est": hourly,
            "monthly_cost_est": monthly,
            "params": params,
        }
        # ranking key: must meet SLA first, then lower gpu count, then lower monthly cost, then better p95
        rank_key = (
            0 if meets_sla else 1,
            0 if fits_vram else 1,
            n_gpu,
            monthly,
            p95_s,
            -tok_s,
        )
        rec["_rank_key"] = rank_key
        plans.append(rec)

    plans.sort(key=lambda x: x["_rank_key"])
    for p in plans:
        p.pop("_rank_key", None)

    payload = {
        "workload": workload,
        "avg_output_tokens_assumed": avg_out,
        "target_peak_rps": args.peak_rps,
        "sla_p95_s": args.sla_p95,
        "gpu_vram_mb": args.gpu_vram_mb,
        "gpu_hourly_cost": args.gpu_hourly_cost,
        "safety_util": args.safety_util,
        "plans": plans,
        "top_recommendation": plans[0] if plans else None,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md = []
    md.append("# Capacity Planner")
    md.append("")
    md.append(f"- workload: `{workload}`")
    md.append(f"- target_peak_rps: `{args.peak_rps}`")
    md.append(f"- SLA P95: `{args.sla_p95}s`")
    md.append(f"- assumed avg output tokens/request: `{avg_out}`")
    md.append("")
    md.append(
        "| rank | method | meets SLA | est rps/gpu | GPUs | hourly $ | monthly $ | P95 | TTFT | tok/s | VRAM MB |"
    )
    md.append("|---:|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, p in enumerate(plans, 1):
        md.append(
            f"| {i} | {p.get('method')} | {'Y' if p.get('meets_sla') else 'N'} | "
            f"{p.get('est_rps_per_gpu', 0):.2f} | {p.get('recommended_gpu_count')} | "
            f"{p.get('hourly_cost_est', 0):.2f} | {p.get('monthly_cost_est', 0):.0f} | "
            f"{p.get('p95_s', 0):.3f} | {p.get('ttft_p95_s', 0):.3f} | {p.get('tok_s', 0):.1f} | {p.get('vram_peak_mb', 0):.0f} |"
        )

    if payload["top_recommendation"]:
        top = payload["top_recommendation"]
        md.append("")
        md.append("## Recommendation card")
        md.append("")
        md.append(f"- 推荐方法：**{top.get('method')}**")
        md.append(
            f"- 建议 GPU 数：**{top.get('recommended_gpu_count')}**（峰值 {args.peak_rps} RPS）"
        )
        md.append(f"- 建议并发上限：**{top.get('recommended_concurrency_limit')}**")
        md.append(
            f"- 预计成本：**${top.get('hourly_cost_est'):.2f}/h**（约 **${top.get('monthly_cost_est'):.0f}/月**）"
        )
        md.append(
            f"- 风险：{'满足SLA' if top.get('meets_sla') else 'P95未满足SLA，需降载或加卡'}；VRAM={'安全' if top.get('fits_vram_95pct') else '接近上限'}"
        )

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"✅ wrote {out_json}")
    print(f"✅ wrote {out_md}")


if __name__ == "__main__":
    main()
