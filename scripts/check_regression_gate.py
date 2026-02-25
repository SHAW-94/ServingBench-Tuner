from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def f(x, d=0.0):
    try:
        return float(x)
    except Exception:
        return d


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fail CI if formal comparison regresses beyond thresholds."
    )
    ap.add_argument("--summary", default="results/formal_comparison/comparison_summary.json")
    ap.add_argument("--target", default="NSGA2-best", help="method row to validate")
    ap.add_argument("--baseline", default="Baseline")
    ap.add_argument(
        "--max-p95-regression",
        type=float,
        default=0.10,
        help="allowed P95 increase ratio vs baseline",
    )
    ap.add_argument(
        "--max-ttft-regression",
        type=float,
        default=0.10,
        help="allowed TTFT increase ratio vs baseline",
    )
    ap.add_argument(
        "--min-quality-ratio",
        type=float,
        default=0.98,
        help="min target_quality / baseline_quality",
    )
    ap.add_argument("--max-vram-mb", type=float, default=21000.0)
    ap.add_argument("--max-timeout-rate", type=float, default=0.02)
    args = ap.parse_args()

    p = Path(args.summary)
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = d.get("rows") or []
    by = {r.get("method"): r for r in rows if isinstance(r, dict)}
    base = by.get(args.baseline)
    tgt = by.get(args.target)
    if not base or not tgt:
        print("❌ Missing baseline or target row in summary")
        sys.exit(2)

    failures: list[str] = []
    if f(base.get("p95_s")) > 0 and f(tgt.get("p95_s")) > f(base.get("p95_s")) * (
        1 + args.max_p95_regression
    ):
        failures.append(
            f"P95 regressed too much: {f(tgt.get('p95_s')):.3f}s vs baseline {f(base.get('p95_s')):.3f}s"
        )
    if f(base.get("ttft_p95_s")) > 0 and f(tgt.get("ttft_p95_s")) > f(base.get("ttft_p95_s")) * (
        1 + args.max_ttft_regression
    ):
        failures.append(
            f"TTFT regressed too much: {f(tgt.get('ttft_p95_s')):.3f}s vs baseline {f(base.get('ttft_p95_s')):.3f}s"
        )
    if (
        f(base.get("quality")) > 0
        and f(tgt.get("quality")) < f(base.get("quality")) * args.min_quality_ratio
    ):
        failures.append(
            f"Quality below threshold: {f(tgt.get('quality')):.3f} vs baseline {f(base.get('quality')):.3f}"
        )
    if f(tgt.get("vram_peak_mb")) > args.max_vram_mb:
        failures.append(
            f"VRAM too high: {f(tgt.get('vram_peak_mb')):.0f}MB > {args.max_vram_mb:.0f}MB"
        )
    if f(tgt.get("timeout_rate")) > args.max_timeout_rate:
        failures.append(
            f"Timeout rate too high: {f(tgt.get('timeout_rate')):.3f} > {args.max_timeout_rate:.3f}"
        )
    if not bool(tgt.get("is_feasible", False)):
        failures.append("Target row is not feasible.")

    if failures:
        print("❌ Regression gate failed")
        for m in failures:
            print("-", m)
        sys.exit(1)

    print("✅ Regression gate passed")
    print(f"target={args.target} baseline={args.baseline}")


if __name__ == "__main__":
    main()
