#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = summary.get("rows")
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


def _ci95(vals: list[float]) -> tuple[float, float, float]:
    m = mean(vals)
    if len(vals) <= 1:
        return m, 0.0, 0.0
    s = stdev(vals)
    ci = 1.96 * s / math.sqrt(len(vals))
    return m, s, ci


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Repeat formal comparison across seeds and aggregate mean/std/CI"
    )
    ap.add_argument("--workload", default="agent_like")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed-start", type=int, default=42)
    ap.add_argument("--script", default="scripts/run_formal_comparison.py")
    ap.add_argument("--out-dir", default="results/formal_comparison/stability")
    ap.add_argument(
        "--skip-run", action="store_true", help="Only aggregate existing snapshots in out-dir/runs"
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_run:
        for i in range(args.repeats):
            seed = args.seed_start + i
            cmd = [sys.executable, args.script, "--workload", args.workload, "--seed", str(seed)]
            print("$", " ".join(cmd))
            subprocess.run(cmd, check=True)
            src = Path("results/formal_comparison/comparison_summary.json")
            if not src.exists():
                raise SystemExit("comparison_summary.json not found after run")
            dst = runs_dir / f"comparison_summary_seed{seed}.json"
            shutil.copy2(src, dst)
            print(f"  copied -> {dst}")

    files = sorted(runs_dir.glob("comparison_summary_seed*.json"))
    if not files:
        raise SystemExit(f"No summaries found in {runs_dir}")

    all_runs = [_read_json(p) for p in files]
    methods = sorted({r.get("method") for s in all_runs for r in _rows(s) if r.get("method")})
    metric_keys = [
        "p95_s",
        "ttft_p95_s",
        "tok_s",
        "vram_peak_mb",
        "quality",
        "tail_amp",
        "cost_proxy",
        "timeout_rate",
    ]

    agg: dict[str, Any] = {
        "workload": args.workload,
        "n_runs": len(all_runs),
        "seeds": [args.seed_start + i for i in range(len(files))],
        "methods": {},
    }

    for m in methods:
        rows_m = [next((r for r in _rows(s) if r.get("method") == m), None) for s in all_runs]
        rows_m = [r for r in rows_m if r]
        if not rows_m:
            continue
        mrec: dict[str, Any] = {"n": len(rows_m), "metrics": {}}
        for k in metric_keys:
            vals = [float(r.get(k)) for r in rows_m if isinstance(r.get(k), (int, float))]
            if not vals:
                continue
            mu, sd, ci = _ci95(vals)
            mrec["metrics"][k] = {"mean": mu, "std": sd, "ci95": ci}
        # include one example params for audit
        mrec["example_params"] = rows_m[0].get("params")
        agg["methods"][m] = mrec

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stability_summary.json").write_text(
        json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = []
    md.append("# Formal Comparison Stability Summary")
    md.append("")
    md.append(f"- workload: `{args.workload}`")
    md.append(f"- runs: `{len(files)}`")
    md.append("")
    md.append("## Mean ± std (95% CI)")
    md.append("")
    md.append(
        "| method | P95 (s) | TTFT P95 (s) | tok/s | VRAM MB | quality | tail_amp | cost proxy |"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for method, rec in agg["methods"].items():
        met = rec["metrics"]

        def fmt(name: str, digits: int = 3) -> str:
            x = met.get(name)
            if not x:
                return "-"
            return f"{x['mean']:.{digits}f} ± {x['std']:.{digits}f} (±{x['ci95']:.{digits}f})"

        md.append(
            "| "
            + method
            + " | "
            + fmt("p95_s", 3)
            + " | "
            + fmt("ttft_p95_s", 3)
            + " | "
            + fmt("tok_s", 1)
            + " | "
            + fmt("vram_peak_mb", 0)
            + " | "
            + fmt("quality", 4)
            + " | "
            + fmt("tail_amp", 3)
            + " | "
            + fmt("cost_proxy", 4)
            + " |"
        )
    (out_dir / "stability_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"✅ wrote {out_dir / 'stability_summary.json'}")
    print(f"✅ wrote {out_dir / 'stability_summary.md'}")


if __name__ == "__main__":
    main()
