from __future__ import annotations

import argparse
import json
from pathlib import Path

from servingbench_tuner.reporting.report import RenderReportConfig, render_report


def _find_latest_nsga_artifact() -> Path | None:
    root = Path("results/artifacts")
    if not root.exists():
        return None
    cands = sorted(
        [p for p in root.iterdir() if p.is_dir() and "nsga2" in p.name],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None


def _write_readme_snippet(summary_path: Path, out_path: Path) -> None:
    d = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = d.get("rows") or []
    if not rows:
        return
    by = {r.get("method"): r for r in rows if isinstance(r, dict)}
    order = ["Baseline", "Random-best", "NSGA2-best"]

    lines = []
    lines.append("## Formal Comparison Snapshot")
    lines.append("")
    lines.append(
        f"- workload: `{d.get('workload')}`  | seed: `{d.get('seed')}`  | nsga_algo: `{d.get('nsga_algo')}`"
    )
    lines.append("")
    lines.append(
        "| Method | Feasible | Quality | P95(s) | TTFT P95(s) | tok/s | VRAM(MB) | Tail | Cost |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k in order:
        r = by.get(k)
        if not r:
            continue
        lines.append(
            "| {m} | {f} | {q:.3f} | {p95:.3f} | {ttft:.3f} | {tok:.1f} | {vram:.0f} | {tail:.2f} | {cost:.4f} |".format(
                m=k,
                f="Y" if r.get("is_feasible") else "N",
                q=float(r.get("quality", 0)),
                p95=float(r.get("p95_s", 0)),
                ttft=float(r.get("ttft_p95_s", 0)),
                tok=float(r.get("tok_s", 0)),
                vram=float(r.get("vram_peak_mb", 0)),
                tail=float(r.get("tail_amp", 0)),
                cost=float(r.get("cost_proxy", 0)),
            )
        )
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/run_formal_comparison.py --workload agent_like")
    lines.append("python scripts/postprocess_formal_comparison.py")
    lines.append("```")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Re-render formal comparison reports and generate README snippet."
    )
    ap.add_argument(
        "--artifact-dir",
        default="",
        help="results/artifacts/<formal_nsga2_*>, auto-detect if empty",
    )
    ap.add_argument("--reports-dir", default="reports/formal_comparison")
    ap.add_argument("--summary", default="results/formal_comparison/comparison_summary.json")
    ap.add_argument("--write-readme-snippet", action="store_true")
    args = ap.parse_args()

    art = Path(args.artifact_dir) if args.artifact_dir else _find_latest_nsga_artifact()
    if art is None or not art.exists():
        raise SystemExit("Cannot find NSGA artifact dir under results/artifacts")

    out = render_report(
        art, out_dir=Path(args.reports_dir), cfg=RenderReportConfig(out_dir=args.reports_dir)
    )
    print("report_md :", out.get("report_md"))
    print("report_html:", out.get("report_html"))

    if args.write_readme_snippet:
        sp = Path(args.summary)
        if sp.exists():
            outp = Path("results/formal_comparison/README_top_snippet.md")
            _write_readme_snippet(sp, outp)
            print("readme snippet:", outp)
        else:
            print("summary not found, skip README snippet:", sp)


if __name__ == "__main__":
    main()
