from __future__ import annotations

import argparse
import json
from pathlib import Path


def _f(x, d=0.0):
    try:
        return float(x)
    except Exception:
        return d


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inspect formal comparison summary and print a production-style verdict."
    )
    ap.add_argument("--summary", default="results/formal_comparison/comparison_summary.json")
    args = ap.parse_args()

    p = Path(args.summary)
    if not p.exists():
        raise SystemExit(f"not found: {p}")

    d = load(p)
    rows = d.get("rows") or []
    if not rows:
        raise SystemExit("No rows found in comparison_summary.json")

    by_method = {r.get("method"): r for r in rows if isinstance(r, dict)}
    base = by_method.get("Baseline")
    rnd = by_method.get("Random-best")
    ns = by_method.get("NSGA2-best")

    print(f"workload: {d.get('workload')}")
    print(f"seed: {d.get('seed')}")
    print(f"nsga_algo: {d.get('nsga_algo')}")
    print()

    print("=== head-to-head (vs Baseline) ===")
    if base:
        print(
            f"Baseline   P95={_f(base.get('p95_s')):.3f}s TTFT={_f(base.get('ttft_p95_s')):.3f}s tok/s={_f(base.get('tok_s')):.1f} VRAM={_f(base.get('vram_peak_mb')):.0f}MB quality={_f(base.get('quality')):.3f}"
        )
    if rnd:
        print(
            f"Random-best P95={_f(rnd.get('p95_s')):.3f}s TTFT={_f(rnd.get('ttft_p95_s')):.3f}s tok/s={_f(rnd.get('tok_s')):.1f} VRAM={_f(rnd.get('vram_peak_mb')):.0f}MB quality={_f(rnd.get('quality')):.3f}"
        )
    if ns:
        print(
            f"NSGA2-best  P95={_f(ns.get('p95_s')):.3f}s TTFT={_f(ns.get('ttft_p95_s')):.3f}s tok/s={_f(ns.get('tok_s')):.1f} VRAM={_f(ns.get('vram_peak_mb')):.0f}MB quality={_f(ns.get('quality')):.3f}"
        )
    print()

    if base and ns:
        print("=== deltas (NSGA2-best vs Baseline) ===")

        def pct(new, old):
            if old == 0:
                return 0.0
            return (new - old) / old * 100.0

        print(
            f"P95      {pct(_f(ns.get('p95_s')), _f(base.get('p95_s'))):+.1f}%  (lower is better)"
        )
        print(
            f"TTFT P95 {pct(_f(ns.get('ttft_p95_s')), _f(base.get('ttft_p95_s'))):+.1f}%  (lower is better)"
        )
        print(
            f"tok/s    {pct(_f(ns.get('tok_s')), _f(base.get('tok_s'))):+.1f}%  (higher is better)"
        )
        print(
            f"VRAM     {pct(_f(ns.get('vram_peak_mb')), _f(base.get('vram_peak_mb'))):+.1f}%  (lower is better)"
        )
        print(
            f"Quality  {pct(_f(ns.get('quality')), _f(base.get('quality'))):+.1f}%  (higher is better)"
        )
        print()

        print("=== recommendation ===")
        wl = str(d.get("workload") or "")
        if wl == "agent_like":
            if (
                _f(ns.get("p95_s")) <= _f(base.get("p95_s")) * 1.05
                and _f(ns.get("ttft_p95_s")) <= _f(base.get("ttft_p95_s")) * 1.05
            ):
                print("✅ NSGA2-best can be promoted for agent_like (interactive-friendly).")
            else:
                print(
                    "⚠️ NSGA2-best improves throughput/cost but still hurts interactivity; tighten P95/TTFT objective/constraints and rerun."
                )
        else:
            if (
                _f(ns.get("tok_s")) > _f(base.get("tok_s"))
                and _f(ns.get("quality")) >= _f(base.get("quality")) * 0.98
            ):
                print("✅ NSGA2-best is a strong candidate for production trial.")
            else:
                print(
                    "⚠️ Keep Baseline as production default; use NSGA2 result as exploration signal."
                )
        print()

    print("=== top params ===")
    for label, row in [("Baseline", base), ("Random-best", rnd), ("NSGA2-best", ns)]:
        if row:
            print(f"[{label}]")
            print(json.dumps(row.get("params") or {}, indent=2, ensure_ascii=False))
            print()


if __name__ == "__main__":
    main()
