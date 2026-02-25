from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from servingbench_tuner.quality.enterprise_gate import (  # noqa: E402
    EnterpriseQualityGateConfig,
    evaluate_enterprise_quality_gate,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        required=True,
        help="Path to real_validation_summary.json or comparison_summary.json",
    )
    p.add_argument("--baseline-method", default="Baseline")
    p.add_argument("--method", default="NSGA2-best")
    p.add_argument(
        "--config", default=str(REPO_ROOT / "configs" / "quality" / "enterprise_gate.json")
    )
    return p.parse_args()


def _load_cfg(path: str) -> EnterpriseQualityGateConfig:
    p = Path(path)
    if not p.exists():
        return EnterpriseQualityGateConfig()
    d = json.loads(p.read_text(encoding="utf-8"))
    return EnterpriseQualityGateConfig(
        **{k: v for k, v in d.items() if hasattr(EnterpriseQualityGateConfig, k)}
    )


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = data.get("rows") or data.get("results") or []
    if not rows:
        raise SystemExit(f"no rows/results found in {args.input}")

    by_method = {r.get("method"): r for r in rows if isinstance(r, dict)}
    if args.baseline_method not in by_method:
        raise SystemExit(f"baseline method not found: {args.baseline_method}")
    if args.method not in by_method:
        raise SystemExit(f"target method not found: {args.method}")

    baseline = by_method[args.baseline_method]
    target = by_method[args.method]
    cfg = _load_cfg(args.config)
    res = evaluate_enterprise_quality_gate(target, baseline, cfg)

    if res.passed:
        print("✅ Enterprise quality breakdown gate passed")
        print(f"target={args.method} baseline={args.baseline_method}")
    else:
        print("❌ Enterprise quality breakdown gate failed")
        print(f"target={args.method} baseline={args.baseline_method}")
        for k, v in res.reasons.items():
            print(f"- {k}: {v}")

    print("ratios:")
    for k, v in res.ratios.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
