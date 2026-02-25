#!/usr/bin/env bash
set -euo pipefail

WORKLOAD="${1:-agent_like}"
SEED="${2:-42}"

python scripts/run_formal_comparison.py --workload "$WORKLOAD" --seed "$SEED"
python scripts/postprocess_formal_comparison.py --write-readme-snippet || true
python scripts/inspect_formal_comparison.py || true
python scripts/check_regression_gate.py || true

echo
echo "Optional real-backend validation (vLLM / OpenAI-compatible):"
echo "  python scripts/run_real_backend_validation.py --model <your-model> --workload $WORKLOAD --stream"
echo "  python scripts/check_quality_breakdown_gate.py --input results/real_validation/real_validation_summary.json"
