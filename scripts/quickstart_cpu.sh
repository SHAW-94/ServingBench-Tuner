#!/usr/bin/env bash
set -euo pipefail

# quickstart_cpu.sh
# No-GPU quickstart: create minimal configs + run mock backend pipeline (if CLI exists).
#
# Usage:
#   ./scripts/quickstart_cpu.sh
#
# What it does:
# 1) Ensures you are at repo root (pyproject.toml exists)
# 2) Creates minimal quickstart configs under configs/ (only if missing)
# 3) Generates a tiny synthetic "public workload" trace under data/workloads/
# 4) Runs:
#    - benchmark (mock backend)
#    - tune (mock backend; small budget)
#    - report (for the best/top run)
#
# Notes:
# - This script assumes your CLI module exists: `python -m servingbench_tuner.cli`.
# - If CLI is not implemented yet, it will print instructions and exit 0.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ ! -f "pyproject.toml" ]]; then
  echo "ERROR: pyproject.toml not found. Run this from repo root."
  exit 1
fi

mkdir -p configs/workload configs/quality configs/tuner configs/experiments configs/serving
mkdir -p data/workloads data/eval_pack results/artifacts reports

# 1) Create minimal eval pack (tiny; replace with your 80~200 pack later)
EVAL_PATH="data/eval_pack/quickstart_eval.jsonl"
if [[ ! -f "${EVAL_PATH}" ]]; then
  cat > "${EVAL_PATH}" <<'EOF'
{"id":"qa_1","type":"closed_qa","prompt":"1+1=?","expected":"2"}
{"id":"json_1","type":"json_schema","prompt":"Return JSON: {\"name\": string, \"age\": integer}. Use name=Alice age=30.","schema":{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"],"additionalProperties":false}}
{"id":"sum_1","type":"judge","prompt":"Summarize: ServingBench-Tuner measures latency and cost under quality constraints.","rubric":"Clarity and faithfulness. Score 1-5."}
EOF
  echo "[+] Wrote ${EVAL_PATH}"
fi

# 2) Minimal quality config
QUALITY_CFG="configs/quality/quickstart.yaml"
if [[ ! -f "${QUALITY_CFG}" ]]; then
  cat > "${QUALITY_CFG}" <<EOF
eval_pack: "${EVAL_PATH}"
evaluators:
  - name: exact_match
    for_types: ["closed_qa"]
  - name: json_schema
    for_types: ["json_schema"]
  - name: llm_judge
    for_types: ["judge"]
    # For quickstart, you can set judge_backend=mock so CI/CPU works.
    judge_backend: "mock"
    temperature: 0
    seed: 42

gate:
  min_overall: 0.80
  min_relative_to_baseline: 0.98
  hard_fail_tasks: ["json_schema"]
EOF
  echo "[+] Wrote ${QUALITY_CFG}"
fi

# 3) Create a tiny "public workload" trace (synthetic)
python scripts/make_public_workload.py \
  --out-dir data/workloads \
  --profiles quickstart \
  --n-requests 30 \
  --seed 42 >/dev/null

WORKLOAD_CFG="configs/workload/quickstart_cpu.yaml"
if [[ ! -f "${WORKLOAD_CFG}" ]]; then
  cat > "${WORKLOAD_CFG}" <<'EOF'
name: "quickstart_cpu"
seed: 42
arrival:
  mode: replay
  trace_path: "data/workloads/quickstart_trace.jsonl"
concurrency_limit: 8
timeout_s: 10.0
retries: 0
length_dist:
  prompt:
    type: empirical
    path: "data/workloads/quickstart_len_prompt.json"
  output:
    type: empirical
    path: "data/workloads/quickstart_len_output.json"
sessions:
  enabled: true
  turns: 3
  context_growth: agent_like
  max_context_tokens: 2048
warmup_requests: 5
duration_s: 30
EOF
  echo "[+] Wrote ${WORKLOAD_CFG}"
fi

# 4) Minimal tuner config (small budget)
TUNER_CFG="configs/tuner/quickstart_nsga2.yaml"
if [[ ! -f "${TUNER_CFG}" ]]; then
  cat > "${TUNER_CFG}" <<'EOF'
algorithm: nsga2
seed: 42
population: 8
generations: 3
objectives:
  - name: p95_latency_s
    direction: minimize
  - name: cost_proxy
    direction: minimize
  - name: tok_s
    direction: maximize
constraints:
  max_timeout_rate: 0.05
  max_vram_mb: 999999     # mock backend ignores; keeps interface consistent
  min_quality_overall: 0.80
search_space:
  # keep <= 10 params for MVP (mock backend can still simulate effects)
  concurrency:
    type: int
    low: 1
    high: 16
  max_new_tokens:
    type: int
    low: 32
    high: 256
  max_batch_tokens:
    type: int
    low: 512
    high: 4096
  kv_cache_limit_mb:
    type: int
    low: 256
    high: 4096
EOF
  echo "[+] Wrote ${TUNER_CFG}"
fi

# 5) Experiments config (optional)
EXP_CFG="configs/experiments/quickstart.yaml"
if [[ ! -f "${EXP_CFG}" ]]; then
  cat > "${EXP_CFG}" <<'EOF'
seed: 42
repeats: 1
warmup_requests: 5
backend: mock
quality_config: "configs/quality/quickstart.yaml"
EOF
  echo "[+] Wrote ${EXP_CFG}"
fi

# Detect whether CLI exists
if ! python -c "import servingbench_tuner" >/dev/null 2>&1; then
  echo
  echo "[!] servingbench_tuner package not importable yet."
  echo "    - Install deps: pip install -r requirements.lock.txt && pip install -e '.[dev]'"
  echo "    - Implement src/servingbench_tuner/cli.py (Typer app) to run benchmark/tune/report."
  echo "    Exiting quickstart without running benchmark."
  exit 0
fi

echo
echo "[+] Running CPU quickstart (mock backend)..."
set -x

python -m servingbench_tuner.cli benchmark \
  --backend mock \
  --workload "${WORKLOAD_CFG}" \
  --quality "${QUALITY_CFG}" \
  --out results

python -m servingbench_tuner.cli tune \
  --backend mock \
  --workload "${WORKLOAD_CFG}" \
  --quality "${QUALITY_CFG}" \
  --tuner "${TUNER_CFG}" \
  --out results

# If your tune command prints a best run_id to stdout, you can capture it.
# Here we just build a report from "latest" if your CLI supports it; otherwise no-op.
python -m servingbench_tuner.cli report --latest --out "reports/quickstart_report.html" || true

set +x
echo
echo "[✅] Quickstart finished."
echo "Artifacts: results/   Reports: reports/"
