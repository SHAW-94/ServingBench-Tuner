#!/usr/bin/env bash
set -euo pipefail

# start_vllm_server.sh
# No-docker launcher for vLLM OpenAI-compatible server.
#
# Usage:
#   ./scripts/start_vllm_server.sh configs/serving/vllm/base.yaml
#
# The YAML is expected to look like (all keys optional except model):
#   model: "meta-llama/Meta-Llama-3-8B-Instruct"
#   host: "0.0.0.0"
#   port: 8000
#   dtype: "auto"
#   tensor_parallel_size: 1
#   gpu_memory_utilization: 0.90
#   max_model_len: 8192
#   served_model_name: "llama3-8b"
#   api_key: "token-xxx"
#   extra_args: ["--max-num-seqs", "256"]
#
# Notes:
# - vLLM CLI flags can evolve. To avoid mismatch, put uncommon flags into extra_args.
# - This script does not manage process supervisors. Use tmux/systemd if needed.

CONFIG_PATH="${1:-}"
if [[ -z "${CONFIG_PATH}" ]]; then
  echo "ERROR: Missing config yaml path."
  echo "Usage: $0 configs/serving/vllm/base.yaml"
  exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: Config file not found: ${CONFIG_PATH}"
  exit 1
fi

# Ensure vllm exists
if ! command -v vllm >/dev/null 2>&1; then
  echo "ERROR: 'vllm' command not found."
  echo "Install with: pip install -e '.[gpu]'  (on Linux + CUDA GPU)"
  exit 1
fi

# Parse YAML via python (PyYAML is in base deps)
read -r MODEL HOST PORT DTYPE TP GPU_MEM_UTIL MAX_MODEL_LEN SERVED_NAME API_KEY EXTRA_ARGS_JSON <<EOF
$(
python - <<'PY'
import json, sys, yaml
path = sys.argv[1]
cfg = yaml.safe_load(open(path, "r", encoding="utf-8")) or {}

def get(k, default=None):
    return cfg.get(k, default)

model = get("model", "")
host  = str(get("host", "0.0.0.0"))
port  = str(get("port", 8000))
dtype = str(get("dtype", "auto"))

tp = get("tensor_parallel_size", None)
tp = "" if tp is None else str(tp)

gpu_mem = get("gpu_memory_utilization", None)
gpu_mem = "" if gpu_mem is None else str(gpu_mem)

max_len = get("max_model_len", None)
max_len = "" if max_len is None else str(max_len)

served = get("served_model_name", None)
served = "" if served is None else str(served)

api_key = get("api_key", None)
api_key = "" if api_key is None else str(api_key)

extra_args = get("extra_args", []) or []
if not isinstance(extra_args, list):
    raise SystemExit("extra_args must be a list, e.g. ['--max-num-seqs','256']")
extra_json = json.dumps(extra_args)

print(model, host, port, dtype, tp, gpu_mem, max_len, served, api_key, extra_json)
PY
"${CONFIG_PATH}"
)
EOF

if [[ -z "${MODEL}" ]]; then
  echo "ERROR: 'model' is required in ${CONFIG_PATH}"
  exit 1
fi

# Build vLLM command (only include args if provided)
cmd=(vllm serve "${MODEL}" "--host" "${HOST}" "--port" "${PORT}" "--dtype" "${DTYPE}")

if [[ -n "${TP}" ]]; then
  cmd+=("--tensor-parallel-size" "${TP}")
fi
if [[ -n "${GPU_MEM_UTIL}" ]]; then
  cmd+=("--gpu-memory-utilization" "${GPU_MEM_UTIL}")
fi
if [[ -n "${MAX_MODEL_LEN}" ]]; then
  cmd+=("--max-model-len" "${MAX_MODEL_LEN}")
fi
if [[ -n "${SERVED_NAME}" ]]; then
  cmd+=("--served-model-name" "${SERVED_NAME}")
fi
if [[ -n "${API_KEY}" ]]; then
  cmd+=("--api-key" "${API_KEY}")
fi

# Append extra_args
python - <<'PY' "${EXTRA_ARGS_JSON}"
import json, sys
args = json.loads(sys.argv[1])
# print args one per line so bash can read safely
for a in args:
    print(a)
PY | while IFS= read -r a; do
  [[ -n "$a" ]] && cmd+=("$a")
done

echo "[+] Starting vLLM server with config: ${CONFIG_PATH}"
echo "[+] Command:"
printf '    %q ' "${cmd[@]}"
echo
echo

exec "${cmd[@]}"
