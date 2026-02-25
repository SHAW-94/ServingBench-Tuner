# ServingBench-Tuner

Production-oriented tuning toolkit for **LLM inference serving** under real workload constraints.

ServingBench-Tuner focuses on a practical engineering problem: **improving latency / throughput / memory efficiency without breaking output quality**, and making the result reproducible, comparable, and reviewable.

---

## What this project does

### 1) Workload-driven benchmark harness
This project benchmarks serving configurations against workload profiles that look like production traffic, not just synthetic single-request tests.

Implemented workload styles include:
- **`short_qa`**: short prompts / short outputs (chat, support QA)
- **`long_gen`**: longer generations (content generation, drafting)
- **`agent_like`**: multi-turn / context-growing interactions (Agent / RAG-like patterns)

The workload layer supports arrival patterns, concurrency limits, token length behavior, and replay-style evaluation flow in a reproducible way.

### 2) End-to-end and server-side metrics
The benchmark pipeline tracks serving-facing metrics that matter in production:
- **TTFT** (time to first token)
- **P95/P99 latency**
- **Throughput** (tok/s)
- **Timeout / error rate**
- **Tail amplification (tail_amp)**
- **Tail-latency attribution fields** (queue / prefill / decode / retry)
- **VRAM peak** (when runtime monitoring is available; can be unavailable on some setups)

### 3) Quality guardrails
The project includes a quality gate so “faster” configs do not silently degrade quality.

Implemented quality checks include:
- overall quality score gate
- **quality breakdown support** in real validation outputs:
  - QA
  - structured output
  - code
  - summary

This keeps performance tuning aligned with production requirements.

### 4) Multi-objective auto-tuning
The tuner supports:
- **Manual baseline** (hand-tuned reference)
- **Random Search** (baseline optimizer)
- **NSGA-II** (multi-objective search)

It produces:
- feasible candidate set (under constraints)
- top recommendations
- Pareto-oriented comparison artifacts

The project also includes scenario-aware search-space shaping (especially important for `agent_like`) and regression gating logic.

### 5) Reporting and decision artifacts
The reporting pipeline generates artifacts for engineering review and rollout discussions:
- comparison table / summary JSON
- markdown + HTML report
- recommendation cards
- Pareto front plot
- tail-latency attribution plot
- README-ready summary snippet
- stability summary across multiple seeds
- capacity planning summary (SLA/RPS-oriented recommendation output)

### 6) Real backend validation (OpenAI-compatible API)
Besides simulated/formal comparison runs, the project supports **real backend validation** against an OpenAI-compatible endpoint (recommended: **vLLM**).

This lets you re-check the same Baseline / Random / NSGA2 recommendations on a real model server and collect actual:
- TTFT / P95 / P99
- tok/s
- timeout / error rate
- quality breakdown
- VRAM (if available)

---

## Why this is useful in practice

Most benchmark repos stop at “which config is faster.”

ServingBench-Tuner is built for the production-facing question:

> Under quality and resource constraints, which serving configuration should we deploy for **this workload**, and what tradeoffs do we accept?

This repo is useful for:
- model-serving performance tuning
- rollout regression checks
- capacity planning discussions
- SLA-focused latency optimization (especially tail latency)
- documenting serving configuration choices with reproducible evidence

---

## Repository highlights

- **Workload-aware benchmarking** (not single synthetic prompt tests)
- **Quality gate integrated** into tuning / evaluation workflow
- **NSGA-II + Random + manual baseline** for credible comparison
- **Tail-latency attribution outputs** (queue / prefill / decode / retry)
- **Reproducible artifacts** (JSON/Markdown/HTML, SQLite-backed run logging)
- **CI + tests + pre-commit hooks** for engineering hygiene
- **Real-backend validation path** for vLLM/OpenAI-compatible servers

---

## Project structure

```text
src/servingbench_tuner/
  backends/           # serving backends (OpenAI-compatible vLLM path, mock/local backends)
  client/             # request runner, pacing, tracing
  workload/           # workload schema / replay / arrival / session behavior
  metrics/            # e2e and server-side metrics
  quality/            # datasets, evaluators, quality gate
  tuner/              # search space, constraints, objectives, random search, NSGA-II, orchestrator
  experiments/        # regression/store helpers
  reporting/          # report rendering, plots, templates
  planner/            # capacity planning helpers

scripts/
  run_formal_comparison.py
  postprocess_formal_comparison.py
  inspect_formal_comparison.py
  check_regression_gate.py
  run_top_tier_pipeline.sh
  run_real_backend_validation.py
  check_quality_breakdown_gate.py
```

---

## Quickstart

### 1) Environment

```bash
conda env create -f environment.yml
conda activate sbt
pip install -e .
```

Optional but recommended:

```bash
pre-commit install
pytest -q
```

### 2) Run the full formal comparison pipeline (fast framework validation)

```bash
bash scripts/run_top_tier_pipeline.sh agent_like
```

This runs:
- formal comparison (Baseline / Random / NSGA2)
- post-processing (report plots + README snippet)
- result inspection summary
- regression gate check
- stability runs (multi-seed)
- capacity planning output

---

## Main outputs

After a successful run, you will usually see artifacts under:

- `results/formal_comparison/`
- `reports/formal_comparison/.../`

Typical files:
- `comparison_table.md`
- `comparison_summary.json`
- `README_top_snippet.md`
- `enterprise_<workload>_recommendation.md`
- `stability/stability_summary.md`
- `capacity_plan.md`
- `report.md`
- `report.html`

These are designed to be reviewable by engineers, not just scripts.

---

## Real backend validation (vLLM / OpenAI-compatible)

Use this when you want to validate the formal-comparison conclusions on a real model server.

### Option A: vLLM (recommended)

Start an OpenAI-compatible server (example):

```bash
python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port 8001 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --served-model-name sbt-qwen05b
```

Then run real validation:

```bash
python scripts/run_real_backend_validation.py \
  --model sbt-qwen05b \
  --base-url http://127.0.0.1:8001/v1 \
  --workload agent_like \
  --stream

python scripts/check_quality_breakdown_gate.py \
  --input results/real_validation/real_validation_summary.json
```

### Option B: Any OpenAI-compatible local server
If you are using another OpenAI-compatible server (including lightweight local setups), point `--base-url` and `--model` to that service.

---

## Recommended evaluation workflow

### For framework development (fast loop)
1. `bash scripts/run_top_tier_pipeline.sh agent_like`
2. Inspect `comparison_summary.json` and `report.html`
3. Check `stability_summary.md`
4. Review `capacity_plan.md`

### For deployment-facing validation (slower but more realistic)
1. Start a real OpenAI-compatible backend (vLLM)
2. Run `run_real_backend_validation.py`
3. Run `check_quality_breakdown_gate.py`
4. Compare real metrics vs. formal-comparison recommendation

---

## Interpreting results

When comparing Baseline / Random / NSGA2:

### For `agent_like`
Prioritize (interactive experience first):
- **P95 latency**
- **TTFT P95**
- **tail_amp**

Then evaluate:
- **tok/s**
- **VRAM headroom**
- **quality / quality breakdown**

The repo already includes regression-gate checks so a throughput-only win does not automatically pass if interactivity regresses too much.

---

## Testing and code quality

This repository includes:
- unit tests (`pytest`)
- `ruff` + `ruff-format` + `black`
- pre-commit hooks
- CI workflow for validation

---

## Scope and positioning

ServingBench-Tuner is a **production-oriented benchmarking and tuning framework**.

It is not only a raw benchmark script and not only an optimizer demo. The strongest part of the project is the combination of:
- workload realism
- quality guardrails
- multi-objective search
- regression gating
- reporting artifacts for engineering decisions

---

## Example commands (copy/paste)

### Formal pipeline

```bash
bash scripts/run_top_tier_pipeline.sh agent_like
```

### Inspect outputs

```bash
python scripts/inspect_formal_comparison.py
python scripts/check_regression_gate.py
```

### Real backend validation

```bash
python scripts/run_real_backend_validation.py \
  --model sbt-qwen05b \
  --base-url http://127.0.0.1:8001/v1 \
  --workload agent_like \
  --stream

python scripts/check_quality_breakdown_gate.py \
  --input results/real_validation/real_validation_summary.json
```

---

## Notes

- Some metrics (such as VRAM peak) depend on runtime visibility and monitoring support.
- Formal comparison runs are useful for fast iteration; real backend validation is the recommended final check before trusting a serving recommendation.

