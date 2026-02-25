SHELL := /bin/bash
PY := python

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "ServingBench-Tuner (no-docker) targets:"
	@echo ""
	@echo "  make venv        - create venv in .venv (pip-based)"
	@echo "  make install     - install package (editable) + base deps"
	@echo "  make install-dev - install with dev extras"
	@echo "  make lock        - regenerate requirements.lock.txt (pip-tools)"
	@echo ""
	@echo "  make bench       - run benchmark (workload + serving config)"
	@echo "  make tune        - run tuner (NSGA-II + random baseline optional)"
	@echo "  make report      - generate report.html (from run id)"
	@echo "  make regress     - regression check between two run ids"
	@echo ""
	@echo "  make fmt         - format (black + ruff format)"
	@echo "  make lint        - lint (ruff)"
	@echo "  make test        - tests (pytest)"
	@echo "  make precommit   - run pre-commit on all files"

.PHONY: venv
venv:
	@$(PY) -m venv .venv
	@echo "Activate: source .venv/bin/activate"

.PHONY: install
install:
	@pip install -U pip
	@pip install -e .

.PHONY: install-dev
install-dev:
	@pip install -U pip
	@pip install -e ".[dev]"

# Optional:
# GPU: pip install -e ".[gpu,dev]"
# CPU: pip install -e ".[cpu,dev]"

.PHONY: lock
lock:
	@command -v pip-compile >/dev/null 2>&1 || (echo "pip-tools not found. Run: pip install -U pip-tools" && exit 1)
	@pip-compile -q --resolver=backtracking --strip-extras -o requirements.lock.txt pyproject.toml
	@echo "Wrote requirements.lock.txt"

# ---- Project commands (require CLI implemented as sbt entrypoint) ----
WORKLOAD ?= configs/workload/agent_like.yaml
SERVING  ?= configs/serving/vllm/base.yaml
TUNER    ?= configs/tuner/nsga2.yaml
OUTDIR   ?= results
RUN_ID   ?=
BASE_ID  ?=
CAND_ID  ?=
REPORT_OUT ?= reports/report.html

.PHONY: bench
bench:
	@$(PY) -m servingbench_tuner.cli benchmark --workload $(WORKLOAD) --serving $(SERVING) --out $(OUTDIR)

.PHONY: tune
tune:
	@$(PY) -m servingbench_tuner.cli tune --workload $(WORKLOAD) --tuner $(TUNER) --out $(OUTDIR)

.PHONY: report
report:
	@if [ -z "$(RUN_ID)" ]; then echo "RUN_ID required. Example: make report RUN_ID=run_2026xxxx"; exit 1; fi
	@$(PY) -m servingbench_tuner.cli report --run-id $(RUN_ID) --out $(REPORT_OUT)

.PHONY: regress
regress:
	@if [ -z "$(BASE_ID)" ] || [ -z "$(CAND_ID)" ]; then echo "BASE_ID and CAND_ID required. Example: make regress BASE_ID=... CAND_ID=..."; exit 1; fi
	@$(PY) -m servingbench_tuner.cli regress --base $(BASE_ID) --cand $(CAND_ID)

# ---- Dev tools ----
.PHONY: fmt
fmt:
	@ruff format .
	@black .

.PHONY: lint
lint:
	@ruff check .

.PHONY: test
test:
	@pytest -q

.PHONY: precommit
precommit:
	@pre-commit run --all-files
