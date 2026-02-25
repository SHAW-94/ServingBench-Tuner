from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from servingbench_tuner.backends.base import BackendAdapter, GenerationRequest
from servingbench_tuner.client.runner import (
    LoadRunner,
    RunnerConfig,
    RunResult,
    default_request_builder,
)
from servingbench_tuner.client.tracing import RequestTrace
from servingbench_tuner.core.types import E2EMetrics, GateResult, QualitySummary, ServerMetrics
from servingbench_tuner.experiments.store import ResultStore, RunRecord
from servingbench_tuner.metrics.e2e import aggregate_e2e_metrics
from servingbench_tuner.metrics.server import ServerMonitor
from servingbench_tuner.quality.dataset import EvalExample, load_eval_pack
from servingbench_tuner.quality.evaluators.base import EvaluatorRegistry
from servingbench_tuner.quality.evaluators.code_unittest import CodeUnitTestEvaluator
from servingbench_tuner.quality.evaluators.exact_match import ExactMatchEvaluator
from servingbench_tuner.quality.evaluators.json_schema import JSONSchemaEvaluator
from servingbench_tuner.quality.evaluators.llm_judge import LLMJudgeEvaluator
from servingbench_tuner.quality.gate import QualityGate, evaluate_pack


@dataclass
class ExperimentConfig:
    """
    One-click experiment config:
    - warmup_requests: first N requests ignored in E2E metrics
    - repeats: number of repeated benchmark runs for robustness
    - seed: for any stochastic components in workload generation (handled upstream)
    - record_traces: save per-request traces to artifacts (JSONL)
    - measure_server: enable NVML/torch sampling (GPU env)
    """

    warmup_requests: int = 5
    repeats: int = 3
    seed: int = 42

    record_traces: bool = True

    measure_server: bool = False
    server_sample_interval_s: float = 0.2
    gpu_index: int = 0

    # quality
    run_quality: bool = True
    quality_fail_fast: bool = True
    quality_policy: dict[str, Any] = None  # passed to QualityGate
    eval_pack_path: str = "data/eval_pack/eval_pack.jsonl"

    # judge config
    enable_llm_judge: bool = False
    judge_model: str | None = None
    judge_backend: BackendAdapter | None = None

    # storage
    experiment_id: str = "exp_default"
    model: str = ""
    workload_sig: str = ""
    notes: dict[str, Any] = None


def _run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


def _dump_traces_jsonl(path: Path, traces: list[RequestTrace]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in sorted(traces, key=lambda x: (x.arrival_s, x.request_id)):
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")


def _aggregate_repeat_summaries(e2e_list: list[E2EMetrics]) -> dict[str, Any]:
    """
    Robust aggregation across repeats.
    We keep per-repeat metrics + median summary + simple dispersion (min/max).
    """
    if not e2e_list:
        return {"repeats": 0, "per_repeat": [], "median": {}}

    per = [asdict(m) for m in e2e_list]

    # median of selected keys (you can expand)
    keys = list(per[0].keys())
    med: dict[str, float] = {}
    for k in keys:
        vals = [float(getattr(m, k)) for m in e2e_list]
        vals_sorted = sorted(vals)
        mid = len(vals_sorted) // 2
        if len(vals_sorted) % 2 == 1:
            med[k] = float(vals_sorted[mid])
        else:
            med[k] = float((vals_sorted[mid - 1] + vals_sorted[mid]) / 2.0)

    # min/max for quick stability signal
    minmax: dict[str, Any] = {}
    for k in keys:
        vals = [float(getattr(m, k)) for m in e2e_list]
        minmax[k] = {"min": float(min(vals)), "max": float(max(vals))}

    return {
        "repeats": len(e2e_list),
        "per_repeat": per,
        "median": med,
        "minmax": minmax,
    }


async def _infer_outputs_for_eval_pack(
    backend: BackendAdapter,
    examples: list[EvalExample],
    model: str | None = None,
) -> dict[str, str]:
    """
    Simple inference loop for quality pack.
    For production you might add concurrency; for 80~200 items sequential is OK.
    """
    outputs: dict[str, str] = {}
    for ex in examples:
        req = GenerationRequest(
            model=model,
            messages=[{"role": "user", "content": ex.prompt}],
            max_new_tokens=int(ex.get("max_new_tokens", 256)),
            temperature=float(ex.get("temperature", 0.0)),
            top_p=float(ex.get("top_p", 1.0)),
            stream=False,
            timeout_s=float(ex.get("timeout_s", 60.0)),
            extra=dict(ex.get("extra", {}) or {}),
        )
        res = await backend.generate(req)
        outputs[ex.id] = res.text
    return outputs


def build_default_registry(
    enable_llm_judge: bool = False,
    judge_backend: BackendAdapter | None = None,
    judge_model: str | None = None,
) -> EvaluatorRegistry:
    evaluators = [
        ExactMatchEvaluator(),
        JSONSchemaEvaluator(),
        CodeUnitTestEvaluator(timeout_s=10),
    ]
    if enable_llm_judge:
        evaluators.append(
            LLMJudgeEvaluator(
                judge_backend=judge_backend,
                judge_model=judge_model,
                temperature=0.0,
                seed=42,
            )
        )
    return EvaluatorRegistry(evaluators)


class ExperimentRunner:
    """
    Orchestrates:
      - warmup + repeats
      - e2e aggregation
      - optional server sampling
      - optional quality gate
      - result persistence: SQLite + artifacts
    """

    def __init__(
        self,
        backend: BackendAdapter,
        store: ResultStore | None = None,
    ) -> None:
        self.backend = backend
        self.store = store or ResultStore()

    async def run_benchmark_repeats(
        self,
        events: list[Any],
        request_builder: Callable[[Any], GenerationRequest] = default_request_builder,
        runner_cfg: RunnerConfig | None = None,
        exp_cfg: ExperimentConfig | None = None,
        serving_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        events: List[ArrivalEvent] or anything compatible with request_builder
        """
        exp_cfg = exp_cfg or ExperimentConfig()
        runner_cfg = runner_cfg or RunnerConfig()

        # ensure experiment exists
        self.store.ensure_experiment(
            experiment_id=exp_cfg.experiment_id,
            model=exp_cfg.model,
            workload_sig=exp_cfg.workload_sig,
            notes=exp_cfg.notes or {},
        )

        run_id = _run_id()
        art_dir = self.store.make_run_artifacts_dir(exp_cfg.experiment_id, run_id)

        # record inputs
        self.store.write_json_artifact(
            art_dir / "run_config.json",
            {
                "experiment": asdict(exp_cfg),
                "runner": asdict(runner_cfg),
                "serving_config": serving_config or {},
            },
        )

        e2e_repeats: list[E2EMetrics] = []
        all_traces: list[RequestTrace] = []
        server_summary: ServerMetrics | None = None

        async def _one_repeat(
            rep_idx: int,
        ) -> tuple[E2EMetrics, list[RequestTrace], ServerMetrics | None]:
            # server monitor (optional)
            if exp_cfg.measure_server:
                async with ServerMonitor(
                    sample_interval_s=exp_cfg.server_sample_interval_s,
                    gpu_index=exp_cfg.gpu_index,
                    enable_nvml=True,
                    enable_torch_fragmentation=True,
                ) as mon:
                    lr = LoadRunner(self.backend, runner_cfg, request_builder=request_builder)
                    rr: RunResult = await lr.run(events)
                    # metrics
                    e2e, debug = aggregate_e2e_metrics(
                        rr.traces, warmup_requests=exp_cfg.warmup_requests
                    )
                    sm = mon.summary()
                    # persist repeat artifacts
                    if exp_cfg.record_traces:
                        _dump_traces_jsonl(
                            art_dir / f"traces_repeat_{rep_idx:02d}.jsonl", rr.traces
                        )
                    self.store.write_json_artifact(
                        art_dir / f"e2e_repeat_{rep_idx:02d}.json",
                        {"e2e": asdict(e2e), "debug": debug},
                    )
                    self.store.write_json_artifact(
                        art_dir / f"server_repeat_{rep_idx:02d}.json",
                        {"server": asdict(sm), "samples": mon.debug_samples()},
                    )
                    return e2e, rr.traces, sm
            else:
                lr = LoadRunner(self.backend, runner_cfg, request_builder=request_builder)
                rr = await lr.run(events)
                e2e, debug = aggregate_e2e_metrics(
                    rr.traces, warmup_requests=exp_cfg.warmup_requests
                )
                if exp_cfg.record_traces:
                    _dump_traces_jsonl(art_dir / f"traces_repeat_{rep_idx:02d}.jsonl", rr.traces)
                self.store.write_json_artifact(
                    art_dir / f"e2e_repeat_{rep_idx:02d}.json",
                    {"e2e": asdict(e2e), "debug": debug},
                )
                return e2e, rr.traces, None

        # repeats (sequential for reproducibility)
        for i in range(exp_cfg.repeats):
            e2e_i, traces_i, server_i = await _one_repeat(i)
            e2e_repeats.append(e2e_i)
            all_traces.extend(traces_i)
            if server_i is not None:
                server_summary = (
                    server_i  # last summary is usually fine; you can aggregate if needed
                )

        agg = _aggregate_repeat_summaries(e2e_repeats)
        self.store.write_json_artifact(art_dir / "e2e_aggregated.json", agg)

        # quality (optional)
        quality_summary: QualitySummary | None = None
        gate_result: GateResult | None = None
        quality_details_path = ""

        if exp_cfg.run_quality:
            examples = load_eval_pack(exp_cfg.eval_pack_path)
            registry = build_default_registry(
                enable_llm_judge=exp_cfg.enable_llm_judge,
                judge_backend=exp_cfg.judge_backend,
                judge_model=exp_cfg.judge_model,
            )
            outputs_by_id = await _infer_outputs_for_eval_pack(
                self.backend, examples, model=exp_cfg.model or None
            )
            quality_details_path = str(art_dir / "quality_details.json")

            quality_summary, _results = await evaluate_pack(
                examples=examples,
                outputs_by_id=outputs_by_id,
                registry=registry,
                details_out_path=quality_details_path,
                fail_fast=bool(exp_cfg.quality_fail_fast),
            )

            policy = exp_cfg.quality_policy or {"gate": {"min_overall": 0.0}}
            gate = QualityGate(policy)
            gate_result = gate.check(
                quality_summary, baseline=None
            )  # baseline wired in higher-level regression

            self.store.write_json_artifact(
                art_dir / "quality_summary.json",
                {"quality": asdict(quality_summary), "gate": asdict(gate_result)},
            )

        # persist run record
        # choose aggregated e2e: use median as main number (production-friendly)
        median_e2e = agg.get("median", {})
        e2e_main = median_e2e if isinstance(median_e2e, dict) else asdict(e2e_repeats[-1])

        rec = RunRecord(
            experiment_id=exp_cfg.experiment_id,
            run_id=run_id,
            ts=time.time(),
            config_json={
                "runner": asdict(runner_cfg),
                "serving_config": serving_config or {},
                "experiment": asdict(exp_cfg),
            },
            e2e_json=e2e_main,
            server_json=(asdict(server_summary) if server_summary is not None else None),
            quality_json=(asdict(quality_summary) if quality_summary is not None else None),
            gate_json=(asdict(gate_result) if gate_result is not None else None),
            feasible=(bool(gate_result.passed) if gate_result is not None else None),
            violations_json=None,
            artifacts_dir=str(art_dir),
        )
        self.store.record_run(rec)

        return {
            "experiment_id": exp_cfg.experiment_id,
            "run_id": run_id,
            "artifacts_dir": str(art_dir),
            "e2e_aggregated": agg,
            "server_summary": (asdict(server_summary) if server_summary else None),
            "quality_summary": (asdict(quality_summary) if quality_summary else None),
            "gate": (asdict(gate_result) if gate_result else None),
            "quality_details_path": quality_details_path,
        }
