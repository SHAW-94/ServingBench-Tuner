from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from servingbench_tuner.core.types import E2EMetrics, GateResult, QualitySummary, ServerMetrics

from .constraints import ConstraintConfig, check_constraints
from .nsga2 import NSGA2Config, PymooNotInstalled, nsga2_optimize
from .objectives import ObjectiveConfig, extract_objective_values, objective_vector
from .pareto import pareto_front_indices, recommend_topk
from .random_search import RandomSearchConfig, random_search
from .search_space import Candidate, SearchSpace, default_search_space


@dataclass
class EvalOutcome:
    """Standardized outcome produced by eval_fn(cand)."""

    e2e: E2EMetrics
    server: ServerMetrics | None = None
    quality: QualitySummary | None = None
    gate: GateResult | None = None
    run_id: str = ""
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "e2e": asdict(self.e2e),
            "server": asdict(self.server) if self.server else None,
            "quality": asdict(self.quality) if self.quality else None,
            "gate": asdict(self.gate) if self.gate else None,
            "run_id": self.run_id,
            "notes": self.notes or {},
        }


def _ensure_sqlite(path: str | Path) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tuning_runs (
          id TEXT PRIMARY KEY,
          ts REAL,
          algo TEXT,
          workload_sig TEXT,
          model TEXT,
          notes TEXT
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS candidates (
          run_id TEXT,
          idx INTEGER,
          params_json TEXT,
          feasible INTEGER,
          violations_json TEXT,
          objectives_json TEXT,
          objective_vec_json TEXT,
          outcome_json TEXT,
          PRIMARY KEY (run_id, idx)
        );
        """
    )
    conn.commit()
    return conn


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


@dataclass
class OrchestratorConfig:
    """Tuning orchestration config."""

    algo: str = "random"
    outputs_dir: str = "results/artifacts"
    sqlite_path: str = "results/runs.sqlite"

    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)

    random_cfg: RandomSearchConfig = field(default_factory=RandomSearchConfig)
    nsga2_cfg: NSGA2Config = field(default_factory=NSGA2Config)

    topk: int = 5
    prefer_pareto_only: bool = True
    weights: list[float] | None = None


@dataclass
class TuningResult:
    run_id: str
    algo: str
    candidates: list[Candidate]
    outcomes: list[dict[str, Any]]
    feasible: list[bool]
    violations: list[dict[str, Any]]
    objective_values: list[dict[str, float]]
    objective_vecs: list[list[float]]
    pareto_indices: list[int]
    recommendations: list[dict[str, Any]]
    artifacts_dir: str


class TuningOrchestrator:
    """ask -> eval_fn -> objectives/constraints -> log -> pareto/recommendations."""

    def __init__(
        self, space: SearchSpace | None = None, cfg: OrchestratorConfig | None = None
    ) -> None:
        self.space = space or default_search_space()
        self.cfg = cfg or OrchestratorConfig()

    async def run(
        self,
        eval_fn: Callable[[Candidate], Any],
        run_id: str,
        workload_sig: str = "",
        model: str = "",
        baseline_quality: QualitySummary | None = None,
        notes: dict[str, Any] | None = None,
    ) -> TuningResult:
        algo = self.cfg.algo.lower().strip()
        ts = time.time()

        artifacts_dir = Path(self.cfg.outputs_dir) / run_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        conn = _ensure_sqlite(self.cfg.sqlite_path)
        conn.execute(
            "INSERT OR REPLACE INTO tuning_runs(id, ts, algo, workload_sig, model, notes) VALUES (?,?,?,?,?,?)",
            (run_id, ts, algo, workload_sig, model, _json(notes or {})),
        )
        conn.commit()

        async def _eval_and_normalize(c: Candidate) -> dict[str, Any]:
            out = await eval_fn(c)
            if isinstance(out, EvalOutcome):
                return out.to_dict()
            if isinstance(out, dict):
                return out
            raise TypeError("eval_fn must return EvalOutcome or dict")

        def _objective_fn(out: dict[str, Any]) -> list[float]:
            e2e = E2EMetrics(**out["e2e"])
            server = ServerMetrics(**out["server"]) if out.get("server") else None
            quality = QualitySummary(**out["quality"]) if out.get("quality") else None
            vals = extract_objective_values(e2e, server, quality, self.cfg.objective)
            vec, _ = objective_vector(vals, self.cfg.objective)
            return vec

        candidates: list[Candidate] = []
        outcomes: list[dict[str, Any]] = []
        feasible: list[bool] = []
        violations: list[dict[str, Any]] = []
        obj_vals_list: list[dict[str, float]] = []
        obj_vecs: list[list[float]] = []

        if algo == "random":
            rs = await random_search(self.space, self.cfg.random_cfg, eval_fn=_eval_and_normalize)
            candidates = rs.candidates
            outcomes = rs.outcomes
        elif algo == "nsga2":
            try:
                ns = await nsga2_optimize(
                    space=self.space,
                    cfg=self.cfg.nsga2_cfg,
                    eval_fn=_eval_and_normalize,
                    objective_fn=_objective_fn,
                    constraint_fn=lambda out: check_constraints(
                        E2EMetrics(**out["e2e"]),
                        ServerMetrics(**out["server"]) if out.get("server") else None,
                        QualitySummary(**out["quality"]) if out.get("quality") else None,
                        self.cfg.constraints,
                        baseline_quality,
                    ),
                )
                candidates = ns.candidates
                outcomes = ns.raw_outcomes
                obj_vecs = [list(v) for v in ns.F]
            except PymooNotInstalled:
                # keep old behavior for missing dependency only
                rs = await random_search(
                    self.space, self.cfg.random_cfg, eval_fn=_eval_and_normalize
                )
                candidates = rs.candidates
                outcomes = rs.outcomes
                algo = "random_fallback"
        else:
            raise ValueError(f"unknown algo: {algo}")

        for idx, (cand, out) in enumerate(zip(candidates, outcomes, strict=False)):
            e2e = E2EMetrics(**out["e2e"])
            server = ServerMetrics(**out["server"]) if out.get("server") else None
            quality = QualitySummary(**out["quality"]) if out.get("quality") else None

            feas, vio = check_constraints(
                e2e, server, quality, self.cfg.constraints, baseline_quality
            )
            feasible.append(bool(feas))
            violations.append(vio)

            vals = extract_objective_values(e2e, server, quality, self.cfg.objective)
            obj_vals_list.append(vals)

            if idx >= len(obj_vecs):
                vec, _ = objective_vector(vals, self.cfg.objective)
                obj_vecs.append(vec)

            cand_art = {
                "idx": idx,
                "params": cand.to_dict(),
                "serving_config": self.space.to_serving_config(cand),
                "feasible": bool(feas),
                "violations": vio,
                "objective_values": vals,
                "objective_vec": obj_vecs[idx],
                "outcome": out,
            }
            (artifacts_dir / f"candidate_{idx:04d}.json").write_text(
                json.dumps(cand_art, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            conn.execute(
                """
                INSERT OR REPLACE INTO candidates(
                  run_id, idx, params_json, feasible, violations_json,
                  objectives_json, objective_vec_json, outcome_json
                ) VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    idx,
                    _json(cand.to_dict()),
                    1 if feas else 0,
                    _json(vio),
                    _json(vals),
                    _json(obj_vecs[idx]),
                    _json(out),
                ),
            )

        conn.commit()
        conn.close()

        feas_indices = [i for i, ok in enumerate(feasible) if ok]
        F_feas = [obj_vecs[i] for i in feas_indices]
        pareto_local = pareto_front_indices(F_feas)
        pareto_indices = [feas_indices[i] for i in pareto_local] if pareto_local else feas_indices

        recs: list[dict[str, Any]] = []
        if feas_indices:
            F_rank = [obj_vecs[i] for i in feas_indices]
            rec = recommend_topk(
                F=F_rank,
                k=self.cfg.topk,
                weights=self.cfg.weights,
                prefer_pareto_only=self.cfg.prefer_pareto_only,
            )
            for r in rec:
                gidx = feas_indices[r.idx]
                recs.append(
                    {
                        "candidate_idx": gidx,
                        "score": r.score,
                        "note": r.note,
                        "params": candidates[gidx].to_dict(),
                        "objective_values": obj_vals_list[gidx],
                        "objective_vec": obj_vecs[gidx],
                        "serving_config": self.space.to_serving_config(candidates[gidx]),
                    }
                )

        summary = {
            "run_id": run_id,
            "algo": algo,
            "workload_sig": workload_sig,
            "model": model,
            "n_candidates": len(candidates),
            "n_feasible": sum(1 for f in feasible if f),
            "pareto_indices": pareto_indices,
            "recommendations": recs,
            "constraints": asdict(self.cfg.constraints),
            "objective": asdict(self.cfg.objective),
        }
        (artifacts_dir / "tuning_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return TuningResult(
            run_id=run_id,
            algo=algo,
            candidates=candidates,
            outcomes=outcomes,
            feasible=feasible,
            violations=violations,
            objective_values=obj_vals_list,
            objective_vecs=obj_vecs,
            pareto_indices=pareto_indices,
            recommendations=recs,
            artifacts_dir=str(artifacts_dir),
        )
