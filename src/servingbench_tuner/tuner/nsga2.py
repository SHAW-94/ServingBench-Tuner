from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from servingbench_tuner.tuner.search_space import Candidate, SearchSpace


class PymooNotInstalled(RuntimeError):
    pass


@dataclass
class NSGA2Config:
    seed: int = 42
    pop_size: int = 24
    n_gen: int = 10


@dataclass
class NSGA2Result:
    X: list[list[float]]  # all evaluated decision vectors (history)
    F: list[list[float]]  # all evaluated objective vectors (history, minimization)
    candidates: list[Candidate]  # decoded candidates (history)
    raw_outcomes: list[dict[str, Any]]  # raw eval outputs (history)


def _sanitize_objective_vec(
    vec: list[float] | tuple[float, ...] | np.ndarray, m: int
) -> np.ndarray:
    arr = np.asarray(list(vec), dtype=float).reshape(-1)
    if arr.size != m:
        return np.full((m,), 1e9, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 1e9)
    return arr.astype(float)


def _merge_population(pop, off):
    """
    pymoo compatibility:
    - some versions: pop.union(off)
    - some versions: Population.merge(pop, off)
    - some versions support pop + off
    """
    if hasattr(pop, "union"):
        return pop.union(off)

    try:
        from pymoo.core.population import Population  # type: ignore

        if hasattr(Population, "merge"):
            return Population.merge(pop, off)
    except Exception:
        pass

    try:
        return pop + off
    except Exception as e:
        raise RuntimeError("Unable to merge pymoo populations for this pymoo version") from e


def _fallback_outcome(
    template: dict[str, Any], reason: str, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Always return an orchestrator-compatible outcome shape with e2e/server/quality/gate keys.
    """
    out = copy.deepcopy(template)
    if not isinstance(out, dict):
        out = {}

    # hard guarantee required keys exist
    out.setdefault("e2e", {})
    out.setdefault("server", None)
    out.setdefault("quality", None)
    out.setdefault("gate", None)

    meta = out.get("_nsga2_meta")
    if not isinstance(meta, dict):
        meta = {}
    meta["reason"] = reason
    if extra:
        meta.update(extra)
    out["_nsga2_meta"] = meta
    return out


async def nsga2_optimize(
    space: SearchSpace,
    cfg: NSGA2Config,
    eval_fn: Callable[[Candidate], Any],
    objective_fn: Callable[[dict[str, Any]], list[float]],
    constraint_fn: Callable[[dict[str, Any]], tuple[bool, dict[str, Any]]] | None = None,
) -> NSGA2Result:
    """
    Async-friendly NSGA-II (manual generational loop using pymoo operators).

    - eval_fn(candidate) -> dict outcome
    - objective_fn(outcome) -> objective vector (minimization)
    - constraint_fn(outcome) -> (feasible, violations)
      We encode a single inequality constraint:
        G = 0.0   (feasible)
        G = 1.0   (infeasible)
      and pymoo expects G as numpy array.
    """
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2  # type: ignore
        from pymoo.core.problem import Problem  # type: ignore
    except Exception as e:
        raise PymooNotInstalled("pymoo is not installed. Install with: pip install pymoo") from e

    # Build numeric bounds for pymoo
    xl: list[float] = []
    xu: list[float] = []

    for s in space.specs:
        ptype = getattr(s, "ptype", None)
        if ptype == "int":
            lo = float(int(s.low) if getattr(s, "low", None) is not None else 0)
            hi = float(int(s.high) if getattr(s, "high", None) is not None else lo)
            xl.append(lo)
            xu.append(hi)
        elif ptype == "float":
            lo = float(s.low) if getattr(s, "low", None) is not None else 0.0
            hi = float(s.high) if getattr(s, "high", None) is not None else lo
            xl.append(lo)
            xu.append(hi)
        elif ptype == "bool":
            xl.append(0.0)
            xu.append(1.0)
        elif ptype == "categorical":
            choices = getattr(s, "choices", None) or []
            if len(choices) == 0:
                xl.append(0.0)
                xu.append(0.0)
            else:
                xl.append(0.0)
                xu.append(float(len(choices) - 1))
        else:
            xl.append(0.0)
            xu.append(1.0)

    # Probe once to determine objective dimension and build a fallback outcome template
    dummy_out = await eval_fn(space.defaults())
    if not isinstance(dummy_out, dict):
        raise TypeError("nsga2_optimize expects eval_fn to return dict outcomes")
    fallback_template = _fallback_outcome(dummy_out, "template_probe")

    dummy_f = objective_fn(dummy_out)
    m_obj = len(dummy_f)
    has_constraint = constraint_fn is not None
    n_constr = 1 if has_constraint else 0

    class AsyncProblem(Problem):
        def __init__(self) -> None:
            super().__init__(
                n_var=len(space.specs),
                n_obj=m_obj,
                n_constr=n_constr,
                xl=np.asarray(xl, dtype=float),
                xu=np.asarray(xu, dtype=float),
            )

    problem = AsyncProblem()
    algo = NSGA2(pop_size=cfg.pop_size, eliminate_duplicates=True)
    algo.setup(problem, seed=cfg.seed, verbose=False)

    # Global history (all evaluated points, not just final pop)
    X_hist: list[list[float]] = []
    F_hist: list[list[float]] = []
    cand_hist: list[Candidate] = []
    raw_outcomes: list[dict[str, Any]] = []

    async def eval_population(pop) -> None:
        """Evaluate a pymoo Population in-place (assign ind.F / ind.G as np.ndarray)."""
        nonlocal X_hist, F_hist, cand_hist, raw_outcomes

        for ind in pop:
            x = np.asarray(ind.X, dtype=float).reshape(-1)
            cand = space.decode_from_pymoo(x.tolist())

            ok, errs = space.validate(cand)
            feasible = False

            if not ok:
                outc = _fallback_outcome(
                    fallback_template,
                    "invalid_candidate",
                    {"validation_errors": errs},
                )
                f_arr = np.full((m_obj,), 1e9, dtype=float)
            else:
                try:
                    maybe_out = await eval_fn(cand)
                    if isinstance(maybe_out, dict):
                        outc = maybe_out
                    else:
                        outc = _fallback_outcome(
                            fallback_template,
                            "non_dict_eval_outcome",
                            {"type": str(type(maybe_out))},
                        )
                except Exception as ex:
                    outc = _fallback_outcome(
                        fallback_template,
                        "eval_exception",
                        {"error": str(ex)},
                    )

                # 保证 orchestrator 需要的 key 存在
                if "e2e" not in outc:
                    outc = _fallback_outcome(
                        outc if isinstance(outc, dict) else fallback_template, "missing_e2e"
                    )

                try:
                    f_arr = _sanitize_objective_vec(objective_fn(outc), m_obj)
                except Exception as ex:
                    outc = _fallback_outcome(outc, "objective_exception", {"error": str(ex)})
                    f_arr = np.full((m_obj,), 1e9, dtype=float)

                if has_constraint:
                    try:
                        feasible, _vio = constraint_fn(outc)  # type: ignore[misc]
                    except Exception as ex:
                        outc = _fallback_outcome(outc, "constraint_exception", {"error": str(ex)})
                        feasible = False
                else:
                    feasible = True

            # Assign pymoo fields (must be ndarray, not list)
            ind.F = f_arr
            if has_constraint:
                ind.G = np.asarray([0.0 if feasible else 1.0], dtype=float)

            # Save history
            raw_outcomes.append(outc)
            X_hist.append(x.tolist())
            F_hist.append(f_arr.tolist())
            cand_hist.append(cand)

    # --- Initial population ---
    pop = algo.initialization.do(problem, cfg.pop_size, algorithm=algo)
    await eval_population(pop)

    # CRITICAL: rank/crowding must be assigned before first mating
    pop = algo.survival.do(problem, pop, cfg.pop_size, algorithm=algo)

    # --- Generational loop ---
    for _gen in range(cfg.n_gen):
        off = algo.mating.do(problem, pop, cfg.pop_size, algorithm=algo)
        await eval_population(off)

        merged = _merge_population(pop, off)
        pop = algo.survival.do(problem, merged, cfg.pop_size, algorithm=algo)

    return NSGA2Result(
        X=X_hist,
        F=F_hist,
        candidates=cand_hist,
        raw_outcomes=raw_outcomes,
    )
