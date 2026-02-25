from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """
    a dominates b if a is <= b in all dims and < in at least one dim.
    (Assumes minimization objectives.)
    """
    assert len(a) == len(b)
    le_all = True
    lt_any = False
    for x, y in zip(a, b, strict=False):
        if x > y:
            le_all = False
            break
        if x < y:
            lt_any = True
    return le_all and lt_any


def pareto_front_indices(F: list[list[float]]) -> list[int]:
    """
    Return indices of nondominated points in F (minimization).
    """
    n = len(F)
    if n == 0:
        return []
    nd = []
    for i in range(n):
        dominated_flag = False
        for j in range(n):
            if i == j:
                continue
            if dominates(F[j], F[i]):
                dominated_flag = True
                break
        if not dominated_flag:
            nd.append(i)
    return nd


@dataclass
class Recommendation:
    idx: int
    score: float
    note: str = ""


def weighted_score(F: Sequence[float], weights: Sequence[float]) -> float:
    """
    Simple weighted sum (minimization).
    Lower is better.
    """
    return float(sum(w * x for w, x in zip(weights, F, strict=False)))


def recommend_topk(
    F: list[list[float]],
    k: int = 5,
    weights: list[float] | None = None,
    prefer_pareto_only: bool = True,
) -> list[Recommendation]:
    """
    Recommend Top-K configs from objective vectors.
    - If prefer_pareto_only: select from Pareto front first.
    - weights: if provided, use weighted sum to rank (minimization).
      Otherwise, default weights=1 for all dims.
    """
    if not F:
        return []
    dim = len(F[0])
    w = weights or [1.0] * dim

    candidate_indices = list(range(len(F)))
    if prefer_pareto_only:
        candidate_indices = pareto_front_indices(F) or candidate_indices

    ranked = sorted(
        candidate_indices,
        key=lambda i: weighted_score(F[i], w),
    )
    out: list[Recommendation] = []
    for i in ranked[: max(1, k)]:
        out.append(
            Recommendation(
                idx=i,
                score=weighted_score(F[i], w),
                note="pareto" if i in set(candidate_indices) else "all",
            )
        )
    return out
