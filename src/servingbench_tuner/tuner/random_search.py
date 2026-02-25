from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from servingbench_tuner.tuner.search_space import Candidate, SearchSpace


@dataclass
class RandomSearchConfig:
    seed: int = 42
    n_trials: int = 30


@dataclass
class RandomSearchResult:
    candidates: list[Candidate]
    outcomes: list[dict[str, Any]]  # arbitrary outcome dicts from eval_fn
    errors: list[str]


async def random_search(
    space: SearchSpace,
    cfg: RandomSearchConfig,
    eval_fn: Callable[[Candidate], Any],
    progress_cb: Callable[[int, int], None] | None = None,
) -> RandomSearchResult:
    """
    Baseline random search:
      for trial in n_trials:
        sample candidate
        validate
        eval_fn(candidate) -> outcome
    """
    rng = random.Random(cfg.seed)
    candidates: list[Candidate] = []
    outcomes: list[dict[str, Any]] = []
    errors: list[str] = []

    for i in range(cfg.n_trials):
        if progress_cb:
            progress_cb(i, cfg.n_trials)

        cand = space.sample_random(rng)
        ok, errs = space.validate(cand)
        if not ok:
            errors.append(f"invalid candidate: {errs}")
            continue

        candidates.append(cand)

        try:
            out = await eval_fn(cand)
            # store as dict if possible
            if isinstance(out, dict):
                outcomes.append(out)
            else:
                outcomes.append({"outcome": out})
        except Exception as e:
            errors.append(str(e))
            outcomes.append({"error": str(e)})

    return RandomSearchResult(candidates=candidates, outcomes=outcomes, errors=errors)
