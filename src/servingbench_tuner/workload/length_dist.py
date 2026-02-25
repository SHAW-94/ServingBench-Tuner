from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .schema import DistSpec, EmpiricalDistSpec, FixedDistSpec, LogNormalDistSpec


class TokenSampler(Protocol):
    def sample(self, rng: random.Random) -> int: ...


@dataclass
class FixedSampler(TokenSampler):
    value: int

    def sample(self, rng: random.Random) -> int:
        return int(self.value)


@dataclass
class EmpiricalSampler(TokenSampler):
    values: list[int]

    def sample(self, rng: random.Random) -> int:
        if not self.values:
            return 0
        return int(rng.choice(self.values))


@dataclass
class LogNormalSampler(TokenSampler):
    mu: float
    sigma: float
    min_tokens: int
    max_tokens: int

    def sample(self, rng: random.Random) -> int:
        x = rng.lognormvariate(self.mu, self.sigma)
        v = int(round(x))
        if v < self.min_tokens:
            v = self.min_tokens
        if v > self.max_tokens:
            v = self.max_tokens
        return v


def _load_empirical_values(path: str | Path) -> list[int]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"empirical dist file not found: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or "values" not in obj:
        raise ValueError(f"empirical dist json must contain 'values' list: {p}")
    values = obj["values"]
    if not isinstance(values, list) or not values:
        raise ValueError(f"empirical dist 'values' must be a non-empty list: {p}")
    out = [int(v) for v in values]
    return out


def build_sampler(spec: DistSpec) -> TokenSampler:
    if isinstance(spec, FixedDistSpec):
        return FixedSampler(value=spec.value)
    if isinstance(spec, EmpiricalDistSpec):
        values = _load_empirical_values(spec.path)
        return EmpiricalSampler(values=values)
    if isinstance(spec, LogNormalDistSpec):
        return LogNormalSampler(
            mu=spec.mu, sigma=spec.sigma, min_tokens=spec.min_tokens, max_tokens=spec.max_tokens
        )
    raise TypeError(f"Unsupported dist spec: {type(spec)}")


@dataclass
class LengthSamplers:
    prompt: TokenSampler
    output: TokenSampler

    def sample_prompt(self, rng: random.Random) -> int:
        return self.prompt.sample(rng)

    def sample_output(self, rng: random.Random) -> int:
        return self.output.sample(rng)


def build_length_samplers(length_dist_spec) -> LengthSamplers:
    """
    Build prompt/output samplers from LengthDistSpec.
    """
    return LengthSamplers(
        prompt=build_sampler(length_dist_spec.prompt),
        output=build_sampler(length_dist_spec.output),
    )
