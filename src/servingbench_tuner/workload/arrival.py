from __future__ import annotations

import math
import random
import uuid
from collections.abc import Iterable
from dataclasses import dataclass

from .replay import ArrivalEvent, load_replay_events
from .schema import WorkloadSpec


@dataclass
class ArrivalPlan:
    """
    A list of arrival events (timestamp-sorted), used by the load generator.
    """

    events: list[ArrivalEvent]

    def __iter__(self) -> Iterable[ArrivalEvent]:
        return iter(self.events)


def _poisson_arrivals(rps: float, duration_s: float, rng: random.Random) -> list[float]:
    """
    Generate arrival timestamps (seconds since start) with exponential inter-arrival.
    """
    t = 0.0
    out: list[float] = []
    lam = max(1e-9, rps)
    while t < duration_s:
        u = rng.random()
        dt = -math.log(1.0 - u) / lam
        t += dt
        if t <= duration_s:
            out.append(t)
    return out


def _burst_arrivals(
    peak_rps: float,
    base_rps: float,
    on_s: float,
    off_s: float,
    cycles: int,
    rng: random.Random,
) -> list[float]:
    """
    Generate arrivals for burst pattern cycles. Each cycle = on_s + off_s.
    """
    out: list[float] = []
    t0 = 0.0
    for _ in range(cycles):
        # on segment
        out.extend([t0 + x for x in _poisson_arrivals(peak_rps, on_s, rng)])
        t0 += on_s
        # off segment
        if off_s > 0:
            out.extend([t0 + x for x in _poisson_arrivals(base_rps, off_s, rng)])
        t0 += off_s
    out.sort()
    return out


def generate_arrival_plan(
    spec: WorkloadSpec,
    rng: random.Random | None = None,
) -> ArrivalPlan:
    """
    Produce a list of ArrivalEvent from WorkloadSpec.arrival:
    - poisson: timestamps only; token lengths sampled later from length_dist
    - burst: timestamps only; token lengths sampled later
    - replay: uses trace jsonl events (may include token lengths & session fields)
    """
    rng = rng or random.Random(spec.seed)

    if spec.arrival.mode == "replay":
        events = load_replay_events(spec.arrival.trace_path)  # type: ignore[arg-type]
        # normalize start at 0
        if events:
            tmin = min(e.timestamp_s for e in events)
            for e in events:
                e.timestamp_s -= tmin
        return ArrivalPlan(events=events)

    if spec.arrival.mode == "poisson":
        ts = _poisson_arrivals(spec.arrival.rps or 1.0, float(spec.duration_s), rng)
    elif spec.arrival.mode == "burst":
        b = spec.arrival.burst
        assert b is not None
        ts = _burst_arrivals(b.peak_rps, b.base_rps, b.on_s, b.off_s, b.cycles, rng)
        # In burst mode, duration_s is ignored (pattern defines length); but we can clip if desired.
        if spec.duration_s > 0:
            ts = [t for t in ts if t <= spec.duration_s]
    else:
        raise ValueError(f"Unknown arrival mode: {spec.arrival.mode}")

    events: list[ArrivalEvent] = []
    for i, t in enumerate(ts):
        rid = f"req_{i:06d}_{uuid.uuid4().hex[:8]}"
        events.append(
            ArrivalEvent(
                timestamp_s=float(t),
                request_id=rid,
                session_id=None,
                turn_id=None,
                category=str(spec.arrival.mode),
                input_tokens=None,
                output_tokens=None,
                raw=None,
            )
        )
    return ArrivalPlan(events=events)
