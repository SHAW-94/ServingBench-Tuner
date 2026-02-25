from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass

import anyio

from servingbench_tuner.workload.replay import ArrivalEvent


@dataclass
class PacingConfig:
    """
    Event scheduling config.

    time_scale:
      - 1.0 = real-time pacing
      - >1.0 = faster than real time (useful for CI)
      - <1.0 = slower
    """

    time_scale: float = 1.0
    clock: str = "time"  # "time" uses time.time() wall clock, "monotonic" uses time.monotonic()

    def now(self) -> float:
        return time.monotonic() if self.clock == "monotonic" else time.time()


class Pacer:
    """
    Schedule ArrivalEvent list by their timestamp_s (seconds since start).
    """

    def __init__(self, cfg: PacingConfig | None = None) -> None:
        self.cfg = cfg or PacingConfig()

    async def iter_events(self, events: Iterable[ArrivalEvent]) -> AsyncIterator[ArrivalEvent]:
        """
        Yield events at (timestamp_s / time_scale) relative to pacing start.
        """
        start = self.cfg.now()
        for ev in sorted(list(events), key=lambda e: (e.timestamp_s, e.request_id)):
            target = start + (float(ev.timestamp_s) / max(1e-9, self.cfg.time_scale))
            delay = target - self.cfg.now()
            if delay > 0:
                await anyio.sleep(delay)
            yield ev

    async def sleep_until(self, t_rel_s: float, start_t: float) -> None:
        """
        Sleep until t_rel_s since start_t (start_t in cfg.now() time base).
        """
        target = start_t + (t_rel_s / max(1e-9, self.cfg.time_scale))
        delay = target - self.cfg.now()
        if delay > 0:
            await anyio.sleep(delay)
