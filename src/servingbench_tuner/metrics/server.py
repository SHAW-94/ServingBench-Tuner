from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

import anyio

from servingbench_tuner.core.types import ServerMetrics

from .analysis import mean, percentile
from .nvml import NVMLCollector, NVMLNotAvailable, NVMLSample


def _torch_fragmentation_ratio() -> float | None:
    """
    Approximate fragmentation ratio using torch CUDA allocator stats:
      reserved / allocated
    If torch/cuda not available, returns None.
    """
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return None
        reserved = float(torch.cuda.memory_reserved())
        allocated = float(torch.cuda.memory_allocated())
        if allocated <= 1e-9:
            return None
        return reserved / allocated
    except Exception:
        return None


class ServerMonitor:
    """
    Periodically samples server-side metrics (NVML + optional torch allocator stats).

    Usage:
      mon = ServerMonitor(sample_interval_s=0.2)
      async with mon:
          ... run load ...
      metrics = mon.summary()

    If NVML is unavailable, it gracefully degrades and still can provide
    torch fragmentation ratio if available (but without memory peak).
    """

    def __init__(
        self,
        sample_interval_s: float = 0.2,
        gpu_index: int = 0,
        enable_nvml: bool = True,
        enable_torch_fragmentation: bool = True,
    ) -> None:
        self.sample_interval_s = float(sample_interval_s)
        self.gpu_index = int(gpu_index)
        self.enable_nvml = bool(enable_nvml)
        self.enable_torch_fragmentation = bool(enable_torch_fragmentation)

        self._nvml: NVMLCollector | None = None
        self._samples: list[NVMLSample] = []
        self._frag_samples: list[float] = []
        self._tg: anyio.abc.TaskGroup | None = None
        self._running = False

    async def __aenter__(self) -> ServerMonitor:
        self._running = True
        if self.enable_nvml:
            try:
                self._nvml = NVMLCollector()
            except NVMLNotAvailable:
                self._nvml = None

        self._tg = anyio.create_task_group()
        await self._tg.__aenter__()
        self._tg.start_soon(self._loop)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._running = False
        if self._tg is not None:
            await self._tg.__aexit__(exc_type, exc, tb)
            self._tg = None
        if self._nvml is not None:
            self._nvml.shutdown()
            self._nvml = None

    async def _loop(self) -> None:
        while self._running:
            t_s = time.time()

            if self._nvml is not None:
                try:
                    samples = self._nvml.sample(t_s)
                    # keep only selected gpu index if exists
                    for s in samples:
                        if s.index == self.gpu_index:
                            self._samples.append(s)
                except Exception:
                    # tolerate sampling errors
                    pass

            if self.enable_torch_fragmentation:
                fr = _torch_fragmentation_ratio()
                if fr is not None and fr > 0:
                    self._frag_samples.append(float(fr))

            await anyio.sleep(self.sample_interval_s)

    def summary(self) -> ServerMetrics:
        """
        Compute ServerMetrics:
        - vram_peak_mb: max NVML used memory
        - vram_avg_mb: avg NVML used memory
        - vram_fragmentation_ratio: torch reserved/allocated median (if available) else 1.0
        - gpu_util_avg: avg NVML GPU utilization if available
        """
        used = [s.memory_used_mb for s in self._samples]
        util = [s.utilization_gpu for s in self._samples if s.utilization_gpu is not None]

        vram_peak = max(used) if used else 0.0
        vram_avg = mean(used) if used else 0.0

        # fragmentation: prefer median of torch ratio samples
        frag = percentile(self._frag_samples, 50.0) if self._frag_samples else 1.0

        gpu_util_avg = mean(util) if util else None

        return ServerMetrics(
            vram_peak_mb=float(vram_peak),
            vram_avg_mb=float(vram_avg),
            vram_fragmentation_ratio=float(frag),
            gpu_util_avg=(float(gpu_util_avg) if gpu_util_avg is not None else None),
        )

    def debug_samples(self) -> dict[str, Any]:
        return {
            "nvml_samples": [asdict(s) for s in self._samples],
            "torch_frag_samples": list(self._frag_samples),
        }
