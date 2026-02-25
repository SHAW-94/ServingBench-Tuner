from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass


@dataclass
class NVMLGPUInfo:
    index: int
    name: str
    uuid: str
    memory_total_mb: float


@dataclass
class NVMLSample:
    t_s: float
    index: int
    memory_used_mb: float
    memory_total_mb: float
    utilization_gpu: float | None = None
    utilization_mem: float | None = None


class NVMLNotAvailable(RuntimeError):
    pass


class NVMLCollector:
    """
    Thin wrapper around pynvml.

    Design goals:
    - optional dependency (CI/CPU should work without it)
    - stable sample dict format
    """

    def __init__(self) -> None:
        try:
            import pynvml  # type: ignore
        except Exception as e:
            raise NVMLNotAvailable("pynvml not installed or NVML unavailable") from e

        self.pynvml = pynvml
        try:
            self.pynvml.nvmlInit()
        except Exception as e:
            raise NVMLNotAvailable("nvmlInit failed") from e

        self._handles = []
        self._infos: list[NVMLGPUInfo] = []
        self._init_devices()

    def _init_devices(self) -> None:
        n = self.pynvml.nvmlDeviceGetCount()
        for i in range(n):
            h = self.pynvml.nvmlDeviceGetHandleByIndex(i)
            name = self.pynvml.nvmlDeviceGetName(h).decode("utf-8", errors="ignore")  # type: ignore
            uuid = self.pynvml.nvmlDeviceGetUUID(h).decode("utf-8", errors="ignore")  # type: ignore
            mem = self.pynvml.nvmlDeviceGetMemoryInfo(h)
            total_mb = float(mem.total) / (1024.0 * 1024.0)
            self._handles.append(h)
            self._infos.append(NVMLGPUInfo(index=i, name=name, uuid=uuid, memory_total_mb=total_mb))

    def list_gpus(self) -> list[NVMLGPUInfo]:
        return list(self._infos)

    def sample(self, t_s: float) -> list[NVMLSample]:
        out: list[NVMLSample] = []
        for info, h in zip(self._infos, self._handles, strict=False):
            mem = self.pynvml.nvmlDeviceGetMemoryInfo(h)
            used_mb = float(mem.used) / (1024.0 * 1024.0)
            total_mb = float(mem.total) / (1024.0 * 1024.0)
            util_gpu = util_mem = None
            try:
                util = self.pynvml.nvmlDeviceGetUtilizationRates(h)
                util_gpu = float(util.gpu)
                util_mem = float(util.memory)
            except Exception:
                pass

            out.append(
                NVMLSample(
                    t_s=float(t_s),
                    index=int(info.index),
                    memory_used_mb=used_mb,
                    memory_total_mb=total_mb,
                    utilization_gpu=util_gpu,
                    utilization_mem=util_mem,
                )
            )
        return out

    def shutdown(self) -> None:
        with suppress(Exception):
            self.pynvml.nvmlShutdown()
