from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return p.returncode, p.stdout.strip()
    except Exception as e:
        return 1, str(e)


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _git_commit() -> str | None:
    code, out = _run(["git", "rev-parse", "HEAD"])
    return out if code == 0 and out else None


def _pip_freeze_hash() -> str | None:
    code, out = _run([sys.executable, "-m", "pip", "freeze"])
    if code != 0 or not out:
        return None
    lines = sorted([ln.strip() for ln in out.splitlines() if ln.strip()])
    return _sha256_text("\n".join(lines))


def _package_version(pkg: str) -> str | None:
    try:
        import importlib.metadata as md

        return md.version(pkg)
    except Exception:
        return None


def _nvidia_smi() -> dict[str, Any]:
    info: dict[str, Any] = {"available": False}
    code, out = _run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
    )
    if code != 0 or not out:
        return info
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"name": parts[0], "driver": parts[1], "memory_total": parts[2]})
    info["available"] = True
    info["gpus"] = gpus

    code2, out2 = _run(["nvidia-smi"])
    if code2 == 0 and out2:
        cuda_ver = None
        for ln in out2.splitlines():
            if "CUDA Version" in ln:
                idx = ln.find("CUDA Version:")
                if idx != -1:
                    cuda_ver = ln[idx:].split("CUDA Version:")[-1].strip().split()[0]
                    break
        info["cuda_version_hint"] = cuda_ver
    return info


def _torch_cuda() -> dict[str, Any]:
    info: dict[str, Any] = {"torch": None, "cuda_available": None, "cuda_version": None}
    try:
        import torch  # type: ignore

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
    except Exception:
        pass
    return info


@dataclass(frozen=True)
class EnvSignature:
    generated_at: str
    git_commit: str | None
    python: str
    platform: dict[str, Any]
    lockfile_path: str
    lockfile_sha256: str | None
    pip_freeze_sha256: str | None
    packages: dict[str, str | None]
    nvidia: dict[str, Any]
    torch: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "git_commit": self.git_commit,
            "python": self.python,
            "platform": self.platform,
            "lockfile": {"path": self.lockfile_path, "sha256": self.lockfile_sha256},
            "pip_freeze_sha256": self.pip_freeze_sha256,
            "packages": self.packages,
            "nvidia": self.nvidia,
            "torch": self.torch,
        }


def collect_env_signature(lockfile: str = "requirements.lock.txt") -> EnvSignature:
    now = datetime.now(timezone.utc).isoformat()
    lock_path = Path(lockfile)

    sig = EnvSignature(
        generated_at=now,
        git_commit=_git_commit(),
        python=sys.version.split()[0],
        platform={
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        lockfile_path=str(lock_path),
        lockfile_sha256=_sha256_file(lock_path),
        pip_freeze_sha256=_pip_freeze_hash(),
        packages={
            "servingbench-tuner": _package_version("servingbench-tuner"),
            "vllm": _package_version("vllm"),
            "pymoo": _package_version("pymoo"),
        },
        nvidia=_nvidia_smi(),
        torch=_torch_cuda(),
    )
    return sig
