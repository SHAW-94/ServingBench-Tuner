#!/usr/bin/env python3
"""
export_env_signature.py

Print/write an environment signature for reproducibility (no docker version).
Captures:
- git commit
- python version
- lockfile hash
- pip-freeze hash
- engine versions (vllm if installed)
- GPU/driver (via nvidia-smi if available)
- CUDA hints (via nvidia-smi/torch if available)

Usage:
  python scripts/export_env_signature.py
  python scripts/export_env_signature.py --out results/artifacts/env_signature.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False
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
    # stable hash: sort lines
    lines = sorted([ln.strip() for ln in out.splitlines() if ln.strip()])
    return _sha256_text("\n".join(lines))


def _package_version(pkg: str) -> str | None:
    try:
        import importlib.metadata as md

        return md.version(pkg)
    except Exception:
        return None


def _nvidia_smi() -> dict[str, Any]:
    # Query minimal GPU info
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
    # CUDA version (as printed by nvidia-smi)
    code2, out2 = _run(["nvidia-smi"])
    if code2 == 0 and out2:
        # naive parse
        cuda_ver = None
        for ln in out2.splitlines():
            if "CUDA Version" in ln:
                # e.g. "| NVIDIA-SMI 535.xx   Driver Version: 535.xx   CUDA Version: 12.2 |"
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
        info["cuda_version"] = torch.version.cuda
    except Exception:
        pass
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="", help="write json to file (optional)")
    ap.add_argument("--lockfile", type=str, default="requirements.lock.txt", help="lockfile path")
    args = ap.parse_args()

    now = datetime.now(timezone.utc).isoformat()

    lock_path = Path(args.lockfile)
    sig: dict[str, Any] = {
        "generated_at": now,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "git_commit": _git_commit(),
        "lockfile": {
            "path": str(lock_path),
            "sha256": _sha256_file(lock_path),
        },
        "pip_freeze_sha256": _pip_freeze_hash(),
        "packages": {
            "servingbench_tuner": _package_version("servingbench-tuner"),
            "vllm": _package_version("vllm"),
            "pymoo": _package_version("pymoo"),
        },
        "nvidia": _nvidia_smi(),
        "torch": _torch_cuda(),
    }

    s = json.dumps(sig, ensure_ascii=False, indent=2)
    print(s)

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(s, encoding="utf-8")
        print(f"\n[OK] Wrote env signature to: {outp}")


if __name__ == "__main__":
    main()
