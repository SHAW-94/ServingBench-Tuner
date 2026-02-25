from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .schema import WorkloadSpec


def load_workload_yaml(path: str | Path, resolve_relative_paths: bool = True) -> WorkloadSpec:
    """
    Parse workload.yaml -> WorkloadSpec (Pydantic), with validation.
    If resolve_relative_paths=True, empirical/replay paths are resolved relative to YAML's directory.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"workload yaml not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"workload yaml must be a mapping: {p}")

    spec = WorkloadSpec.model_validate(raw)
    if resolve_relative_paths:
        spec = spec.resolve_paths(p.parent)

    return spec


def dump_workload_yaml(spec: WorkloadSpec, path: str | Path) -> None:
    """
    Dump a WorkloadSpec to YAML (canonical-ish). This is useful for storing
    a "resolved" workload config after path resolution / defaults filled.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    y = yaml.safe_dump(spec.to_dict(), sort_keys=False, allow_unicode=True)
    p.write_text(y, encoding="utf-8")


def canonical_json(spec: WorkloadSpec) -> str:
    """
    Canonical JSON string for hashing / reproducibility signatures.
    """
    obj = spec.to_dict()
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def workload_signature(spec: WorkloadSpec) -> str:
    """
    Stable SHA256 of canonical JSON representation.
    """
    s = canonical_json(spec).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def maybe_apply_overrides(spec: WorkloadSpec, overrides: dict[str, Any] | None) -> WorkloadSpec:
    """
    Apply shallow overrides (dict merge) then re-validate.
    (If you want deep merges, do it before calling this helper.)
    """
    if not overrides:
        return spec
    merged = {**spec.to_dict(), **overrides}
    return WorkloadSpec.model_validate(merged)
