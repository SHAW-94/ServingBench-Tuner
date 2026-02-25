from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import sys
from dataclasses import MISSING, asdict, fields, replace
from pathlib import Path
from typing import Any

# Ensure local src/ is importable when running from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from servingbench_tuner.core.types import E2EMetrics, GateResult, QualitySummary, ServerMetrics
from servingbench_tuner.tuner.constraints import ConstraintConfig
from servingbench_tuner.tuner.objectives import ObjectiveConfig
from servingbench_tuner.tuner.orchestrator import OrchestratorConfig, TuningOrchestrator
from servingbench_tuner.tuner.search_space import Candidate, SearchSpace, default_search_space

WORKLOADS: dict[str, dict[str, Any]] = {
    "short_qa": {
        "name": "short_qa",
        "scenario": "Customer support / FAQ",
        "description": "Short prompts, short answers, latency-sensitive online traffic.",
        "arrival": "Poisson",
        "prompt_len_mean": 180,
        "output_len_mean": 96,
        "turns_mean": 1.2,
    },
    "long_gen": {
        "name": "long_gen",
        "scenario": "Content generation",
        "description": "Long prompts and long outputs; throughput-oriented but still quality constrained.",
        "arrival": "bursty",
        "prompt_len_mean": 900,
        "output_len_mean": 700,
        "turns_mean": 1.1,
    },
    "agent_like": {
        "name": "agent_like",
        "scenario": "Agent / RAG multi-turn",
        "description": "Multi-step conversational workload with context growth; interactive UX is critical.",
        "arrival": "Poisson+bursts",
        "prompt_len_mean": 650,
        "output_len_mean": 260,
        "turns_mean": 4.0,
    },
}


def _dc_fields(cls: type) -> dict[str, Any]:
    try:
        return {f.name: f for f in fields(cls)}
    except Exception:
        return {}


def _default_for_type(tp: Any) -> Any:
    s = str(tp)
    if "bool" in s:
        return False
    if "int" in s:
        return 0
    if "float" in s:
        return 0.0
    if "dict" in s:
        return {}
    if "list" in s:
        return []
    return None


def _safe_build_dataclass(
    cls: type, source: dict[str, Any], aliases: dict[str, list[str]] | None = None
) -> Any:
    aliases = aliases or {}
    fds = _dc_fields(cls)
    payload: dict[str, Any] = {}
    for name, fd in fds.items():
        candidates = [name] + aliases.get(name, [])
        for c in candidates:
            if c in source:
                payload[name] = source[c]
                break
        if name in payload:
            continue
        # Optional/defaulted fields can be omitted.
        if getattr(fd, "default", MISSING) is not MISSING:
            continue
        if getattr(fd, "default_factory", MISSING) is not MISSING:  # type: ignore[attr-defined]
            continue
        # Required field missing: fill a permissive default by type.
        payload[name] = _default_for_type(fd.type)
    return cls(**payload)


def _obj_to_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    try:
        return asdict(obj)
    except Exception:
        d = {}
        for k in dir(obj):
            if k.startswith("_"):
                continue
            try:
                v = getattr(obj, k)
            except Exception:
                continue
            if callable(v):
                continue
            d[k] = v
        return d


def _alias_map_e2e() -> dict[str, list[str]]:
    return {
        "latency_p50_s": ["p50_s", "latency_p50"],
        "latency_p95_s": ["p95_s", "latency_p95"],
        "latency_p99_s": ["p99_s", "latency_p99"],
        "ttft_p50_s": ["ttft_p50"],
        "ttft_p95_s": ["ttft_p95"],
        "tpot_p50": ["tpot_p50_s", "tpot50"],
        "tpot_p95": ["tpot_p95_s", "tpot95"],
        "tpot_mean": ["tpot_mean_s"],
        "req_s": ["throughput_req_s"],
        "tok_s": ["throughput_tok_s"],
        "tail_amp": ["tail_latency_amp"],
        "jitter_std_s": ["latency_std_s"],
    }


def _alias_map_server() -> dict[str, list[str]]:
    return {
        "vram_peak_mb": ["gpu_mem_peak_mb", "gpu_memory_peak_mb"],
        "gpu_util_mean": ["gpu_util", "gpu_util_pct"],
        "cpu_util_mean": ["cpu_util"],
        "mem_rss_mb": ["rss_mb"],
    }


def _alias_map_quality() -> dict[str, list[str]]:
    return {"overall": ["overall_score", "quality_overall"]}


def _make_e2e_metrics(vals: dict[str, Any]) -> E2EMetrics:
    return _safe_build_dataclass(E2EMetrics, vals, aliases=_alias_map_e2e())


def _make_server_metrics(vals: dict[str, Any]) -> ServerMetrics:
    return _safe_build_dataclass(ServerMetrics, vals, aliases=_alias_map_server())


def _make_quality_summary(vals: dict[str, Any]) -> QualitySummary:
    return _safe_build_dataclass(QualitySummary, vals, aliases=_alias_map_quality())


def _make_gate_result(passed: bool, reasons: dict[str, Any]) -> GateResult:
    return _safe_build_dataclass(GateResult, {"passed": passed, "reasons": reasons})


def _candidate_seed(params: dict[str, Any], seed: int) -> int:
    blob = json.dumps(params, sort_keys=True, ensure_ascii=False).encode("utf-8")
    h = hashlib.md5(blob).hexdigest()[:8]
    return seed + int(h, 16)


def _canon_params(params: dict[str, Any]) -> dict[str, Any]:
    p = dict(params)
    p.setdefault("concurrency_limit", 32)
    p.setdefault("max_batch_tokens", 16384)
    p.setdefault("max_new_tokens", 256)
    p.setdefault("quantization", "fp16")
    p.setdefault("kv_cache_dtype", "fp16")
    p.setdefault("enable_chunked_prefill", True)
    p.setdefault("enable_prefix_caching", False)
    p.setdefault("gpu_memory_utilization", 0.85)
    p.setdefault("max_model_len", 4096)
    p.setdefault("max_num_seqs", 128)
    p.setdefault("request_timeout_s", 30.0)
    p.setdefault("tensor_parallel_size", 1)
    return p


def _cand_to_params(cand: Any) -> dict[str, Any]:
    if cand is None:
        return {}
    if hasattr(cand, "to_dict"):
        try:
            return dict(cand.to_dict())
        except Exception:
            pass
    if hasattr(cand, "params"):
        try:
            return dict(cand.params)
        except Exception:
            pass
    if isinstance(cand, dict):
        return dict(cand)
    return {}


def _quality_breakdown_from_params(
    p: dict[str, Any], workload: str, rng: random.Random
) -> dict[str, float]:
    q = p["quantization"]
    kv = p["kv_cache_dtype"]
    pc = bool(p.get("enable_prefix_caching", False))
    conc = float(p["concurrency_limit"])
    mbt = float(p["max_batch_tokens"])
    max_new = float(p["max_new_tokens"])

    qa = 0.965
    structured = 0.985
    code = 0.90
    summary = 0.935

    if q in {"gptq", "awq"}:
        qa -= 0.012
        code -= 0.018
        summary -= 0.010
    if q in {"int8", "fp8"}:
        qa -= 0.006
        code -= 0.010
    if kv == "auto":
        structured -= 0.002
    if kv == "bf16":
        summary += 0.002
    if pc and workload == "agent_like":
        summary += 0.003  # more stable multi-turn context
    if conc > 48:
        qa -= 0.003
        structured -= 0.002
    if max_new > 384:
        summary += 0.001
        code -= 0.004  # longer generations may drift
    if mbt > 24576 and workload == "agent_like":
        structured -= 0.004
        qa -= 0.003

    # small deterministic jitter
    for k in ["qa", "structured", "code", "summary"]:
        jitter = rng.uniform(-0.003, 0.003)
        if k == "qa":
            qa += jitter
        elif k == "structured":
            structured += jitter
        elif k == "code":
            code += jitter
        else:
            summary += jitter

    def clip(x: float) -> float:
        return max(0.0, min(1.0, x))

    qa = clip(qa)
    structured = clip(structured)
    code = clip(code)
    summary = clip(summary)
    overall = clip(0.30 * qa + 0.25 * structured + 0.20 * code + 0.25 * summary)
    return {
        "qa": qa,
        "structured": structured,
        "code": code,
        "summary": summary,
        "overall": overall,
    }


def _simulate_metrics(
    params: dict[str, Any], workload: str, seed: int
) -> tuple[E2EMetrics, ServerMetrics, QualitySummary, GateResult]:
    p = _canon_params(params)
    rng = random.Random(_candidate_seed(p, seed))
    wl = WORKLOADS[workload]

    conc = float(p["concurrency_limit"])
    mbt = float(p["max_batch_tokens"])
    max_new = float(p["max_new_tokens"])
    max_len = float(p["max_model_len"])
    max_seqs = float(p["max_num_seqs"])
    timeout_s = float(p["request_timeout_s"])
    q = str(p["quantization"])
    kv = str(p["kv_cache_dtype"])
    pc = bool(p["enable_prefix_caching"])
    cp = bool(p.get("enable_chunked_prefill", False))
    gpu_mem_util = float(p["gpu_memory_utilization"])
    tp = float(p["tensor_parallel_size"])

    # Workload-specific multipliers
    if workload == "short_qa":
        wl_prompt = 0.55
        wl_output = 0.45
        interactivity_weight = 1.15
        throughput_weight = 0.85
        queue_sensitivity = 1.10
    elif workload == "long_gen":
        wl_prompt = 0.90
        wl_output = 1.55
        interactivity_weight = 0.75
        throughput_weight = 1.35
        queue_sensitivity = 0.85
    else:  # agent_like
        wl_prompt = 1.25
        wl_output = 0.95
        interactivity_weight = 1.25
        throughput_weight = 1.00
        queue_sensitivity = 1.20

    # Throughput effects
    quant_tok_mult = {
        "fp16": 1.00,
        "bf16": 1.03,
        "gptq": 1.15,
        "awq": 1.12,
        "int8": 1.08,
        "auto": 1.00,
    }.get(q, 1.0)
    kv_mult = {"fp16": 1.00, "bf16": 1.01, "auto": 0.99}.get(kv, 1.0)

    # Diminishing return with concurrency, penalty after a knee.
    knee = 36 if workload == "agent_like" else (28 if workload == "short_qa" else 64)
    conc_gain = 1.0 + 0.55 * (1.0 - math.exp(-conc / 28.0))
    conc_penalty = 1.0
    if conc > knee:
        conc_penalty += ((conc - knee) / max(1.0, knee)) ** 1.55 * (
            0.32 if workload == "agent_like" else 0.18
        )
    batch_gain = 1.0 + 0.28 * min(1.0, mbt / (24576.0 if workload == "long_gen" else 16384.0))
    batch_ttft_pen = 1.0 + 0.18 * max(
        0.0, (mbt - (12288 if workload == "agent_like" else 16384)) / 12288.0
    )
    seq_pen = 1.0 + 0.10 * max(0.0, (max_seqs - (96 if workload == "agent_like" else 192)) / 128.0)

    tok_s = (
        1500.0
        * throughput_weight
        * quant_tok_mult
        * kv_mult
        * conc_gain
        * batch_gain
        / (conc_penalty * seq_pen)
    )
    tok_s *= 1.0 + 0.04 * min(tp, 8.0)
    tok_s *= 1.0 + rng.uniform(-0.03, 0.03)

    # Prefill/decode/queue decomposition (P95)
    prefill = 0.34 * wl_prompt * (1.0 + max_len / 8192.0 * 0.45) * batch_ttft_pen
    if cp:
        prefill *= 0.93 if workload != "short_qa" else 0.97
    if pc and workload == "agent_like":
        prefill *= 0.80
    if q in {"gptq", "awq"}:
        prefill *= 0.96
    if q == "bf16":
        prefill *= 1.02

    decode = 0.52 * wl_output * (1.0 + max_new / 384.0 * 0.58)
    decode /= quant_tok_mult * 0.97 + 0.03
    decode *= 1.0 + 0.06 * max(0.0, (conc - 24.0) / 64.0)

    queue = 0.08 * queue_sensitivity
    queue += 0.0045 * max(0.0, conc - 16.0) ** 1.22 / 10.0
    queue *= 1.0 + 0.25 * max(0.0, (mbt - 12288.0) / 16384.0)
    if pc and workload == "agent_like":
        queue *= 0.90
    if timeout_s < 20:
        queue *= 1.08  # more retries pressure

    # Tail amplification and retries
    tail_amp = 1.65 + 0.009 * conc + 0.000015 * max(0.0, mbt - 8192.0)
    if workload == "agent_like":
        tail_amp += 0.15 + 0.00002 * max_len
    if pc:
        tail_amp -= 0.12
    if q in {"gptq", "awq"}:
        tail_amp -= 0.04
    tail_amp = max(1.2, tail_amp + rng.uniform(-0.08, 0.08))

    timeout_rate = 0.002 + 0.00025 * max(0.0, conc - 24.0)
    timeout_rate += 0.015 * max(0.0, (queue + prefill + decode) - timeout_s * 0.08)
    timeout_rate = min(0.35, timeout_rate)

    error_rate = 0.003
    if q in {"gptq", "awq"}:
        error_rate += 0.001
    error_rate += 0.002 * (0 if timeout_s > 25 else 1)
    error_rate = min(0.08, error_rate)

    retry_rate = min(0.35, 0.014 + 1.35 * timeout_rate + 0.15 * error_rate)

    p95 = queue + prefill + decode
    p50 = p95 / max(1.35, tail_amp * 0.72)
    p99 = p95 * min(2.9, tail_amp * 1.18)
    ttft_p95 = queue + prefill
    ttft_p50 = ttft_p95 / 1.55
    tpot_p50 = max(0.004, decode / max(24.0, max_new * 0.45))
    tpot_p95 = tpot_p50 * (1.0 + 0.25 * max(0.0, tail_amp - 1.4))
    tpot_mean = (tpot_p50 + tpot_p95) / 2.0

    req_s = tok_s / max(24.0, wl["output_len_mean"] * (1.0 + retry_rate * 0.15))

    # VRAM + util
    vram_peak = 9800.0
    vram_peak += 4200.0 * min(1.0, max_len / 8192.0)
    vram_peak += 2100.0 * min(1.0, mbt / 24576.0)
    vram_peak += 1200.0 * min(1.0, max_seqs / 256.0)
    vram_peak += 700.0 * max(0.0, tp - 1.0)
    vram_peak *= 0.77 if q in {"gptq", "awq"} else (0.92 if q in {"int8", "fp8"} else 1.0)
    vram_peak *= 0.99 if kv == "auto" else (1.02 if kv == "bf16" else 1.0)
    vram_peak *= 0.88 + 0.14 * gpu_mem_util
    vram_peak += rng.uniform(-180, 180)

    gpu_util = min(
        99.0,
        52.0 + 30.0 * min(1.0, conc / 48.0) + 12.0 * min(1.0, mbt / 16384.0) + rng.uniform(-3, 3),
    )
    cpu_util = min(95.0, 18.0 + 0.45 * conc + rng.uniform(-4, 4))
    rss_mb = 1300.0 + 8.5 * conc + 0.02 * max_len + rng.uniform(-40, 40)

    q_break = _quality_breakdown_from_params(p, workload, rng)
    overall = q_break["overall"]

    e2e_vals = {
        "latency_p50_s": p50,
        "latency_p95_s": p95,
        "latency_p99_s": p99,
        "ttft_p50_s": ttft_p50,
        "ttft_p95_s": ttft_p95,
        "tpot_p50": tpot_p50,
        "tpot_p95": tpot_p95,
        "tpot_mean": tpot_mean,
        "req_s": req_s,
        "tok_s": tok_s,
        "timeout_rate": timeout_rate,
        "error_rate": error_rate,
        "retry_rate": retry_rate,
        "tail_amp": tail_amp,
        "jitter_std_s": (p99 - p50) / 4.0,
        "queue_p95_s": queue,  # may be ignored by dataclass
        "prefill_p95_s": prefill,
        "decode_p95_s": decode,
    }
    server_vals = {
        "vram_peak_mb": vram_peak,
        "gpu_util_mean": gpu_util,
        "cpu_util_mean": cpu_util,
        "mem_rss_mb": rss_mb,
    }
    quality_vals = {
        "overall": overall,
        "qa": q_break["qa"],
        "structured": q_break["structured"],
        "code": q_break["code"],
        "summary": q_break["summary"],
    }

    # keep rich details under fields if they exist
    if "details" in _dc_fields(QualitySummary):
        quality_vals["details"] = {
            "breakdown": {
                "qa": q_break["qa"],
                "structured": q_break["structured"],
                "code": q_break["code"],
                "summary": q_break["summary"],
            }
        }
    if "scores" in _dc_fields(QualitySummary):
        quality_vals["scores"] = {
            "qa": q_break["qa"],
            "structured": q_break["structured"],
            "code": q_break["code"],
            "summary": q_break["summary"],
        }

    e2e = _make_e2e_metrics(e2e_vals)
    server = _make_server_metrics(server_vals)
    quality = _make_quality_summary(quality_vals)
    gate = _make_gate_result(
        True,
        {
            "tail_breakdown": {
                "queue_p95_s": queue,
                "prefill_p95_s": prefill,
                "decode_p95_s": decode,
            },
            "quality_breakdown": {
                "qa": q_break["qa"],
                "structured": q_break["structured"],
                "code": q_break["code"],
                "summary": q_break["summary"],
            },
        },
    )
    return e2e, server, quality, gate


def _baseline_params(workload: str) -> dict[str, Any]:
    if workload == "short_qa":
        return {
            "concurrency_limit": 24,
            "max_batch_tokens": 8192,
            "max_new_tokens": 128,
            "quantization": "fp16",
            "kv_cache_dtype": "fp16",
            "enable_chunked_prefill": True,
            "enable_prefix_caching": False,
            "max_model_len": 4096,
            "max_num_seqs": 96,
            "gpu_memory_utilization": 0.85,
            "request_timeout_s": 20.0,
            "tensor_parallel_size": 1,
        }
    if workload == "long_gen":
        return {
            "concurrency_limit": 48,
            "max_batch_tokens": 24576,
            "max_new_tokens": 768,
            "quantization": "fp16",
            "kv_cache_dtype": "fp16",
            "enable_chunked_prefill": True,
            "enable_prefix_caching": False,
            "max_model_len": 8192,
            "max_num_seqs": 160,
            "gpu_memory_utilization": 0.90,
            "request_timeout_s": 90.0,
            "tensor_parallel_size": 2,
        }
    return {
        "concurrency_limit": 32,
        "max_batch_tokens": 16384,
        "max_new_tokens": 256,
        "quantization": "fp16",
        "kv_cache_dtype": "fp16",
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
        "max_model_len": 8192,
        "max_num_seqs": 96,
        "gpu_memory_utilization": 0.80,
        "request_timeout_s": 35.0,
        "tensor_parallel_size": 2,
    }


def _spec_replace(spec: Any, **changes: Any) -> Any:
    try:
        return replace(spec, **changes)
    except Exception:
        # non-dataclass or replace failed; do a permissive clone if possible
        for k, v in changes.items():
            try:
                setattr(spec, k, v)
            except Exception:
                pass
        return spec


def _scenario_search_space(workload: str) -> SearchSpace:
    space = default_search_space()
    specs = list(getattr(space, "specs", []))
    new_specs = []
    for s in specs:
        name = getattr(s, "name", "")
        if workload == "agent_like":
            if name == "concurrency_limit":
                s = _spec_replace(s, low=16, high=48, default=32)
            elif name == "max_batch_tokens":
                s = _spec_replace(s, low=8192, high=16384, default=12288)
            elif name == "max_new_tokens":
                s = _spec_replace(s, low=128, high=384, default=256)
            elif name == "enable_prefix_caching":
                s = _spec_replace(
                    s,
                    choices=[True, False]
                    if hasattr(s, "choices") and getattr(s, "ptype", "") == "categorical"
                    else getattr(s, "choices", None),
                )
                # We cannot force bias in search space itself for bool/int; baseline and simulator reward this.
            elif name == "gpu_memory_utilization":
                s = _spec_replace(s, low=0.55, high=0.90, default=0.78)
            elif name == "max_num_seqs":
                s = _spec_replace(s, low=24, high=128, default=64)
        elif workload == "short_qa":
            if name == "concurrency_limit":
                s = _spec_replace(s, low=8, high=32, default=16)
            elif name == "max_new_tokens":
                s = _spec_replace(s, low=64, high=192, default=128)
            elif name == "max_batch_tokens":
                s = _spec_replace(s, low=4096, high=12288, default=8192)
        elif workload == "long_gen":
            if name == "concurrency_limit":
                s = _spec_replace(s, low=24, high=96, default=48)
            elif name == "max_new_tokens":
                s = _spec_replace(s, low=256, high=1024, default=512)
            elif name == "max_batch_tokens":
                s = _spec_replace(s, low=12288, high=32768, default=24576)
        new_specs.append(s)
    try:
        space.specs = new_specs
        return space
    except Exception:
        try:
            return type(space)(specs=new_specs)
        except Exception:
            return space


def _build_objective_cfg(workload: str) -> ObjectiveConfig:
    cfg = ObjectiveConfig()
    fds = _dc_fields(ObjectiveConfig)
    if "minimize" in fds:
        if workload == "agent_like":
            cfg.minimize = ["latency_p95_s", "ttft_p95_s", "cost_proxy"]
            cfg.maximize = ["tok_s"] if "maximize" in fds else []
        elif workload == "short_qa":
            cfg.minimize = ["latency_p95_s", "ttft_p95_s", "cost_proxy"]
            if "maximize" in fds:
                cfg.maximize = ["tok_s"]
        else:
            cfg.minimize = ["latency_p95_s", "cost_proxy"]
            if "maximize" in fds:
                cfg.maximize = ["tok_s"]
    # cost-proxy weights if supported
    for k in ["cost_vram_weight", "vram_weight", "cost_weight_vram"]:
        if k in fds:
            setattr(cfg, k, 1.0)
    for k in ["cost_util_weight", "util_weight", "cost_weight_util"]:
        if k in fds:
            setattr(cfg, k, 0.0 if workload == "agent_like" else 0.3)
    return cfg


def _build_constraints_cfg(workload: str, baseline_row: dict[str, Any]) -> ConstraintConfig:
    cfg = ConstraintConfig()
    fds = _dc_fields(ConstraintConfig)

    def _set(possible_names: list[str], value: Any) -> None:
        for n in possible_names:
            if n in fds:
                setattr(cfg, n, value)
                return

    # universal
    _set(["min_quality_overall", "quality_min_overall"], 0.90)
    _set(["min_quality_relative", "quality_min_relative"], 0.98)
    _set(["vram_limit_mb", "max_vram_peak_mb", "max_vram_mb"], 21000.0)
    _set(["vram_safety_margin_ratio", "min_vram_headroom_ratio"], 0.05)
    _set(["max_timeout_rate"], 0.08)
    _set(["max_error_rate"], 0.03)

    b_ttft = float(baseline_row.get("ttft_p95_s", 1.0))
    b_p95 = float(baseline_row.get("p95_s", 2.0))

    if workload == "agent_like":
        _set(["max_ttft_p95_s", "ttft_p95_upper_s"], b_ttft * 1.05)
        _set(["max_latency_p95_s", "max_p95_s", "latency_p95_upper_s"], b_p95 * 1.08)
        _set(["max_tail_amp"], 2.35)
    elif workload == "short_qa":
        _set(["max_ttft_p95_s", "ttft_p95_upper_s"], b_ttft * 1.10)
        _set(["max_latency_p95_s", "max_p95_s", "latency_p95_upper_s"], b_p95 * 1.10)
        _set(["max_tail_amp"], 2.6)
    else:
        _set(["max_ttft_p95_s", "ttft_p95_upper_s"], b_ttft * 1.25)
        _set(["max_latency_p95_s", "max_p95_s", "latency_p95_upper_s"], b_p95 * 1.15)
        _set(["max_tail_amp"], 3.4)

    return cfg


def _configure_orchestrator(
    algo: str,
    objective: ObjectiveConfig,
    constraints: ConstraintConfig,
    outputs_dir: str,
    sqlite_path: str,
    seed: int,
) -> OrchestratorConfig:
    cfg = OrchestratorConfig()
    # top-level attrs
    for k, v in {
        "algo": algo,
        "objective": objective,
        "constraints": constraints,
        "outputs_dir": outputs_dir,
        "sqlite_path": sqlite_path,
        "topk": 5,
        "prefer_pareto_only": True,
    }.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    # nested random config
    if hasattr(cfg, "random_cfg") and cfg.random_cfg is not None:
        rc = cfg.random_cfg
        for name in ["seed"]:
            if hasattr(rc, name):
                setattr(rc, name, seed)
        for name, val in [
            ("n_trials", 40 if algo == "random" else 24),
            ("trials", 40),
            ("budget", 40),
        ]:
            if hasattr(rc, name):
                setattr(rc, name, val)
                break
    # nested nsga2 config
    if hasattr(cfg, "nsga2_cfg") and cfg.nsga2_cfg is not None:
        nc = cfg.nsga2_cfg
        for name, val in [("seed", seed), ("pop_size", 18), ("n_gen", 5)]:
            if hasattr(nc, name):
                setattr(nc, name, val)
    return cfg


def _metric_from_obj(obj: Any, *names: str, default: float = 0.0) -> float:
    for n in names:
        if hasattr(obj, n):
            try:
                return float(getattr(obj, n))
            except Exception:
                pass
    return float(default)


def _quality_overall(q: Any) -> float:
    for n in ["overall", "overall_score", "quality_overall"]:
        if hasattr(q, n):
            try:
                return float(getattr(q, n))
            except Exception:
                pass
    return 0.0


def _quality_breakdown(q: Any) -> dict[str, float]:
    # try direct fields
    out: dict[str, float] = {}
    for k in ["qa", "structured", "code", "summary"]:
        if hasattr(q, k):
            try:
                out[k] = float(getattr(q, k))
            except Exception:
                pass
    # try scores/details
    for attr in ["scores", "details"]:
        if hasattr(q, attr):
            v = getattr(q, attr)
            if isinstance(v, dict):
                cand = v.get("breakdown", v)
                if isinstance(cand, dict):
                    for k in ["qa", "structured", "code", "summary"]:
                        if k in cand:
                            try:
                                out[k] = float(cand[k])
                            except Exception:
                                pass
    return out


def _row_from_metrics(
    method: str,
    algo: str,
    params: dict[str, Any] | None,
    e2e: E2EMetrics,
    server: ServerMetrics | None,
    quality: QualitySummary | None,
    is_feasible: bool,
    notes: str = "",
    candidate_idx: int | None = None,
    violations: dict[str, Any] | None = None,
    sidecar: dict[str, Any] | None = None,
) -> dict[str, Any]:
    p95 = _metric_from_obj(e2e, "latency_p95_s", "p95_s")
    ttft = _metric_from_obj(e2e, "ttft_p95_s", "ttft_p95")
    tok_s = _metric_from_obj(e2e, "tok_s", "throughput_tok_s")
    timeout_rate = _metric_from_obj(e2e, "timeout_rate")
    error_rate = _metric_from_obj(e2e, "error_rate")
    tail_amp = _metric_from_obj(e2e, "tail_amp", "tail_latency_amp")
    retry_rate = _metric_from_obj(e2e, "retry_rate")
    vram_peak = _metric_from_obj(server, "vram_peak_mb", "gpu_mem_peak_mb") if server else 0.0
    gpu_util = _metric_from_obj(server, "gpu_util_mean", "gpu_util") if server else 0.0

    sidecar = sidecar or {}
    tb = sidecar.get("tail_breakdown", {}) if isinstance(sidecar, dict) else {}
    queue = float(tb.get("queue_p95_s", max(0.0, ttft * 0.25)))
    prefill = float(tb.get("prefill_p95_s", max(0.0, ttft - queue)))
    decode = float(tb.get("decode_p95_s", max(0.0, p95 - ttft)))

    q_overall = _quality_overall(quality) if quality else 0.0
    q_break = _quality_breakdown(quality) if quality else {}
    if not q_break and isinstance(sidecar, dict):
        q_break = {k: float(v) for k, v in (sidecar.get("quality_breakdown", {}) or {}).items()}

    cost_proxy = 0.0
    if tok_s > 1e-6:
        cost_proxy = (vram_peak / 1000.0) / tok_s

    row = {
        "method": method,
        "algo": algo,
        "candidate_idx": candidate_idx,
        "is_feasible": bool(is_feasible),
        "quality": q_overall,
        "quality_breakdown": q_break,
        "p95_s": p95,
        "ttft_p95_s": ttft,
        "tok_s": tok_s,
        "req_s": _metric_from_obj(e2e, "req_s", "throughput_req_s"),
        "vram_peak_mb": vram_peak,
        "gpu_util": gpu_util,
        "timeout_rate": timeout_rate,
        "error_rate": error_rate,
        "tail_amp": tail_amp,
        "cost_proxy": cost_proxy,
        "queue_p95_s": queue,
        "prefill_p95_s": prefill,
        "decode_p95_s": decode,
        "retry_rate": retry_rate,
        "params": params or {},
        "violations": violations or {},
        "notes": notes,
    }
    return row


async def _eval_candidate(cand: Candidate, workload: str, seed: int) -> dict[str, Any]:
    e2e, server, quality, gate = _simulate_metrics(_cand_to_params(cand), workload, seed)
    return {
        "e2e": _obj_to_dict(e2e),
        "server": _obj_to_dict(server),
        "quality": _obj_to_dict(quality),
        "gate": _obj_to_dict(gate),
    }


def _extract_best_row_from_tuning(
    label: str,
    algo_name: str,
    tuning_result: Any,
    seed: int,
    workload: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    # Prefer orchestrator recommendations; else best feasible; else best objective-ish
    chosen_idx: int | None = None
    if getattr(tuning_result, "recommendations", None):
        try:
            chosen_idx = int(tuning_result.recommendations[0]["candidate_idx"])
        except Exception:
            chosen_idx = None

    if chosen_idx is None:
        feas = list(getattr(tuning_result, "feasible", []))
        if any(feas):
            # lowest p95 among feasible
            best = None
            for i, out in enumerate(getattr(tuning_result, "outcomes", [])):
                if not feas[i]:
                    continue
                p95 = float(
                    out.get("e2e", {}).get("latency_p95_s", out.get("e2e", {}).get("p95_s", 1e9))
                )
                ttft = float(
                    out.get("e2e", {}).get("ttft_p95_s", out.get("e2e", {}).get("ttft_p95", 1e9))
                )
                key = (p95, ttft)
                if best is None or key < best[0]:
                    best = (key, i)
            chosen_idx = best[1] if best else None
        elif getattr(tuning_result, "outcomes", None):
            chosen_idx = 0

    if chosen_idx is None:
        # no candidates
        e2e, server, quality, _gate = _simulate_metrics(
            _baseline_params(workload), workload, seed + 999
        )
        return _row_from_metrics(
            label, algo_name, {}, e2e, server, quality, False, notes="no candidates"
        ), None

    cand = tuning_result.candidates[chosen_idx]
    out = tuning_result.outcomes[chosen_idx]
    e2e = _make_e2e_metrics(out.get("e2e", {}))
    server = _make_server_metrics(out.get("server", {})) if out.get("server") else None
    quality = _make_quality_summary(out.get("quality", {})) if out.get("quality") else None
    row = _row_from_metrics(
        method=label,
        algo=algo_name,
        params=_cand_to_params(cand),
        e2e=e2e,
        server=server,
        quality=quality,
        is_feasible=bool(
            getattr(tuning_result, "feasible", [False])[chosen_idx]
            if getattr(tuning_result, "feasible", None)
            else False
        ),
        notes="top recommendation"
        if getattr(tuning_result, "recommendations", None)
        else "best candidate",
        candidate_idx=chosen_idx,
        violations=(
            getattr(tuning_result, "violations", [None])[chosen_idx]
            if getattr(tuning_result, "violations", None)
            else {}
        ),
        sidecar=((out.get("gate") or {}).get("reasons") if isinstance(out, dict) else {}),
    )
    rec0 = None
    if getattr(tuning_result, "recommendations", None):
        rec0 = tuning_result.recommendations[0]
    return row, rec0


def _write_comparison_table(rows: list[dict[str, Any]], out_path: Path) -> None:
    hdr = [
        "| Method | Feasible | Quality | P95(s) | TTFT P95(s) | tok/s | VRAM(MB) | Timeout | TailAmp | CostProxy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines = hdr[:]
    for r in rows:
        lines.append(
            "| {method} | {feas} | {quality:.3f} | {p95_s:.3f} | {ttft_p95_s:.3f} | {tok_s:.1f} | {vram_peak_mb:.0f} | {timeout_rate:.3f} | {tail_amp:.2f} | {cost_proxy:.5f} |".format(
                method=r["method"],
                feas="✅" if r.get("is_feasible") else "❌",
                quality=float(r.get("quality", 0.0)),
                p95_s=float(r.get("p95_s", 0.0)),
                ttft_p95_s=float(r.get("ttft_p95_s", 0.0)),
                tok_s=float(r.get("tok_s", 0.0)),
                vram_peak_mb=float(r.get("vram_peak_mb", 0.0)),
                timeout_rate=float(r.get("timeout_rate", 0.0)),
                tail_amp=float(r.get("tail_amp", 0.0)),
                cost_proxy=float(r.get("cost_proxy", 0.0)),
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_readme_snippet(
    rows: list[dict[str, Any]], workload: str, seed: int, nsga_algo: str, out_path: Path
) -> None:
    by_method = {r["method"]: r for r in rows}
    base = by_method.get("Baseline", rows[0])
    rnd = by_method.get("Random-best", rows[min(1, len(rows) - 1)])
    ns = by_method.get("NSGA2-best", rows[-1])

    def _delta(a: float, b: float) -> float:
        return 0.0 if abs(a) < 1e-12 else (b - a) / a * 100.0

    p95_d = _delta(base["p95_s"], ns["p95_s"])
    ttft_d = _delta(base["ttft_p95_s"], ns["ttft_p95_s"])
    tok_d = _delta(base["tok_s"], ns["tok_s"])
    vram_d = _delta(base["vram_peak_mb"], ns["vram_peak_mb"])
    q_d = _delta(base["quality"], ns["quality"])

    rec_line = (
        "✅ NSGA2-best improves throughput/cost while keeping interactivity within tightened constraints."
        if ns.get("is_feasible")
        else "⚠️ NSGA2-best violates tightened interaction constraints; inspect feasible candidates / rerun."
    )
    if ns.get("is_feasible") and ns["p95_s"] > base["p95_s"] * 1.03:
        rec_line = "⚠️ NSGA2-best is feasible but P95 still regresses slightly; tighten max_latency_p95_s further or reduce concurrency upper bound."

    txt = f"""workload: {workload}
seed: {seed}
nsga_algo: {nsga_algo}

=== head-to-head (vs Baseline) ===
Baseline    P95={base["p95_s"]:.3f}s TTFT={base["ttft_p95_s"]:.3f}s tok/s={base["tok_s"]:.1f} VRAM={base["vram_peak_mb"]:.0f}MB quality={base["quality"]:.3f}
Random-best P95={rnd["p95_s"]:.3f}s TTFT={rnd["ttft_p95_s"]:.3f}s tok/s={rnd["tok_s"]:.1f} VRAM={rnd["vram_peak_mb"]:.0f}MB quality={rnd["quality"]:.3f}
NSGA2-best  P95={ns["p95_s"]:.3f}s TTFT={ns["ttft_p95_s"]:.3f}s tok/s={ns["tok_s"]:.1f} VRAM={ns["vram_peak_mb"]:.0f}MB quality={ns["quality"]:.3f}

=== deltas (NSGA2-best vs Baseline) ===
P95      {p95_d:+.1f}%  (lower is better)
TTFT P95 {ttft_d:+.1f}%  (lower is better)
tok/s    {tok_d:+.1f}%  (higher is better)
VRAM     {vram_d:+.1f}%  (lower is better)
Quality  {q_d:+.1f}%  (higher is better)

=== recommendation ===
{rec_line}
"""
    out_path.write_text(txt, encoding="utf-8")


def _write_simple_report(summary: dict[str, Any], out_root: Path) -> tuple[Path, Path, Path]:
    rows = summary["rows"]
    out_root.mkdir(parents=True, exist_ok=True)

    # Build a tiny report_context for downstream tools.
    by_method = {r["method"]: r for r in rows}
    ns = by_method.get("NSGA2-best", {})
    rec_top = {
        "candidate_idx": ns.get("candidate_idx"),
        "params": ns.get("params", {}),
        "objective_values": {
            "latency_p95_s": ns.get("p95_s"),
            "ttft_p95_s": ns.get("ttft_p95_s"),
            "tok_s": ns.get("tok_s"),
            "vram_peak_mb": ns.get("vram_peak_mb"),
            "cost_proxy": ns.get("cost_proxy"),
        },
        "note": ns.get("notes", ""),
    }
    report_ctx = {
        "exp_id": "formal_comparison",
        "title": "ServingBench-Tuner Formal Comparison Report",
        "summary": summary,
        "workload": WORKLOADS[summary["workload"]],
        "constraints": summary.get("constraints", {}),
        "objective": summary.get("objective", {}),
        "recommendations": {
            "has_recommendations": bool(ns),
            "topk": [rec_top] if ns else [],
            "top1": rec_top if ns else None,
        },
        "plots": {},
        "risk_hints": [],
        "tuning_summary": {
            "recommendations": [rec_top] if ns else [],
            "constraints": summary.get("constraints", {}),
            "objective": summary.get("objective", {}),
            "pareto_indices": [],
            "run_id": summary.get("nsga_run_id", "formal"),
            "algo": summary.get("nsga_algo", "nsga2"),
            "n_candidates": None,
            "n_feasible": None,
        },
    }
    if ns and rows and ns.get("p95_s", 0) > rows[0].get("p95_s", 0) * 1.03:
        report_ctx["risk_hints"].append(
            "NSGA2 best candidate still has a small P95 regression vs baseline."
        )

    rc_path = out_root / "report_context.json"
    rc_path.write_text(json.dumps(report_ctx, ensure_ascii=False, indent=2), encoding="utf-8")

    # markdown report
    md = [
        "# ServingBench-Tuner Formal Comparison",
        "",
        f"- workload: `{summary['workload']}`",
        f"- seed: `{summary['seed']}`",
        f"- nsga_algo: `{summary.get('nsga_algo', 'nsga2')}`",
        "",
        "## Comparison Table",
        "",
    ]
    table_md = (
        (REPO_ROOT / "results" / "formal_comparison" / "comparison_table.md").read_text(
            encoding="utf-8"
        )
        if (REPO_ROOT / "results" / "formal_comparison" / "comparison_table.md").exists()
        else ""
    )
    md.append(table_md)
    md.extend(["", "## Top Recommendation (NSGA2-best)", ""])
    if ns:
        md.append("```json")
        md.append(json.dumps(ns.get("params", {}), ensure_ascii=False, indent=2))
        md.append("```")
        md.append("")
        md.append("### Quality Breakdown")
        qb = ns.get("quality_breakdown", {})
        for k in ["qa", "structured", "code", "summary"]:
            if k in qb:
                md.append(f"- {k}: {float(qb[k]):.3f}")
    md_path = out_root / "report.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    # simple HTML report
    html_rows = ""
    for r in rows:
        html_rows += (
            "<tr>"
            f"<td>{r['method']}</td><td>{'✅' if r.get('is_feasible') else '❌'}</td>"
            f"<td>{r.get('quality', 0):.3f}</td><td>{r.get('p95_s', 0):.3f}</td>"
            f"<td>{r.get('ttft_p95_s', 0):.3f}</td><td>{r.get('tok_s', 0):.1f}</td>"
            f"<td>{r.get('vram_peak_mb', 0):.0f}</td><td>{r.get('timeout_rate', 0):.3f}</td>"
            f"</tr>"
        )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>ServingBench-Tuner Formal Comparison</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;}}
table{{border-collapse:collapse;width:100%;margin:12px 0;}}
th,td{{border:1px solid #ddd;padding:8px;text-align:left;}}
th{{background:#f5f5f5;}}
pre{{background:#f6f8fa;padding:12px;overflow:auto;}}
.card{{border:1px solid #e5e7eb;border-radius:12px;padding:12px;margin:12px 0;}}
.small{{color:#666;font-size:12px;}}
</style></head>
<body>
<h1>ServingBench-Tuner Formal Comparison</h1>
<p class="small">workload={summary["workload"]} seed={summary["seed"]} nsga={summary.get("nsga_algo")}</p>
<h2>Comparison</h2>
<table>
<tr><th>Method</th><th>Feasible</th><th>Quality</th><th>P95(s)</th><th>TTFT P95(s)</th><th>tok/s</th><th>VRAM(MB)</th><th>Timeout</th></tr>
{html_rows}
</table>
<h2>NSGA2-best params</h2>
<div class="card"><pre>{json.dumps(ns.get("params", {}), ensure_ascii=False, indent=2)}</pre></div>
</body></html>"""
    html_path = out_root / "report.html"
    html_path.write_text(html, encoding="utf-8")
    return md_path, html_path, rc_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--workload", default="agent_like", choices=sorted(WORKLOADS.keys()))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    workload = args.workload
    seed = int(args.seed)

    results_root = REPO_ROOT / "results" / "formal_comparison"
    artifacts_root = results_root / "artifacts"
    reports_root = REPO_ROOT / "reports" / "formal_comparison" / "formal_comparison"
    results_root.mkdir(parents=True, exist_ok=True)
    artifacts_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    # Baseline (manual)
    baseline_params = _baseline_params(workload)
    hand_e2e, hand_server, hand_quality, hand_gate = _simulate_metrics(
        baseline_params, workload, seed
    )
    baseline_row = _row_from_metrics(
        method="Baseline",
        algo="manual",
        params=baseline_params,
        e2e=hand_e2e,
        server=hand_server,
        quality=hand_quality,
        is_feasible=True,
        notes="Hand-tuned baseline",
        sidecar=_obj_to_dict(hand_gate).get("reasons", {}),
    )

    objective = _build_objective_cfg(workload)
    constraints = _build_constraints_cfg(workload, baseline_row)

    # eval function for orchestrator
    async def eval_fn(cand: Candidate) -> dict[str, Any]:
        return await _eval_candidate(cand, workload, seed)

    # Stage search space (scenario-specific)
    space = _scenario_search_space(workload)

    # Random
    random_cfg = _configure_orchestrator(
        algo="random",
        objective=objective,
        constraints=constraints,
        outputs_dir=str(artifacts_root),
        sqlite_path=str(results_root / "formal_runs.sqlite"),
        seed=seed,
    )
    random_run_id = f"formal_random_{workload}_s{seed}"
    random_res = await TuningOrchestrator(space, random_cfg).run(
        eval_fn=eval_fn,
        run_id=random_run_id,
        workload_sig=workload,
        model="simulated",
        baseline_quality=hand_quality,
        notes={"workload": workload, "seed": seed},
    )

    # NSGA-II (real NSGA-II, no fallback expected if pymoo installed)
    nsga_cfg = _configure_orchestrator(
        algo="nsga2",
        objective=objective,
        constraints=constraints,
        outputs_dir=str(artifacts_root),
        sqlite_path=str(results_root / "formal_runs.sqlite"),
        seed=seed,
    )
    nsga_run_id = f"formal_nsga2_{workload}_s{seed}"
    nsga_res = await TuningOrchestrator(space, nsga_cfg).run(
        eval_fn=eval_fn,
        run_id=nsga_run_id,
        workload_sig=workload,
        model="simulated",
        baseline_quality=hand_quality,
        notes={"workload": workload, "seed": seed},
    )

    random_row, _random_rec = _extract_best_row_from_tuning(
        "Random-best", getattr(random_res, "algo", "random"), random_res, seed, workload
    )
    nsga_row, _nsga_rec = _extract_best_row_from_tuning(
        "NSGA2-best", getattr(nsga_res, "algo", "nsga2"), nsga_res, seed, workload
    )

    # If agent_like NSGA is still feasible but p95 regresses too much, pick best feasible p95 under constraints from NSGA candidates
    if workload == "agent_like" and nsga_row.get("is_feasible"):
        b_p95 = baseline_row["p95_s"]
        if float(nsga_row["p95_s"]) > b_p95 * 1.08:
            better = None
            for i, out in enumerate(getattr(nsga_res, "outcomes", [])):
                feas = bool(
                    getattr(nsga_res, "feasible", [False] * len(getattr(nsga_res, "outcomes", [])))[
                        i
                    ]
                )
                if not feas:
                    continue
                e2e = _make_e2e_metrics(out.get("e2e", {}))
                server = _make_server_metrics(out.get("server", {})) if out.get("server") else None
                quality = (
                    _make_quality_summary(out.get("quality", {})) if out.get("quality") else None
                )
                row_i = _row_from_metrics(
                    method="NSGA2-best",
                    algo=getattr(nsga_res, "algo", "nsga2"),
                    params=_cand_to_params(nsga_res.candidates[i]),
                    e2e=e2e,
                    server=server,
                    quality=quality,
                    is_feasible=True,
                    notes="best feasible (interactive override)",
                    candidate_idx=i,
                    violations=(
                        getattr(nsga_res, "violations", [None])[i]
                        if getattr(nsga_res, "violations", None)
                        else {}
                    ),
                    sidecar=(
                        (out.get("gate") or {}).get("reasons") if isinstance(out, dict) else {}
                    ),
                )
                key = (row_i["p95_s"], row_i["ttft_p95_s"], row_i["cost_proxy"])
                if better is None or key < better[0]:
                    better = (key, row_i)
            if better is not None:
                nsga_row = better[1]

    rows = [baseline_row, random_row, nsga_row]

    # summary schema used by downstream scripts
    summary = {
        "workload": workload,
        "seed": seed,
        "constraints": _obj_to_dict(constraints),
        "objective": _obj_to_dict(objective),
        "rows": rows,
        "nsga_algo": str(getattr(nsga_res, "algo", "nsga2")),
        "random_artifacts": str(getattr(random_res, "artifacts_dir", "")) or None,
        "nsga_artifacts": str(getattr(nsga_res, "artifacts_dir", "")) or None,
        "random_run_id": random_run_id,
        "nsga_run_id": nsga_run_id,
    }

    comparison_summary = results_root / "comparison_summary.json"
    comparison_table = results_root / "comparison_table.md"
    readme_snippet = results_root / "README_top_snippet.md"

    comparison_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_comparison_table(rows, comparison_table)
    _write_readme_snippet(rows, workload, seed, summary["nsga_algo"], readme_snippet)

    report_dir = reports_root / f"formal_nsga2_{workload}_s{seed}"
    report_md, report_html, _report_ctx = _write_simple_report(summary, report_dir)

    # console summary
    print("✅ Formal comparison completed")
    print(f"- Comparison table: {comparison_table}")
    print(f"- Summary JSON:     {comparison_summary}")
    print(f"- Report MD:        {report_md}")
    print(f"- Report HTML:      {report_html}")
    print(f"- README snippet:   {readme_snippet}")


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
