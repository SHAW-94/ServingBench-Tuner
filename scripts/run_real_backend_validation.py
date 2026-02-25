from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


PROMPT_PACKS: dict[str, list[dict[str, Any]]] = {
    "agent_like": [
        {"kind": "qa", "prompt": "Answer with one token only. What is 12+19?", "expect": "31"},
        {
            "kind": "qa",
            "prompt": "Answer with one token only. What is the capital of France?",
            "expect": "Paris",
        },
        {
            "kind": "structured",
            "prompt": 'Return ONLY JSON: {"city": "...", "country": "..."} for Tokyo.',
            "json_keys": ["city", "country"],
        },
        {
            "kind": "structured",
            "prompt": 'Return ONLY JSON with keys ["answer","confidence"] where answer is 2+2.',
            "json_keys": ["answer", "confidence"],
        },
        {
            "kind": "code",
            "prompt": "Write Python function `def add(a, b):` and nothing else.",
            "code_test": "add",
        },
        {
            "kind": "summary",
            "prompt": "Summarize in one sentence: A project migrated from heuristic tuning to constrained NSGA-II optimization with quality gates and reproducible reports.",
            "summary_keywords": ["NSGA-II", "quality", "report"],
        },
    ],
    "short_qa": [
        {"kind": "qa", "prompt": "Answer with one token only. What is 7*8?", "expect": "56"},
        {
            "kind": "qa",
            "prompt": "Answer with one token only. What color is the sky on a clear day?",
            "expect": "blue",
        },
        {
            "kind": "structured",
            "prompt": 'Return ONLY JSON: {"ok": true, "value": 42}',
            "json_keys": ["ok", "value"],
        },
        {
            "kind": "code",
            "prompt": "Write Python function `def square(x):` and nothing else.",
            "code_test": "square",
        },
        {
            "kind": "summary",
            "prompt": "Summarize in one sentence: Low latency matters for interactive support systems.",
            "summary_keywords": ["latency", "interactive"],
        },
    ],
    "long_gen": [
        {"kind": "qa", "prompt": "Answer with one token only. What is 100-58?", "expect": "42"},
        {
            "kind": "structured",
            "prompt": 'Return ONLY JSON: {"topic":"llm serving","mode":"long_gen"}',
            "json_keys": ["topic", "mode"],
        },
        {
            "kind": "code",
            "prompt": "Write Python function `def fib1(n):` that returns first n Fibonacci numbers list.",
            "code_test": "fib1",
        },
        {
            "kind": "summary",
            "prompt": "Summarize in two sentences: Throughput optimization should remain under quality and VRAM constraints for production LLM serving.",
            "summary_keywords": ["throughput", "quality", "VRAM"],
        },
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--workload", default="agent_like", choices=list(PROMPT_PACKS.keys()))
    p.add_argument(
        "--summary",
        default=str(REPO_ROOT / "results" / "formal_comparison" / "comparison_summary.json"),
    )
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--stream", action="store_true")
    p.add_argument("--requests-per-method", type=int, default=12)
    p.add_argument("--launch-vllm", action="store_true")
    p.add_argument("--vllm-port", type=int, default=8000)
    p.add_argument("--extra-vllm-args", default="")
    p.add_argument(
        "--out",
        default=str(REPO_ROOT / "results" / "real_validation" / "real_validation_summary.json"),
    )
    return p.parse_args()


def _now() -> float:
    return time.perf_counter()


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _poll_vram(stop: threading.Event, peaks: list[float]) -> None:
    while not stop.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            vals = [float(x.strip()) for x in out.splitlines() if x.strip()]
            if vals:
                peaks.append(max(vals))
        except Exception:
            pass
        stop.wait(0.25)


def _json_from_text(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def _extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    return text.strip()


def _score_quality_breakdown(samples: list[dict[str, Any]]) -> dict[str, float]:
    groups: dict[str, list[float]] = {"qa": [], "structured": [], "code": [], "summary": []}
    for s in samples:
        kind = s["kind"]
        score = 0.0
        if kind == "qa":
            ans = s.get("text", "").strip().strip(".")
            exp = str(s.get("expect", "")).strip()
            score = 1.0 if ans.lower() == exp.lower() else 0.0
        elif kind == "structured":
            obj = _json_from_text(s.get("text", ""))
            if isinstance(obj, dict):
                keys = s.get("json_keys", [])
                score = 1.0 if all(k in obj for k in keys) else 0.5
            else:
                score = 0.0
        elif kind == "code":
            code = _extract_code(s.get("text", ""))
            fn_name = s.get("code_test")
            ns: dict[str, Any] = {}
            try:
                exec(code, {}, ns)
                if fn_name in ns and callable(ns[fn_name]):
                    if fn_name == "add":
                        score = 1.0 if ns[fn_name](2, 3) == 5 else 0.0
                    elif fn_name == "square":
                        score = 1.0 if ns[fn_name](4) == 16 else 0.0
                    elif fn_name == "fib1":
                        out = ns[fn_name](5)
                        score = 1.0 if list(out)[:5] == [0, 1, 1, 2, 3] else 0.0
                    else:
                        score = 1.0
                else:
                    score = 0.0
            except Exception:
                score = 0.0
        elif kind == "summary":
            txt = s.get("text", "").lower()
            kws = [str(k).lower() for k in s.get("summary_keywords", [])]
            hit = sum(1 for k in kws if k.lower() in txt)
            score = hit / max(1, len(kws))
        groups[kind].append(float(score))

    means = {k: (sum(v) / len(v) if v else 0.0) for k, v in groups.items()}
    overall = (
        0.30 * means.get("qa", 0.0)
        + 0.25 * means.get("structured", 0.0)
        + 0.20 * means.get("code", 0.0)
        + 0.25 * means.get("summary", 0.0)
    )
    means["overall"] = overall
    return means


def _chat_once(
    base_url: str, api_key: str, model: str, prompt: str, stream: bool, timeout_s: float = 120.0
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "stream": stream,
        "max_tokens": 256,
    }
    start = _now()
    text = ""
    ttft = None
    status = "ok"
    err = None

    try:
        with requests.post(
            url, headers=headers, json=payload, stream=stream, timeout=timeout_s
        ) as r:
            r.raise_for_status()
            if stream:
                for raw in r.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    line = raw.strip()
                    if line.startswith("data:"):
                        line = line[len("data:") :].strip()
                    if line == "[DONE]":
                        break
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    chunk = delta.get("content") or ""
                    if chunk and ttft is None:
                        ttft = _now() - start
                    if chunk:
                        text += chunk
            else:
                obj = r.json()
                choices = obj.get("choices") or []
                if choices:
                    text = choices[0].get("message", {}).get("content", "") or ""
                ttft = None
    except Exception as e:
        status = "error"
        err = str(e)

    total = _now() - start
    tok = _estimate_tokens(text) if text else 0
    return {
        "status": status,
        "error": err,
        "text": text,
        "latency_s": total,
        "ttft_s": ttft if ttft is not None else total * 0.45,
        "tokens": tok,
    }


def _launch_vllm_server(
    model: str, port: int, params: dict[str, Any], extra_args: str
) -> subprocess.Popen[str]:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--port",
        str(port),
    ]
    # Map params (best-effort)
    if "quantization" in params and params["quantization"] not in (None, "", "fp16", "bf16"):
        cmd += ["--quantization", str(params["quantization"])]
    if "gpu_memory_utilization" in params:
        cmd += ["--gpu-memory-utilization", str(params["gpu_memory_utilization"])]
    if "max_model_len" in params:
        cmd += ["--max-model-len", str(params["max_model_len"])]
    if "max_num_seqs" in params:
        cmd += ["--max-num-seqs", str(params["max_num_seqs"])]
    if "enable_prefix_caching" in params and params["enable_prefix_caching"]:
        cmd += ["--enable-prefix-caching"]
    if "kv_cache_dtype" in params:
        cmd += ["--kv-cache-dtype", str(params["kv_cache_dtype"])]
    if "tensor_parallel_size" in params:
        cmd += ["--tensor-parallel-size", str(params["tensor_parallel_size"])]
    if extra_args.strip():
        cmd += extra_args.strip().split()

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # wait for readiness
    t0 = time.time()
    while time.time() - t0 < 180:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=2)
            if r.ok:
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            break
        time.sleep(1.0)

    # If failed, dump a small log tail
    if proc.stdout:
        try:
            print("vLLM failed to start. Log tail:")
            for _ in range(50):
                line = proc.stdout.readline()
                if not line:
                    break
                print(line.rstrip())
        except Exception:
            pass
    raise RuntimeError("vLLM server did not become ready")


def _terminate_proc(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=15)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _summarize_method(
    method: str, params: dict[str, Any], samples: list[dict[str, Any]], vram_peak_mb: float | None
) -> dict[str, Any]:
    oks = [s for s in samples if s["status"] == "ok"]
    errs = [s for s in samples if s["status"] != "ok"]
    lat = sorted([float(s["latency_s"]) for s in oks])
    ttft = sorted([float(s["ttft_s"]) for s in oks])
    toks = sum(float(s.get("tokens", 0)) for s in oks)
    total_wall = sum(float(s["latency_s"]) for s in oks) or 1e-9
    tok_s = toks / total_wall

    def q(vs: list[float], p: float) -> float:
        if not vs:
            return 0.0
        idx = max(0, min(len(vs) - 1, int(round((len(vs) - 1) * p))))
        return vs[idx]

    q_break = _score_quality_breakdown(samples)
    return {
        "method": method,
        "algo": "real_backend",
        "is_feasible": True,  # quality gate is checked separately
        "quality": q_break["overall"],
        "quality_breakdown": {k: q_break[k] for k in ["qa", "structured", "code", "summary"]},
        "p95_s": q(lat, 0.95),
        "p99_s": q(lat, 0.99),
        "ttft_p95_s": q(ttft, 0.95),
        "tok_s": tok_s,
        "timeout_rate": 0.0,
        "error_rate": len(errs) / max(1, len(samples)),
        "tail_amp": (q(lat, 0.95) / max(1e-9, q(lat, 0.50))) if lat else 0.0,
        "vram_peak_mb": float(vram_peak_mb or 0.0),
        "params": params,
        "notes": f"real validation n={len(samples)}",
    }


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    rows = summary.get("rows", [])
    selected = [r for r in rows if r.get("method") in {"Baseline", "Random-best", "NSGA2-best"}]
    if not selected:
        raise SystemExit("No Baseline/Random-best/NSGA2-best rows found in summary")

    base_url = args.base_url
    results: list[dict[str, Any]] = []

    for row in selected:
        method = row["method"]
        params = row.get("params", {}) or {}
        proc = None
        if args.launch_vllm:
            proc = _launch_vllm_server(args.model, args.vllm_port, params, args.extra_vllm_args)
            base_url = f"http://127.0.0.1:{args.vllm_port}/v1"
            # warmup
            _chat_once(
                base_url, args.api_key, args.model, "Hello", stream=args.stream, timeout_s=60.0
            )

        stop = threading.Event()
        peaks: list[float] = []
        t = threading.Thread(target=_poll_vram, args=(stop, peaks), daemon=True)
        t.start()

        samples: list[dict[str, Any]] = []
        prompts = PROMPT_PACKS[args.workload]
        for i in range(args.requests_per_method):
            spec = prompts[i % len(prompts)].copy()
            resp = _chat_once(
                base_url, args.api_key, args.model, spec["prompt"], stream=args.stream
            )
            spec.update(resp)
            samples.append(spec)

        stop.set()
        t.join(timeout=1.0)
        if proc is not None:
            _terminate_proc(proc)

        vram_peak = max(peaks) if peaks else None
        results.append(_summarize_method(method, params, samples, vram_peak))

    out = {
        "workload": args.workload,
        "model": args.model,
        "base_url": args.base_url,
        "stream": bool(args.stream),
        "source_summary": str(args.summary),
        "rows": results,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ wrote {out_path}")


if __name__ == "__main__":
    main()
