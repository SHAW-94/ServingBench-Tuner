#!/usr/bin/env python3
"""
make_public_workload.py

Generate a PUBLIC (synthetic) workload trace and empirical length distributions
(no enterprise/private data).

Outputs (in --out-dir):
- <profile>_trace.jsonl               replay trace
- <profile>_len_prompt.json           empirical prompt token lengths
- <profile>_len_output.json           empirical output token lengths
- <profile>_profile.md                short rationale of workload profile

Trace schema (jsonl):
{
  "timestamp_s": float,               seconds since start
  "request_id": str,
  "session_id": str | null,
  "turn_id": int | null,
  "category": str,                    e.g. short_qa/long_gen/agent_like
  "input_tokens": int,
  "output_tokens": int
}

This is intentionally simple: it gives you replay mode + realistic long-tail lengths.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Req:
    t: float
    request_id: str
    session_id: str | None
    turn_id: int | None
    category: str
    input_tokens: int
    output_tokens: int


def _poisson_arrivals(rps: float, n: int, rng: random.Random) -> list[float]:
    """Generate n arrival times (seconds since start) with exponential inter-arrival."""
    t = 0.0
    times = []
    for _ in range(n):
        u = rng.random()
        dt = -math.log(1.0 - u) / max(rps, 1e-9)
        t += dt
        times.append(t)
    return times


def _lognormal_tokens(mu: float, sigma: float, low: int, high: int, rng: random.Random) -> int:
    v = rng.lognormvariate(mu, sigma)
    x = int(max(low, min(high, round(v))))
    return x


def _make_profile_requests(
    profile: str, n: int, seed: int
) -> tuple[list[Req], list[int], list[int]]:
    rng = random.Random(seed)

    # Default arrival: poisson-like
    if profile in ("short_qa", "quickstart"):
        rps = 2.0
    elif profile == "long_gen":
        rps = 0.7
    else:  # agent_like
        rps = 1.0

    arrivals = _poisson_arrivals(rps=rps, n=n, rng=rng)

    reqs: list[Req] = []
    prompt_lens: list[int] = []
    output_lens: list[int] = []

    if profile in ("short_qa", "quickstart"):
        # short prompts, short outputs
        for i, t in enumerate(arrivals):
            in_tok = _lognormal_tokens(mu=4.3, sigma=0.35, low=10, high=300, rng=rng)
            out_tok = _lognormal_tokens(mu=3.8, sigma=0.40, low=5, high=200, rng=rng)
            rid = f"req_{i:06d}_{uuid.uuid4().hex[:8]}"
            reqs.append(
                Req(
                    t=t,
                    request_id=rid,
                    session_id=None,
                    turn_id=None,
                    category=profile,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                )
            )
            prompt_lens.append(in_tok)
            output_lens.append(out_tok)

    elif profile == "long_gen":
        # shorter prompt, long output long-tail
        for i, t in enumerate(arrivals):
            in_tok = _lognormal_tokens(mu=5.0, sigma=0.45, low=50, high=1200, rng=rng)
            out_tok = _lognormal_tokens(mu=6.2, sigma=0.55, low=200, high=4096, rng=rng)
            rid = f"req_{i:06d}_{uuid.uuid4().hex[:8]}"
            reqs.append(
                Req(
                    t=t,
                    request_id=rid,
                    session_id=None,
                    turn_id=None,
                    category=profile,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                )
            )
            prompt_lens.append(in_tok)
            output_lens.append(out_tok)

    elif profile == "agent_like":
        # multi-turn sessions; context grows; output variability high
        # We'll create n requests as turns across sessions.
        session_count = max(1, n // 3)
        sessions = [f"sess_{uuid.uuid4().hex[:8]}" for _ in range(session_count)]
        for i, t in enumerate(arrivals):
            sid = sessions[rng.randrange(session_count)]
            turn_id = (i % 5) + 1  # up to 5 turns
            # context growth: later turns get longer prompt
            base_in = _lognormal_tokens(mu=5.1, sigma=0.5, low=80, high=2000, rng=rng)
            in_tok = min(4096, base_in + (turn_id - 1) * rng.randint(80, 220))
            # output length high variance
            out_tok = _lognormal_tokens(mu=5.6, sigma=0.8, low=30, high=3000, rng=rng)

            rid = f"req_{i:06d}_{uuid.uuid4().hex[:8]}"
            reqs.append(
                Req(
                    t=t,
                    request_id=rid,
                    session_id=sid,
                    turn_id=turn_id,
                    category=profile,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                )
            )
            prompt_lens.append(in_tok)
            output_lens.append(out_tok)

    else:
        raise ValueError(f"Unknown profile: {profile}")

    return reqs, prompt_lens, output_lens


def _write_jsonl(path: Path, reqs: Iterable[Req]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in reqs:
            f.write(
                json.dumps(
                    {
                        "timestamp_s": round(r.t, 6),
                        "request_id": r.request_id,
                        "session_id": r.session_id,
                        "turn_id": r.turn_id,
                        "category": r.category,
                        "input_tokens": r.input_tokens,
                        "output_tokens": r.output_tokens,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _write_empirical(path: Path, values: list[int]) -> None:
    # Store raw list + summary stats for reproducibility/debug
    values_sorted = sorted(values)
    n = len(values_sorted)

    def pct(p: float) -> int:
        idx = int(round((p / 100.0) * (n - 1)))
        return values_sorted[max(0, min(n - 1, idx))]

    payload = {
        "type": "empirical",
        "n": n,
        "min": values_sorted[0] if n else None,
        "p50": pct(50) if n else None,
        "p95": pct(95) if n else None,
        "p99": pct(99) if n else None,
        "max": values_sorted[-1] if n else None,
        "values": values_sorted,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_profile_md(path: Path, profile: str, n: int, seed: int) -> None:
    rationale = {
        "quickstart": "Minimal replay workload for CPU-only pipeline smoke tests.",
        "short_qa": "Online short Q&A profile: short prompts, short outputs, Poisson arrivals.",
        "long_gen": "Long generation profile: moderate prompts, long outputs with long-tail.",
        "agent_like": "Agent-like multi-turn profile: sessions with context growth and high output variance.",
    }.get(profile, "Synthetic public workload profile.")

    md = f"""# {profile}

- Requests: {n}
- Seed: {seed}

## Rationale
{rationale}

## Notes
This dataset is synthetic/public (no private logs). It is designed to:
- exercise replay mode
- include realistic long-tail length distributions
- (agent_like) simulate multi-turn context growth

"""
    path.write_text(md, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir", type=str, required=True, help="output directory, e.g. data/workloads"
    )
    ap.add_argument(
        "--profiles",
        type=str,
        default="short_qa,long_gen,agent_like",
        help="comma-separated profiles: short_qa,long_gen,agent_like,quickstart",
    )
    ap.add_argument("--n-requests", type=int, default=300, help="requests per profile")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    for p in profiles:
        reqs, pl, ol = _make_profile_requests(p, args.n_requests, seed=args.seed)
        trace_path = out / f"{p}_trace.jsonl"
        prompt_path = out / f"{p}_len_prompt.json"
        output_path = out / f"{p}_len_output.json"
        md_path = out / f"{p}_profile.md"

        _write_jsonl(trace_path, reqs)
        _write_empirical(prompt_path, pl)
        _write_empirical(output_path, ol)
        _write_profile_md(md_path, p, args.n_requests, args.seed)

        print(
            f"[OK] {p}: wrote {trace_path.name}, {prompt_path.name}, {output_path.name}, {md_path.name}"
        )


if __name__ == "__main__":
    main()
