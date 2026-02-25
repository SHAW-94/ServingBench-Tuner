from __future__ import annotations

import random
import uuid
from dataclasses import dataclass

from .replay import ArrivalEvent
from .schema import SessionSpec


@dataclass
class SessionExpandedPlan:
    """
    Output of expanding events into multi-turn sessions.

    For synthetic arrivals (poisson/burst):
      - each base event can become a session with N turns
      - turns are emitted at increasing timestamps (think time)
      - prompt length can grow by including previous outputs (context growth simulation)

    For replay arrivals:
      - if session_id/turn_id are present, we keep them and do not re-expand.
    """

    events: list[ArrivalEvent]


def _think_time(rng: random.Random, mu: float, sigma: float) -> float:
    # lognormal for positive times, centered around mu-ish
    # When sigma=0, lognormalvariate returns exp(mu).
    if mu <= 0:
        return 0.0
    x = rng.lognormvariate(math_log(mu + 1e-9), sigma)  # approximate
    return float(max(0.0, x))


def math_log(x: float) -> float:
    import math

    return math.log(max(1e-12, x))


def expand_sessions(
    base_events: list[ArrivalEvent],
    session_spec: SessionSpec,
    rng: random.Random,
) -> SessionExpandedPlan:
    """
    Expand base events into multi-turn sessions if enabled and if base events don't already
    contain session_id/turn_id information.

    Context growth simulation:
    - append: prompt grows by accumulating previous outputs
    - window: prompt grows but capped by max_context_tokens (simple truncation)
    - agent_like: step pattern with higher variance (plan->tool->summarize feel)
    """
    if not session_spec.enabled or session_spec.turns <= 1:
        return SessionExpandedPlan(events=base_events)

    # If replay already has sessions/turns, keep as-is.
    if any(e.session_id is not None or e.turn_id is not None for e in base_events):
        return SessionExpandedPlan(events=base_events)

    out: list[ArrivalEvent] = []

    for ev in base_events:
        sid = f"sess_{uuid.uuid4().hex[:8]}"
        t = ev.timestamp_s

        # seed per-session context length
        ctx_tokens = 0
        last_outputs: list[int] = []

        for turn in range(1, session_spec.turns + 1):
            # Simulate "think time" between turns (skip before first)
            if turn > 1:
                # lognormal-ish think time
                import math

                # use lognormal directly with parameters derived from mu/sigma
                # (keep deterministic enough)
                dt = rng.lognormvariate(
                    math.log(max(1e-6, session_spec.think_time_s_mu)),
                    session_spec.think_time_s_sigma,
                )
                t += float(max(0.0, dt))

            # Determine prompt growth
            base_in = ev.input_tokens if ev.input_tokens is not None else 0
            base_out = ev.output_tokens if ev.output_tokens is not None else 0

            if session_spec.context_growth == "append":
                ctx_tokens = base_in + sum(last_outputs)
            elif session_spec.context_growth == "window":
                ctx_tokens = min(session_spec.max_context_tokens, base_in + sum(last_outputs))
            elif session_spec.context_growth == "agent_like":
                # agent-like: plan->tool->summary; higher variance and sometimes big jumps
                # turn roles:
                # 1 plan (short output), 2 tool-call (short output), 3 reasoning/summary (longer)
                role = (turn - 1) % 3
                extra_ctx = 0
                if role == 0:
                    extra_ctx = rng.randint(50, 200)  # planning adds some context
                elif role == 1:
                    extra_ctx = rng.randint(150, 600)  # tool result injected
                else:
                    extra_ctx = rng.randint(100, 400)  # summarize/reason
                ctx_tokens = base_in + sum(last_outputs) + extra_ctx
                ctx_tokens = min(session_spec.max_context_tokens, ctx_tokens)
            else:
                ctx_tokens = base_in

            # Emit a new event (turn)
            out_ev = ArrivalEvent(
                timestamp_s=float(t),
                request_id=f"{ev.request_id}_t{turn}",
                session_id=sid,
                turn_id=turn,
                category="session",
                input_tokens=int(ctx_tokens) if ctx_tokens else ev.input_tokens,
                output_tokens=ev.output_tokens,
                raw=ev.raw,
            )
            out.append(out_ev)

            # Update outputs history for next turn
            if base_out > 0:
                # agent-like increases variance in outputs: tool may be short, summary longer
                if session_spec.context_growth == "agent_like":
                    role = (turn - 1) % 3
                    if role == 1:
                        out_tok = max(10, int(base_out * rng.uniform(0.2, 0.6)))
                    elif role == 2:
                        out_tok = max(30, int(base_out * rng.uniform(0.8, 1.6)))
                    else:
                        out_tok = max(20, int(base_out * rng.uniform(0.3, 1.0)))
                else:
                    out_tok = base_out
            else:
                out_tok = rng.randint(20, 200)

            last_outputs.append(out_tok)

    out.sort(key=lambda e: (e.timestamp_s, e.request_id))
    return SessionExpandedPlan(events=out)
