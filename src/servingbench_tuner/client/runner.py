from __future__ import annotations

import random
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import anyio

from servingbench_tuner.backends.base import BackendAdapter, GenerationRequest, TokenEvent
from servingbench_tuner.client.pacing import Pacer, PacingConfig
from servingbench_tuner.client.tracing import RequestTrace, SpanBreakdown, TokenTiming, now_s
from servingbench_tuner.workload.replay import ArrivalEvent


@dataclass
class RunnerConfig:
    """
    Load runner config (client-side).
    """

    concurrency_limit: int = 64
    timeout_s: float = 30.0
    retries: int = 0
    warmup_requests: int = 0

    # retry behavior
    retry_backoff_base_s: float = 0.20
    retry_backoff_max_s: float = 2.00

    # pacing
    pacing: PacingConfig = field(default_factory=PacingConfig)

    # recording
    record_token_timings: bool = True


@dataclass
class RunResult:
    """
    Results of a load run.
    traces: includes both warmup + measured traces. Use split_warmup() helper.
    """

    traces: list[RequestTrace]
    run_start_wall_s: float
    run_end_wall_s: float

    def split_warmup(self, warmup_n: int) -> tuple[list[RequestTrace], list[RequestTrace]]:
        """
        Return (warmup_traces, measured_traces) by arrival order.
        """
        ordered = sorted(self.traces, key=lambda t: t.arrival_s)
        return ordered[:warmup_n], ordered[warmup_n:]


def _compute_backoff(rng: random.Random, base: float, cap: float, attempt: int) -> float:
    # exponential backoff with jitter
    raw = base * (2 ** max(0, attempt - 1))
    raw = min(cap, raw)
    return raw * (0.8 + 0.4 * rng.random())


class LoadRunner:
    """
    Executes a workload against a BackendAdapter:
    - respects arrival timestamps (pacing)
    - enforces client-side concurrency
    - applies timeout and retry
    - records per-request traces with TTFT/TPOT and span breakdown

    Runner is backend-agnostic: it can run against mock/local_cpu/openai_http/vllm_openai adapters.
    """

    def __init__(
        self,
        backend: BackendAdapter,
        cfg: RunnerConfig,
        request_builder: Callable[[ArrivalEvent], GenerationRequest],
    ) -> None:
        self.backend = backend
        self.cfg = cfg
        self.request_builder = request_builder
        self._sem = anyio.Semaphore(cfg.concurrency_limit)
        self._pacer = Pacer(cfg.pacing)
        self._rng = random.Random(42)

    async def run(self, events: list[ArrivalEvent]) -> RunResult:
        """
        Run load test and return traces (including warmup traces).
        """
        run_start_wall = now_s()
        run_start_clock = self.cfg.pacing.now()

        traces: list[RequestTrace] = []

        async def _one_event(ev: ArrivalEvent) -> None:
            # arrival is when pacing yields the event (relative to run start)
            arrival_rel = (self.cfg.pacing.now() - run_start_clock) * self.cfg.pacing.time_scale
            trace = await self._execute_with_retries(
                ev, arrival_rel, run_start_wall, run_start_clock
            )
            traces.append(trace)

        async with anyio.create_task_group() as tg:
            async for ev in self._pacer.iter_events(events):
                tg.start_soon(_one_event, ev)

        run_end_wall = now_s()
        return RunResult(
            traces=traces, run_start_wall_s=run_start_wall, run_end_wall_s=run_end_wall
        )

    async def _execute_with_retries(
        self,
        ev: ArrivalEvent,
        arrival_rel_s: float,
        run_start_wall: float,
        run_start_clock: float,
    ) -> RequestTrace:
        # last_error = ""
        retry_count = 0

        for attempt in range(1, self.cfg.retries + 2):  # attempt=1..retries+1
            if attempt > 1:
                retry_count += 1
                backoff = _compute_backoff(
                    self._rng, self.cfg.retry_backoff_base_s, self.cfg.retry_backoff_max_s, attempt
                )
                await anyio.sleep(backoff)

            trace = await self._execute_once(ev, arrival_rel_s, run_start_wall, retry_count)
            if trace.status == "ok":
                return trace
            # last_error = trace.error

            # if timeout/error and still have retries, continue
            if attempt <= self.cfg.retries + 1:
                continue

        # should not reach, but fallback
        return trace  # type: ignore[return-value]

    async def _execute_once(
        self,
        ev: ArrivalEvent,
        arrival_rel_s: float,
        run_start_wall: float,
        retry_count: int,
    ) -> RequestTrace:
        """
        One request execution attempt.
        - client_queue_s measures waiting for semaphore
        - send_start_s is when request actually sent (after acquiring slot)
        - first_token_s measured on first streamed delta
        - end_s when stream completes
        """
        # Build request payload
        req = self.request_builder(ev)
        # supply token hints if absent
        if req.input_tokens is None and ev.input_tokens is not None:
            req.input_tokens = ev.input_tokens
        if req.output_tokens is None and ev.output_tokens is not None:
            req.output_tokens = ev.output_tokens

        # Client concurrency gating
        t0 = now_s()
        async with self._sem:
            t_send = now_s()
            client_queue = t_send - t0

            # Time bases: store relative to run_start_wall
            send_start_rel = t_send - run_start_wall

            first_token_rel = send_start_rel  # init; if no token, will become end
            end_rel = send_start_rel
            output_tokens = 0
            input_tokens = int(req.input_tokens or ev.input_tokens or 0)

            status: str = "ok"
            error = ""
            token_timings: list[TokenTiming] = []

            # Span breakdown: fill what we can
            spans = SpanBreakdown(client_queue_s=float(max(0.0, client_queue)))

            # Streaming loop with timeout
            try:
                with anyio.fail_after(req.timeout_s if req.timeout_s else self.cfg.timeout_s):
                    token_index = 0
                    first_seen = False
                    server_meta: dict[str, Any] = {}

                    async for tok in self.backend.generate_stream(req):
                        assert isinstance(tok, TokenEvent)
                        server_meta = tok.meta or server_meta

                        if tok.text:
                            t_now = now_s()
                            if not first_seen:
                                first_seen = True
                                first_token_rel = t_now - run_start_wall
                                # prefill approx from send_start to first token
                                spans.client_prefill_s = float(
                                    max(0.0, (first_token_rel - send_start_rel))
                                )

                            if self.cfg.record_token_timings:
                                token_timings.append(
                                    TokenTiming(index=token_index, t_s=t_now - run_start_wall)
                                )
                            token_index += 1

                        # usage might appear at end; prefer it for output token count
                        if tok.usage and isinstance(tok.usage, dict):
                            # OpenAI uses completion_tokens in usage (sometimes nested)
                            ct = tok.usage.get("completion_tokens", None)
                            if isinstance(ct, int):
                                output_tokens = max(output_tokens, ct)

                        if tok.is_final:
                            break

                    # end time
                    end_rel = now_s() - run_start_wall
                    if not first_seen:
                        # No token observed; treat TTFT as end-start
                        first_token_rel = end_rel
                        spans.client_prefill_s = float(max(0.0, (first_token_rel - send_start_rel)))

                    # decode approx = end - first_token
                    spans.client_decode_s = float(max(0.0, end_rel - first_token_rel))

                    # If backend provides server-side spans (mock backend does)
                    # Accept either direct numbers or via meta keys.
                    for k, attr in [
                        ("queue_time_s", "server_queue_s"),
                        ("prefill_time_s", "server_prefill_s"),
                        ("decode_time_s", "server_decode_s"),
                    ]:
                        if k in server_meta:
                            with suppress(Exception):
                                setattr(spans, attr, float(server_meta[k]))

                    # If output_tokens unknown, fallback to token event count
                    if output_tokens <= 0:
                        output_tokens = (
                            max(1, len(token_timings))
                            if token_timings
                            else int(req.output_tokens or 0)
                        )

            except TimeoutError:
                status = "timeout"
                error = "timeout"
                end_rel = now_s() - run_start_wall
                if first_token_rel == send_start_rel:
                    first_token_rel = end_rel
                spans.client_prefill_s = float(max(0.0, first_token_rel - send_start_rel))
                spans.client_decode_s = float(max(0.0, end_rel - first_token_rel))

            except Exception as e:
                status = "error"
                error = str(e)
                end_rel = now_s() - run_start_wall
                if first_token_rel == send_start_rel:
                    first_token_rel = end_rel
                spans.client_prefill_s = float(max(0.0, first_token_rel - send_start_rel))
                spans.client_decode_s = float(max(0.0, end_rel - first_token_rel))

        return RequestTrace(
            request_id=ev.request_id,
            arrival_s=float(arrival_rel_s),
            send_start_s=float(send_start_rel),
            first_token_s=float(first_token_rel),
            end_s=float(end_rel),
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            status=status,  # type: ignore[arg-type]
            error=error,
            retry_count=int(retry_count),
            session_id=ev.session_id,
            turn_id=ev.turn_id,
            spans=spans,
            token_timings=token_timings,
            meta={"category": ev.category},
        )


def default_request_builder(ev: ArrivalEvent) -> GenerationRequest:
    """
    Default request builder for synthetic/public workloads.
    You can replace it with a builder that uses actual prompt templates or replayed messages.

    Here we only need something deterministic and cheap:
    - prompt text length isn't used by real servers for token count, but is fine as placeholder.
    """
    prompt = f"[{ev.category}] request_id={ev.request_id} input_tokens={ev.input_tokens} output_tokens={ev.output_tokens}"
    return GenerationRequest(
        messages=[{"role": "user", "content": prompt}],
        max_new_tokens=int(ev.output_tokens or 256),
        temperature=0.0,
        top_p=1.0,
        stream=True,
        timeout_s=30.0,
        input_tokens=ev.input_tokens,
        output_tokens=ev.output_tokens,
    )
