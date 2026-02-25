from __future__ import annotations

import json


def test_parse_trace_rows_sorts_and_fills_defaults() -> None:
    from servingbench_tuner.workload.replay import parse_trace_rows

    rows = [
        {"t": 2.0, "id": "b", "output_tokens": 4},
        {"timestamp_s": 1.0, "request_id": "a", "category": "chat", "input_tokens": 3},
        {"timestamp_s": 1.0, "request_id": "aa"},
    ]

    events = parse_trace_rows(rows)

    assert [e.request_id for e in events] == ["a", "aa", "b"]
    assert events[0].category == "chat"
    assert events[1].category == "replay"
    assert events[2].output_tokens == 4
    assert events[0].raw is not None and events[0].raw["request_id"] == "a"


def test_read_jsonl_and_load_replay_events(tmp_path) -> None:
    from servingbench_tuner.workload.replay import load_replay_events, read_jsonl_trace

    p = tmp_path / "trace.jsonl"
    rows = [
        {"timestamp_s": 0.0, "request_id": "x", "input_tokens": 1, "output_tokens": 2},
        {"timestamp_s": 0.2, "request_id": "y", "input_tokens": 3, "output_tokens": 4},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    raw_rows = read_jsonl_trace(p)
    assert len(raw_rows) == 2
    assert raw_rows[0]["request_id"] == "x"

    events = load_replay_events(p)
    assert [e.request_id for e in events] == ["x", "y"]
    assert events[1].input_tokens == 3
