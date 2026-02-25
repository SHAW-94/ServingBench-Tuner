from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


@dataclass
class ExperimentMeta:
    """
    One logical experiment: e.g. "vllm_qwen7b_shortqa_nsga2_2026-02-19".
    """

    experiment_id: str
    ts: float
    model: str
    workload_sig: str
    notes: dict[str, Any]


@dataclass
class RunRecord:
    """
    One run = one repeated measurement under a fixed config.
    A run can include multiple repeats, aggregated into summary metrics.
    """

    experiment_id: str
    run_id: str
    ts: float
    config_json: dict[str, Any]

    # aggregated summaries
    e2e_json: dict[str, Any]
    server_json: dict[str, Any] | None
    quality_json: dict[str, Any] | None
    gate_json: dict[str, Any] | None

    # additional info
    feasible: bool | None = None
    violations_json: dict[str, Any] | None = None

    artifacts_dir: str = ""


class ResultStore:
    """
    SQLite + artifacts directory.

    Tables:
      - experiments
      - runs

    You can query later for regression comparison, reporting, etc.
    """

    def __init__(
        self, sqlite_path: str = "results/runs.sqlite", artifacts_root: str = "results/artifacts"
    ) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.artifacts_root = Path(artifacts_root)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.sqlite_path))
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
              experiment_id TEXT PRIMARY KEY,
              ts REAL,
              model TEXT,
              workload_sig TEXT,
              notes TEXT
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
              experiment_id TEXT,
              run_id TEXT,
              ts REAL,
              config_json TEXT,
              e2e_json TEXT,
              server_json TEXT,
              quality_json TEXT,
              gate_json TEXT,
              feasible INTEGER,
              violations_json TEXT,
              artifacts_dir TEXT,
              PRIMARY KEY (experiment_id, run_id)
            );
            """
        )
        conn.commit()
        conn.close()

    def ensure_experiment(
        self,
        experiment_id: str,
        model: str,
        workload_sig: str,
        notes: dict[str, Any] | None = None,
    ) -> ExperimentMeta:
        ts = time.time()
        meta = ExperimentMeta(
            experiment_id=experiment_id,
            ts=ts,
            model=model,
            workload_sig=workload_sig,
            notes=notes or {},
        )
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO experiments(experiment_id, ts, model, workload_sig, notes) VALUES (?,?,?,?,?)",
            (meta.experiment_id, meta.ts, meta.model, meta.workload_sig, _json(meta.notes)),
        )
        conn.commit()
        conn.close()
        return meta

    def make_run_artifacts_dir(self, experiment_id: str, run_id: str) -> Path:
        p = self.artifacts_root / experiment_id / run_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def write_json_artifact(self, path: Path, obj: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    def write_text_artifact(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def record_run(self, rec: RunRecord) -> None:
        conn = self._connect()
        conn.execute(
            """
            INSERT OR REPLACE INTO runs(
              experiment_id, run_id, ts,
              config_json, e2e_json, server_json, quality_json, gate_json,
              feasible, violations_json, artifacts_dir
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                rec.experiment_id,
                rec.run_id,
                rec.ts,
                _json(rec.config_json),
                _json(rec.e2e_json),
                _json(rec.server_json) if rec.server_json is not None else None,
                _json(rec.quality_json) if rec.quality_json is not None else None,
                _json(rec.gate_json) if rec.gate_json is not None else None,
                (1 if rec.feasible else 0) if rec.feasible is not None else None,
                _json(rec.violations_json) if rec.violations_json is not None else None,
                rec.artifacts_dir,
            ),
        )
        conn.commit()
        conn.close()

    def list_runs(self, experiment_id: str) -> list[dict[str, Any]]:
        conn = self._connect()
        cur = conn.execute(
            """
            SELECT run_id, ts, config_json, e2e_json, server_json, quality_json, gate_json, feasible, violations_json, artifacts_dir
            FROM runs WHERE experiment_id=? ORDER BY ts ASC
            """,
            (experiment_id,),
        )
        rows = cur.fetchall()
        conn.close()

        out: list[dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "run_id": r[0],
                    "ts": r[1],
                    "config": json.loads(r[2]) if r[2] else {},
                    "e2e": json.loads(r[3]) if r[3] else {},
                    "server": json.loads(r[4]) if r[4] else None,
                    "quality": json.loads(r[5]) if r[5] else None,
                    "gate": json.loads(r[6]) if r[6] else None,
                    "feasible": (bool(r[7]) if r[7] is not None else None),
                    "violations": (json.loads(r[8]) if r[8] else None),
                    "artifacts_dir": r[9] or "",
                }
            )
        return out

    def get_run(self, experiment_id: str, run_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        cur = conn.execute(
            """
            SELECT run_id, ts, config_json, e2e_json, server_json, quality_json, gate_json, feasible, violations_json, artifacts_dir
            FROM runs WHERE experiment_id=? AND run_id=?
            """,
            (experiment_id, run_id),
        )
        row = cur.fetchone()
        conn.close()
        if row is None:
            return None
        return {
            "run_id": row[0],
            "ts": row[1],
            "config": json.loads(row[2]) if row[2] else {},
            "e2e": json.loads(row[3]) if row[3] else {},
            "server": json.loads(row[4]) if row[4] else None,
            "quality": json.loads(row[5]) if row[5] else None,
            "gate": json.loads(row[6]) if row[6] else None,
            "feasible": (bool(row[7]) if row[7] is not None else None),
            "violations": (json.loads(row[8]) if row[8] else None),
            "artifacts_dir": row[9] or "",
        }

    def latest_run(self, experiment_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        cur = conn.execute(
            """
            SELECT run_id FROM runs WHERE experiment_id=? ORDER BY ts DESC LIMIT 1
            """,
            (experiment_id,),
        )
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        return self.get_run(experiment_id, row[0])
