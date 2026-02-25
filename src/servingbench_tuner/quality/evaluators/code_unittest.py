from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from servingbench_tuner.quality.dataset import EvalExample

from .base import EvalResult, Evaluator


@dataclass
class CodeUnitTestEvaluator(Evaluator):
    """
    Evaluate generated code by running pytest in an isolated temp directory.

    Example JSONL:
      {
        "id":"c1",
        "type":"code_unittest",
        "prompt":"Write a function add(a,b)",
        "filename":"solution.py",
        "tests":"from solution import add\\n\\ndef test_add():\\n    assert add(2,3)==5\\n"
      }

    SECURITY NOTE:
      Running arbitrary code is inherently risky. This evaluator is intended for:
      - your own curated eval_pack
      - CI in a controlled environment
    It uses a temp directory + timeout + minimal env, but it is not a perfect sandbox.
    """

    name: str = "code_unittest"
    supported_types: Sequence[str] = ("code_unittest",)

    timeout_s: float = 10.0
    python_exe: str = sys.executable

    async def evaluate(self, example: EvalExample, output_text: str) -> EvalResult:
        filename = str(example.get("filename", "solution.py"))
        tests = example.get("tests", None)
        if not isinstance(tests, str) or not tests.strip():
            return EvalResult(
                example.id, example.type, 0.0, False, {"error": "missing tests string"}
            )

        code = output_text or ""
        if not code.strip():
            return EvalResult(example.id, example.type, 0.0, False, {"error": "empty code output"})

        # Run in temp directory
        with tempfile.TemporaryDirectory(prefix="sbt_code_eval_") as td:
            td_path = Path(td)
            sol_path = td_path / filename
            sol_path.write_text(code, encoding="utf-8")

            test_path = td_path / "test_solution.py"
            test_path.write_text(tests, encoding="utf-8")

            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["PYTHONNOUSERSITE"] = "1"
            # reduce accidental network proxy usage
            env.pop("HTTP_PROXY", None)
            env.pop("HTTPS_PROXY", None)
            env.pop("ALL_PROXY", None)

            cmd = [self.python_exe, "-m", "pytest", "-q", str(test_path.name)]
            try:
                p = subprocess.run(
                    cmd,
                    cwd=str(td_path),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=float(self.timeout_s),
                    check=False,
                )
                ok = p.returncode == 0
                score = 1.0 if ok else 0.0
                return EvalResult(
                    example_id=example.id,
                    example_type=example.type,
                    score=score,
                    passed=ok,
                    details={
                        "returncode": p.returncode,
                        "output": (p.stdout[-4000:] if p.stdout else ""),
                        "cmd": cmd,
                        "filename": filename,
                    },
                )
            except subprocess.TimeoutExpired:
                return EvalResult(example.id, example.type, 0.0, False, {"error": "pytest timeout"})
            except Exception as e:
                return EvalResult(example.id, example.type, 0.0, False, {"error": str(e)})
