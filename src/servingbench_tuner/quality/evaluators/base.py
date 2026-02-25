from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from servingbench_tuner.quality.dataset import EvalExample


@dataclass
class EvalResult:
    """
    Result for one example.

    score: normalized [0, 1]
    passed: boolean pass/fail used for gate
    details: evaluator-specific debug info (stored to artifacts)
    """

    example_id: str
    example_type: str
    score: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


class Evaluator(ABC):
    """
    Base evaluator interface.
    """

    name: str = "base"
    supported_types: Sequence[str] = ()

    def supports(self, example: EvalExample) -> bool:
        return (not self.supported_types) or (example.type in self.supported_types)

    @abstractmethod
    async def evaluate(self, example: EvalExample, output_text: str) -> EvalResult:
        """
        Evaluate model output for a single example.
        """
        ...


class EvaluatorRegistry:
    """
    Route examples to evaluators by example.type.
    """

    def __init__(self, evaluators: list[Evaluator]) -> None:
        self.evaluators = evaluators
        self._by_type: dict[str, Evaluator] = {}
        for ev in evaluators:
            for t in ev.supported_types:
                # last one wins (allow override)
                self._by_type[t] = ev

    def get(self, example_type: str) -> Evaluator | None:
        return self._by_type.get(example_type)

    def route(self, example: EvalExample) -> Evaluator:
        ev = self.get(example.type)
        if ev is None:
            raise KeyError(f"No evaluator registered for type={example.type}")
        return ev
