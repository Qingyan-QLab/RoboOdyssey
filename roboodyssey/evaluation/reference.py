"""Private scoring references used by the evaluation runtime."""
from pathlib import Path
from typing import Any


class ScoringReference:
    """Expose current scoring references without implying an action policy."""

    def prepare(self, environment: Any, ground_truth: dict[str, Any], output: str | Path):
        from .task_metrics import prepare_reference
        return prepare_reference(environment, ground_truth, Path(output))
