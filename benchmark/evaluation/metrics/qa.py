from __future__ import annotations

import re
from typing import Any


def normalize_answer(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\u4e00-\u9fff ]", "", str(value).casefold())).strip()


def score_qa(prediction: Any, ground_truth: Any, accepted: list[Any] | None = None) -> dict[str, Any]:
    candidates = [ground_truth] + list(accepted or [])
    # Preserve structure for generated integer/list answers while still
    # allowing natural-language free-form answers.
    def equivalent(a: Any, b: Any) -> bool:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a == b
        if isinstance(a, list) and isinstance(b, list):
            return {normalize_answer(x) for x in a} == {normalize_answer(x) for x in b}
        return normalize_answer(a) == normalize_answer(b)
    exact = any(equivalent(prediction, item) for item in candidates)
    return {"exact_match": exact, "normalized_prediction": normalize_answer(prediction), "normalized_references": [normalize_answer(item) for item in candidates]}
