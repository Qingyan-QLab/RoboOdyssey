from __future__ import annotations

import re
from typing import Any


def normalize_answer(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\u4e00-\u9fff ]", "", str(value).casefold())).strip()


def contextual_answer_correct(value: Any, expected: dict[str, Any]) -> bool:
    """Match a v5 public answer without using action or inspection evidence."""
    if value is None:
        return False
    if expected['answer_type'] == 'boolean':
        tokens = set(re.findall(r'\b(?:yes|no|true|false)\b', str(value).lower()))
        positive = bool(tokens & {'yes', 'true'})
        negative = bool(tokens & {'no', 'false'})
        return (positive != negative) and positive is bool(expected['answer'])
    normalized = normalize_answer(value)
    return any(normalized in {a, 'the '+a, 'in '+a, 'in the '+a, 'it is in the '+a}
               for a in map(normalize_answer, expected['accepted_answers']))


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
    normalized_prediction = normalize_answer(prediction)
    references = [normalize_answer(item) for item in candidates]
    # Deterministic semantic containment handles concise references such as
    # "bathroom" versus a complete natural-language answer. Word boundaries
    # prevent "book" from matching "bookshelf".
    semantic = exact or any(
        bool(reference) and re.search(rf"(?<!\w){re.escape(reference)}(?!\w)",
                                      normalized_prediction) is not None
        for reference in references
    )
    return {"exact_match": exact, "semantic_match": semantic,
            "normalized_prediction": normalized_prediction,
            "normalized_references": references}
