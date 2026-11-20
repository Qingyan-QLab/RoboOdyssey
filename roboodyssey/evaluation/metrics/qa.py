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
