"""Compose session state from immutable scene metadata and ordered deltas."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable


def apply_changes(state: dict[str, Any], changes: Iterable[dict[str, Any]]) -> dict[str, Any]:
    result = deepcopy(state)
    objects = result.setdefault("objects", {})
    for change in changes:
        object_id = str(change.get("object_id", ""))
        if object_id not in objects:
            raise KeyError(f"state change references unknown object: {object_id}")
        after = deepcopy(change.get("after") or {})
        if not after:
            if change.get("pose_after") is not None:
                after["pose"] = deepcopy(change["pose_after"])
            relation = change.get("relation_after") or {}
            for key in ("room_id", "support_object_id", "relation_type"):
                if key in relation:
                    after[key] = deepcopy(relation[key])
        objects[object_id].update(after)
    return result


def compose_session_state(
    previous_state: dict[str, Any], mutation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the current hidden mutation to the previous actual state."""
    return apply_changes(previous_state, (mutation or {}).get("changes", []))
