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
    mutation = mutation or {}
    changes = deepcopy(mutation.get('changes', []))
    if mutation.get('geometry_after'):
        if len(changes) != 1:
            raise ValueError('VLM mutation geometry requires exactly one target')
        from .dataset_loader import canonical_state_from_metadata
        change = changes[0]
        obj = deepcopy(previous_state['objects'][change['object_id']])
        obj.update(geometry=mutation['geometry_after'], held=False, support_object_id=None)
        obj['room_id'] = change['relation_after']['room_id']
        change['after'] = canonical_state_from_metadata({'objects': [obj]}, '')['objects'][change['object_id']]
    result = apply_changes(previous_state, changes)
    held = result.get('held_object')
    if held and any(c.get('object_id') == held for c in changes) and not result['objects'][held].get('held'):
        result['held_object'] = None
    return result
