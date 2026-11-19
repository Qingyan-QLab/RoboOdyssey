"""Object identity, area names and references grounded in actual history. See LICENSE."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
import math
import re
from types import SimpleNamespace

from .state import ATTRIBUTE_KEYS, geometry


def _metadata_reference_holds(object_id, binding, history):
    if not isinstance(binding, dict):
        return False
    entity = history.entities[object_id]
    room, category = binding["room_id"], binding["category"]
    if entity.get("room_id") != room or entity.get("category") != category:
        return False
    peers = [
        o
        for o in history.entities.values()
        if o.get("room_id") == room and o.get("category") == category
    ]
    kind = binding.get("kind")
    if kind == "unique_category_in_room":
        return len(peers) == 1
    if not all(o.get("geometry", {}).get("world_aabb") for o in peers):
        return False
    if kind == "height_extreme":
        values = sorted(
            (
                -binding["sign"]
                * (o["geometry"]["world_aabb"]["max"][2] - o["geometry"]["world_aabb"]["min"][2]),
                o["object_id"],
            )
            for o in peers
        )
    elif kind == "nearest_landmark":
        anchor = history.entities.get(binding["anchor_object_id"])
        if (
            not anchor
            or anchor.get("room_id") != room
            or not anchor.get("geometry", {}).get("world_aabb")
            or sum(
                o.get("room_id") == room and o.get("category") == anchor.get("category")
                for o in history.entities.values()
            )
            != 1
        ):
            return False

        def center(o):
            b = o["geometry"]["world_aabb"]
            return [(b["min"][i] + b["max"][i]) / 2 for i in range(3)]

        values = sorted((math.dist(center(o), center(anchor)), o["object_id"]) for o in peers)
    else:
        return False
    return values[0][1] == object_id and (
        len(values) == 1 or values[1][0] - values[0][0] >= binding["margin_m"]
    )


def same_location(current, initial):
    """Compare physical locations, independent of simulator geometry encoding.

    USD readback may provide an AABB without the cached transform matrix. Small
    float round-off is not a changed landmark; meaningful movement still is.
    This v5 comparison does not alter legacy archive/runtime contracts.
    """
    if any(current.get(k) != initial.get(k) for k in ("room_id", "support_object_id")):
        return False
    if bool(current.get("held")) != bool(initial.get("held")):
        return False
    try:
        a, b = geometry(current)["world_aabb"], geometry(initial)["world_aabb"]
        return all(
            len(a[side]) == len(b[side]) == 3
            and all(abs(x - y) <= 1e-4 for x, y in zip(a[side], b[side]))
            for side in ("min", "max")
        )
    except (KeyError, TypeError, ValueError):
        return False


class RepairUnavailable(ValueError):
    def __init__(self, status, reason, *, code=None, dependency=None):
        super().__init__(reason)
        self.status = status
        self.code = code or status
        self.dependency = dependency


def _dependency_unavailable(binding, matches, *, named=False):
    """Explain missing experience without assuming that a missing log is failure."""
    dependency = {
        key: binding[key] for key in ("session_id", "task_id", "object_id") if key in binding
    }
    if not matches:
        code, reason = (
            "history_record_missing",
            "no actual execution record for the bound source task",
        )
    elif len(matches) > 1:
        code, reason = (
            "history_event_ambiguous",
            "multiple actual events match the bound source task",
        )
    else:
        event = matches[0]
        dependency.update(
            source_success=event.get("success"),
            source_status=event.get("task_status"),
            source_reason=event.get("reason_code"),
        )
        if event.get("dispatched") is False:
            code, reason = (
                "source_task_not_dispatched",
                "source task was not dispatched; no requested experience occurred",
            )
        elif named:
            code, reason = (
                "named_place_not_established",
                "source task did not establish the named place",
            )
        else:
            code, reason = (
                "source_experience_missing",
                "source record does not establish the required experience",
            )
    raise RepairUnavailable("dependency_unsatisfied", reason, code=code, dependency=dependency)


def area_name(cache, state, room_id):
    area = cache["areas"].get(room_id)
    if not area:
        raise RepairUnavailable(
            "state_unavailable", "target has no known area", code="area_unknown"
        )
    for oid in area["anchor_object_ids"]:
        if oid not in state["objects"] or not same_location(
            state["objects"][oid], cache["objects"][oid]["initial"]
        ):
            raise RepairUnavailable(
                "reference_unavailable",
                "area landmark changed: " + oid,
                code="area_landmark_changed",
            )
    return area["text"]


def _with_article(text):
    return text if text.lower().startswith(("the ", "a ", "an ")) else "the " + text


def _appearance(cache, state, oid, peers):
    traits = cache["objects"][oid]["identity_attributes"]
    if state["objects"][oid].get("identity_attributes") is not None:
        traits = state["objects"][oid]["identity_attributes"]
    keys = [key for key in ATTRIBUTE_KEYS if traits.get(key)]
    for count in range(1, len(keys) + 1):
        for subset in combinations(keys, count):
            # An unknown trait does not prove that another object is different.
            def differs(other):
                other_traits = state["objects"][other].get(
                    "identity_attributes",
                    cache["objects"].get(other, {}).get("identity_attributes", {}),
                )
                return any(other_traits.get(k) and other_traits[k] != traits[k] for k in subset)

            if all(differs(other) for other in peers if other != oid):
                return (
                    "the "
                    + " ".join(traits[k] for k in subset)
                    + " "
                    + cache["objects"][oid]["category_name"]
                )
    return None


def object_reference(cache, state, oid, *, hides_room=False, preferred=None):
    entities = state["objects"]
    if oid not in entities:
        raise RepairUnavailable(
            "target_unavailable", "target object is absent", code="target_absent"
        )
    if oid not in cache["objects"]:
        raise RepairUnavailable(
            "target_unavailable",
            "target object is missing from scene cache",
            code="target_cache_missing",
        )
    item, obj = cache["objects"][oid], entities[oid]
    peers = [
        other for other, entity in entities.items() if entity.get("category") == item["category"]
    ]
    if oid not in peers:
        raise RepairUnavailable(
            "state_unavailable", "target category changed", code="target_category_changed"
        )
    initial_peers = {
        other for other, value in cache["objects"].items() if value["category"] == item["category"]
    }
    normalized = {
        other: {**entity, "geometry": geometry(entity)} for other, entity in entities.items()
    }
    context = SimpleNamespace(entities=normalized)
    for expression in sorted(item["expressions"], key=lambda e: e["text"] != preferred):
        if any(
            entities[other].get("identity_attributes") is not None
            and entities[other]["identity_attributes"]
            != cache["objects"].get(other, {}).get("identity_attributes", {})
            for other in peers
        ):
            continue
        if hides_room and (
            expression["spatial"]
            or re.search(
                r"\b(bedroom|bathroom|kitchen|study|balcony|corridor|living room|passage)\b",
                expression["text"],
                re.I,
            )
            or any(
                area["text"].lower() in expression["text"].lower()
                for area in cache["areas"].values()
            )
        ):
            continue
        binding = expression.get("binding")
        if binding:
            valid = _metadata_reference_holds(oid, binding, context)
        else:
            # Unstructured legacy claims are valid only in the same comparison
            # set and context. Never carry an initial spatial phrase across moves.
            dependencies = set(peers)
            if expression["spatial"]:
                dependencies.update(expression["dependency_object_ids"])
            valid = set(peers) == initial_peers and all(
                other in entities
                and other in cache["objects"]
                and same_location(entities[other], cache["objects"][other]["initial"])
                for other in dependencies
            )
            valid = valid and not any(
                other != oid
                and any(
                    e["text"].casefold() == expression["text"].casefold()
                    for e in value["expressions"]
                )
                for other, value in cache["objects"].items()
            )
        if valid:
            return _with_article(expression["text"])
        if not hides_room and not binding and expression["spatial"]:
            room = obj.get("room_id")
            local = {other for other in peers if entities[other].get("room_id") == room}
            original_local = {
                other
                for other in initial_peers
                if cache["objects"][other]["initial"]["room_id"] == room
            }
            dependencies = local | {oid} | set(expression["dependency_object_ids"])
            if local <= original_local and all(
                other in entities
                and other in cache["objects"]
                and same_location(entities[other], cache["objects"][other]["initial"])
                for other in dependencies
            ):
                # The original unique visual description remains valid within
                # an unchanged subset. Add its scope after peers move elsewhere.
                name = area_name(cache, state, room)
                phrase = _with_article(expression["text"])
                return (
                    phrase if name.casefold() in phrase.casefold() else phrase + " in the " + name
                )
    if len(peers) == 1:
        return "the " + item["category_name"]
    phrase = _appearance(cache, state, oid, peers)
    if phrase:
        return phrase
    if hides_room:
        raise RepairUnavailable(
            "reference_unavailable",
            "no unique reference that hides the answer",
            code="answer_safe_reference_unavailable",
        )
    room = obj.get("room_id")
    local = [other for other in peers if entities[other].get("room_id") == room]
    name = area_name(cache, state, room)
    if len(local) == 1:
        return "the " + item["category_name"] + " in the " + name
    phrase = _appearance(cache, state, oid, local)
    if phrase:
        return phrase + " in the " + name
    # Re-evaluate only authored geometric qualifiers; do not invent a ranking
    # that was never validated during construction.
    for expression in item["expressions"]:
        binding = deepcopy(expression.get("binding"))
        if not binding or binding["kind"] not in {"height_extreme", "nearest_landmark"}:
            continue
        binding["room_id"] = room
        if not _metadata_reference_holds(oid, binding, context):
            continue
        if binding["kind"] == "height_extreme":
            adjective = "tallest" if binding["sign"] == 1 else "shortest"
            return f"the {adjective} {item['category_name']} in the {name}"
        anchor = cache["objects"][binding["anchor_object_id"]]["category_name"]
        return f"the {item['category_name']} closest to the {anchor} in the {name}"
    raise RepairUnavailable(
        "reference_unavailable",
        "target cannot be uniquely described with valid cached evidence",
        code="object_reference_ambiguous",
    )
