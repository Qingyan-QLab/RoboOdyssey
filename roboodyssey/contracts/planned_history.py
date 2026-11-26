"""Private, offline event ledger for VLM plans, never an observation log.

All transitions assume successful execution of deterministic compiled GT.
An unknown placement pose invalidates geometry rather than retaining the old AABB.
"""
from __future__ import annotations

from copy import deepcopy
import re


class HistoryError(ValueError):
    """A proposed event contradicts the current state or available history."""


class PlannedHistory:
    def __init__(self, entities: list[dict], rooms: list[dict]) -> None:
        self.entities = {str(item["object_id"]): deepcopy(item) for item in entities}
        self.room_ids = {str(room["room_id"]) for room in rooms}
        self.relations = {
            oid: {"type": "in_room", "room_id": item.get("room_id")}
            for oid, item in self.entities.items()
        }
        self.events: list[dict] = []
        self.recorded_tasks: set[tuple[str, str]] = set()
        self.known_events: list[str] = []
        self.geometry_known = {oid: bool(item.get("geometry", {}).get("world_aabb"))
                               for oid, item in self.entities.items()}

    def resolve(self, reference: dict, *, before_session: str | None = None,
                require_public: bool = True) -> dict:
        """Resolve one exact event; session and task alone may be ambiguous."""
        matches = [event for event in self.events
                   if event["session_id"] == reference.get("session_id")
                   and event["task_id"] == reference.get("task_id")
                   and event["object_id"] == reference.get("object_id")]
        if len(matches) != 1:
            raise HistoryError("memory reference must resolve to exactly one prior event")
        event = matches[0]
        if before_session is not None:
            def index(value: str) -> int:
                if not re.fullmatch(r"session_\d+", value):
                    raise HistoryError("invalid session ID in historical reference")
                return int(value.split("_")[1])
            if index(event["session_id"]) >= index(before_session):
                raise HistoryError("cross-session memory requires an earlier session")
        if require_public and event["event_id"] not in self.known_events:
            raise HistoryError("hidden world changes are not agent knowledge")
        return deepcopy(event)

    def record(self, session_id: str, task: dict, truth: dict) -> list[dict]:
        """Atomically apply a compiled task and record its expected experience."""
        if task.get("task_id") != truth.get("task_id"):
            raise HistoryError("task and GT IDs differ")
        task_id = task["task_id"]
        if (session_id, task_id) in self.recorded_tasks:
            raise HistoryError("duplicate history task")
        candidate = deepcopy(self)
        candidate.recorded_tasks.add((session_id, task_id))
        changes = truth.get("expected_outcome", {}).get("state_changes", [])
        changed = set()
        for change in changes:
            oid = str(change.get("object_id"))
            if oid in changed:
                raise HistoryError("multiple transitions for one object require separate tasks")
            changed.add(oid)
            candidate._transition(session_id, task_id, change, hidden=False)
        # Merely listing an object as a QA target does not establish visual exposure.
        # Explicit navigation establishes a planned visit, not a proven observation.
        if task.get("task_type") == "navigation":
            for oid in task.get("target", {}).get("object_ids", []):
                oid = str(oid)
                if oid not in changed:
                    candidate._append(session_id, task_id, oid, "planned_visit",
                                      candidate.relations.get(oid), candidate.relations.get(oid), False)
        new_events = candidate.events[len(self.events):]
        self.__dict__.update(candidate.__dict__)
        return deepcopy(new_events)

    def accept_placement(self, object_id: str, placement: dict) -> None:
        """Install geometry produced by the trusted placement module after record().

        Callers must not pass VLM response geometry here. This method checks
        identity/relation consistency; geometric validation belongs to placement.py.
        """
        entity = placement.get("entity", {})
        if (entity.get("object_id") != object_id or
                placement.get("relation_after") != self.relations.get(object_id) or
                entity.get("room_id") != self.entities[object_id].get("room_id") or
                not entity.get("geometry", {}).get("world_aabb")):
            raise HistoryError("placement geometry does not match recorded transition")
        self.entities[object_id] = deepcopy(entity)
        self.geometry_known[object_id] = True

    def mutate(self, session_id: str, mutation_id: str, change: dict) -> dict:
        candidate = deepcopy(self)
        if any(e["session_id"] == session_id and e["task_id"] == mutation_id
               for e in self.events):
            raise HistoryError("duplicate mutation event")
        candidate._transition(session_id, mutation_id, change, hidden=True)
        self.__dict__.update(candidate.__dict__)
        return deepcopy(self.events[-1])

    def _transition(self, session_id: str, task_id: str, change: dict, hidden: bool) -> None:
        oid = str(change.get("object_id"))
        if oid not in self.entities:
            raise HistoryError(f"unknown transition object: {oid}")
        before, after = change.get("relation_before"), change.get("relation_after")
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise HistoryError("location transition needs explicit before and after relations")
        before, after = self._relation(before), self._relation(after)
        if before != self.relations[oid]:
            raise HistoryError(f"stale precondition for {oid}: {before} != {self.relations[oid]}")
        if before == after:
            raise HistoryError("location transition must change the relation")
        room_id = after.get("room_id")
        if after["type"] == "on":
            support = self.entities.get(after["object_id"])
            if support is None or after["object_id"] == oid:
                raise HistoryError("unknown or self support")
            room_id = support.get("room_id")
        if room_id not in self.room_ids:
            raise HistoryError("destination must resolve to a known room")
        self.relations[oid] = after
        self.entities[oid]["room_id"] = room_id
        # Relation-only GT cannot justify a particular physical pose.
        self.geometry_known[oid] = False
        self.entities[oid].pop("geometry", None)
        self.entities[oid].pop("world_aabb", None)
        self._append(session_id, task_id, oid, "hidden_change" if hidden else "planned_transfer",
                     before, after, hidden)

    @staticmethod
    def _relation(value: dict) -> dict:
        kind = value.get("type", value.get("relation_type"))
        if kind == "in_room":
            return {"type": kind, "room_id": value.get("room_id")}
        if kind == "on":
            return {"type": kind, "object_id": value.get("object_id", value.get("support_object_id"))}
        raise HistoryError(f"unsupported relation: {kind}")

    def _append(self, session_id: str, task_id: str, oid: str, kind: str,
                before: dict | None, after: dict | None, hidden: bool) -> None:
        if oid not in self.entities:
            raise HistoryError(f"unknown event object: {oid}")
        event_id = f"{session_id}/{task_id}/{oid}"
        self.events.append({"event_id": event_id, "session_id": session_id,
                            "task_id": task_id, "object_id": oid, "kind": kind,
                            "relation_before": deepcopy(before), "relation_after": deepcopy(after),
                            "evidence_status": "planned_success_assumption",
                            "agent_information": "hidden" if hidden else "expected_task_experience"})
        if not hidden:
            self.known_events.append(event_id)

    def snapshot(self) -> dict:
        return deepcopy({"schema_version": 1, "evidence_status": "planned_success_assumption",
                         "relations": self.relations, "geometry_known": self.geometry_known,
                         "events": self.events, "known_event_ids": self.known_events})
