"""Private room answers from explicit current and task-boundary evidence."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .types import StateSource, Status


@dataclass(frozen=True)
class RoomSnapshot:
    object_rooms: Mapping[str, str]
    state_source: StateSource

    def __post_init__(self):
        if self.state_source not in ("planned", "runtime"):
            raise ValueError("state_source must be planned or runtime")
        rooms = dict(self.object_rooms)
        if any(not isinstance(k, str) or not k or not isinstance(v, str) or not v
               for k, v in rooms.items()):
            raise ValueError("known object and room IDs must be nonempty strings")
        object.__setattr__(self, "object_rooms", MappingProxyType(rooms))


@dataclass(frozen=True)
class TaskHistory:
    session_id: str
    task_id: str
    before: RoomSnapshot
    after: RoomSnapshot
    issued: bool = True

    def __post_init__(self):
        if not self.session_id or not self.task_id or type(self.issued) is not bool:
            raise ValueError("history needs an explicit task identity and issued flag")
        if self.before.state_source != self.after.state_source:
            raise ValueError("history cannot mix planned and runtime snapshots")


@dataclass(frozen=True)
class AnswerResult:
    status: Status
    value: str | bool | None = None
    accepted_answers: tuple[str, ...] = ()
    reason: str = ""
    reference_version: str = "room_history_answers_v1"


def room_answer(query: str, object_id: str, current: RoomSnapshot, *,
                area_names: Mapping[str, str], history: TaskHistory | None = None,
                source_task: tuple[str, str] | None = None, invert: bool = False) -> AnswerResult:
    """Resolve a bound query without inventing absent history or public names.

    has_changed compares the current room with the *after* room of the bound
    task, as contextual_requests v3 specifies. Callers resolve language, choose
    the historical binding, authorize fallback conversions and score responses.
    """
    if query not in ("current_room", "room_before", "room_after", "has_changed"):
        return AnswerResult("invalid", reason="unsupported_query")
    if type(invert) is not bool or (invert and query != "has_changed"):
        return AnswerResult("invalid", reason="invalid_answer_mapping")
    if query in ("current_room", "has_changed") and object_id not in current.object_rooms:
        return AnswerResult("invalid", reason="missing_current_object_room")
    if query != "current_room":
        if history is None or source_task is None:
            return AnswerResult("invalid", reason="missing_bound_history")
        if source_task != (history.session_id, history.task_id):
            return AnswerResult("invalid", reason="history_binding_mismatch")
        if not history.issued:
            return AnswerResult("invalid", reason="source_task_not_issued")
        if history.after.state_source != current.state_source:
            return AnswerResult("invalid", reason="state_source_mismatch")
        snapshot = history.before if query == "room_before" else history.after
        if object_id not in snapshot.object_rooms:
            return AnswerResult("invalid", reason="missing_historical_object_room")
        room = snapshot.object_rooms[object_id]
    else:
        room = current.object_rooms[object_id]
    if query == "has_changed":
        changed = current.object_rooms[object_id] != room
        answer = not changed if invert else changed
        return AnswerResult("ok", answer, ("yes", "true") if answer else ("no", "false"))
    name = area_names.get(room)
    if not isinstance(name, str) or not name.strip():
        return AnswerResult("invalid", reason="missing_public_area_name")
    return AnswerResult("ok", name, (name,))
