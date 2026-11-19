"""Frozen language, calendar references, naming and catalog contracts. See LICENSE."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import re
from string import Formatter

from ..artifacts.layout import json_digest as digest


class HistoryError(ValueError):
    """A proposed event contradicts the current state or available history."""


MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def parse_time(value):
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise HistoryError("task time must be an ISO 8601 timestamp with timezone") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.second or parsed.microsecond:
        raise HistoryError("task time requires a timezone and whole-minute precision")
    return parsed


def relative_time(source_time, current_time, *, include_clock=False):
    current = parse_time(current_time)
    source = parse_time(source_time).astimezone(current.tzinfo)
    if source >= current:
        raise HistoryError("historical reference requires an earlier request time")
    days = (current.date() - source.date()).days
    day = {0: "earlier today", 1: "yesterday", 2: "the day before yesterday"}.get(days)
    if day is None:
        day = f"on {MONTHS[source.month - 1]} {source.day}, {source.year}"
    return day + (f" at {source:%H:%M}" if include_clock or days == 0 else "")


def historical_reference(reference):
    noun = reference["object_name"]
    action = {"move": "pick up", "visit": "go to"}.get(reference["requested_action"])
    if (
        not action
        or not isinstance(noun, str)
        or not re.fullmatch(r"[A-Za-z]+(?:[ -][A-Za-z]+)*", noun)
    ):
        raise HistoryError("historical request requires a natural object name and supported action")
    when = relative_time(
        reference["source_time"],
        reference["current_time"],
        include_clock=reference["include_clock"],
    )
    return f"the {noun} I asked you to {action} {when}"


def validate_public_wording(text: str) -> None:
    """Reject privileged position encodings and numbered area identifiers."""
    if re.search(
        r"(?i)(?:center_xyz|world_aabb|world_matrix|\b[xyz]\s*[:=]|[\[(]\s*-?\d+(?:\.\d+)?\s*,\s*-?\d|(?<!\w)-?\d+\.\d+)",
        text,
    ):
        raise HistoryError("public task text must not contain scene coordinates")
    if re.search(
        r"\bsessions?\b|\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th))\s+(?:object|item|day)\b"
        r"|\bday\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|first|second|third)\b"
        r"|\b(?:object|item)\s+(?:number\s+)?\d+|第\s*[一二三四五六七八九十百\d]+\s*(?:天|个|次)",
        text,
        re.I,
    ):
        raise HistoryError(
            "use explicit calendar time and object descriptions, not session/object/day ordinals"
        )
    cleaned = text
    cleaned = re.sub(
        r"\b(?:" + "|".join(MONTHS) + r") (?:[1-9]|[12]\d|3[01]), \d{4}\b", "", cleaned, flags=re.I
    )
    cleaned = re.sub(r"\bat (?:[01]\d|2[0-3]):[0-5]\d\b", "", cleaned, flags=re.I)
    if re.search(r"\d", cleaned):
        raise HistoryError(
            "numeric public references are restricted to calendar dates and clock times"
        )
    if re.search(
        r"(?i)\b(?:corridor|living\s*room|bedroom|master\s*bedroom|bathroom|kitchen|study\s*room|balcony|room|area)[ _-]*\d+\b",
        text,
    ):
        raise HistoryError("use an evidence-grounded area name, not a numbered area ID")


def room_name(room_id: str) -> str:
    value = re.sub(r"([a-z])([A-Z])", r"\1 \2", room_id)
    value = re.sub(r"([A-Za-z])(\d)", r"\1 \2", value)
    return value.replace("_", " ").replace("#", " ").lower()


def clauses_for(contract: dict, *, object_phrase: str | None = None) -> list[str]:
    """Turn a validated semantic contract into safe, meaning-bearing clauses."""
    if contract.get("intent") == "named_place" and contract.get("place_mode") == "revisit":
        return [f"Return to {contract['place_name']}"]
    if contract["reference_mode"] == "history":
        reference = contract["historical_reference"]
        if "source_time" in reference:
            target = historical_reference(reference)
        else:
            raise HistoryError("historical reference requires explicit request time")
    else:
        if not isinstance(object_phrase, str) or not object_phrase.strip():
            raise HistoryError("scene reference requires a validated object phrase")
        validate_public_wording(object_phrase)
        target = object_phrase.strip().rstrip(".")
        if not target.casefold().startswith(("the ", "a ", "an ")):
            target = "the " + target
    intent = contract["intent"]
    if intent == "named_place":
        return [f"Go to {target}", f"remember this place as {contract['place_name']}"]
    if intent == "visit_object":
        return [f"Go to {target}"]
    if intent == "transfer_object":
        if contract["destination_mode"] == "history":
            phase = contract["destination_phase"]
            location = (
                "the area where it was before that move"
                if phase == "before"
                else "the area where you placed it during that move"
            )
        else:
            location = "the " + contract.get(
                "destination_area_name", room_name(contract["destination_room_id"])
            )
            validate_public_wording(location)
        return [f"Pick up {target}", f"place it on the floor in {location}"]
    query = contract["query"]
    if query == "room_before":
        return [f"Which area contained {target} before that move?"]
    if query == "room_after":
        return [f"Which area contained {target} immediately after that move?"]
    if query == "current_room":
        return [f"Find {target}", "report which area currently contains it"]
    if query == "has_changed":
        return [
            f"Find {target}",
            "report whether its area differs from where you left it during that move",
        ]
    raise HistoryError("unsupported language intent")


def clause_options(clause: str) -> list[str]:
    """Reviewed equivalent verbs; target, time and destination stay unchanged."""
    alternatives = [clause]
    for prefix, replacements in (
        ("Go to ", ("Navigate to ", "Move to ")),
        ("Return to ", ("Go back to ", "Navigate back to ")),
        ("Find ", ("Locate ",)),
        ("Pick up ", ("Lift ",)),
        ("place it on the floor in ", ("set it down on the floor in ",)),
        ("report which area currently contains it", ("tell me which area it is in now",)),
    ):
        if clause.startswith(prefix):
            alternatives.extend(value + clause[len(prefix) :] for value in replacements)
    # Cached descriptions sometimes say "on left nightstand". The model's
    # grammatical article changes no referent, direction, action or destination.
    for option in list(alternatives):
        grammatical = re.sub(
            r"\b(on|at|near|beside|under|above|below) (left|right)\b",
            r"\1 the \2",
            option,
            flags=re.IGNORECASE,
        )
        if grammatical not in alternatives:
            alternatives.append(grammatical)
    return alternatives


def validate_realization(text: str, clauses: list[str]) -> str:
    if not isinstance(text, str) or not text.strip() or len(text) > 2000:
        raise HistoryError("language output must be non-empty bounded text")
    validate_public_wording(text)
    remainder = text.strip()
    gaps = []
    for clause in clauses:
        matches = [
            (remainder.casefold().find(option.casefold()), option)
            for option in clause_options(clause)
        ]
        matches = [(offset, option) for offset, option in matches if offset >= 0]
        if not matches:
            raise HistoryError("language output omitted or changed a required semantic clause")
        offset, matched = min(matches, key=lambda value: value[0])
        gaps.append(remainder[:offset])
        remainder = remainder[offset + len(matched) :]
    gaps.append(remainder)
    # Closed connective grammar prevents a model adding a new location or negation.
    for gap in gaps:
        words = re.findall(r"[A-Za-z]+", gap.casefold())
        if any(word not in {"please", "then", "and"} for word in words):
            raise HistoryError("language output added unsupported content")
        if re.sub(r"[A-Za-z\s,.!?;:—-]", "", gap):
            raise HistoryError("language output added unsupported tokens")
    return text.strip()


def shorten_areas(text, registry, descriptions):
    for rid, description in sorted(descriptions.items(), key=lambda item: -len(item[1])):
        entry = registry.get("areas", {}).get(rid)
        if entry and description:
            text = re.sub(
                r"(?<!\w)" + re.escape(description) + r"(?!\w)", entry["en"], text, flags=re.I
            )
    return text


def validate_catalog(value):
    value = deepcopy(value)
    claimed = value.pop("sha256", None)
    if (
        set(value) != {"schema_version", "catalog_version", "templates"}
        or value.get("schema_version") != 3
        or type(value.get("catalog_version")) is not int
        or value["catalog_version"] < 1
        or not isinstance(value["templates"], list)
        or not value["templates"]
    ):
        raise ValueError("unsupported language catalog")
    seen, phrases = set(), set()
    for entry in value["templates"]:
        tid = entry["template_id"]
        if tid in seen or not re.fullmatch(r"[A-Z]+-\d\d", tid):
            raise ValueError("duplicate/invalid template id")
        seen.add(tid)
        required = {
            "kind",
            "query",
            "destination_mode",
            "target_count",
            "execution_order",
            "requires",
            "inspection",
            "answer_unit",
            "answer_mapping",
            "slots",
            "variants",
            "enabled",
            "disabled_reason",
            "template_id",
        }
        if set(entry) != required or not entry["variants"] or "base" not in entry["variants"]:
            raise ValueError("incomplete template semantics: " + tid)
        if (
            entry["kind"]
            not in {
                "visit",
                "transfer",
                "answer",
                "establish_place",
                "revisit_place",
                "introduction",
            }
            or entry["query"]
            not in {None, "current_room", "room_before", "room_after", "has_changed"}
            or entry["destination_mode"] not in {None, "explicit", "before", "after"}
            or entry["execution_order"] not in {"single", "bound_goal_order"}
            or entry["answer_unit"] not in {None, "named_area", "boolean"}
            or type(entry["inspection"]) is not bool
            or type(entry["enabled"]) is not bool
            or not isinstance(entry["requires"], list)
            or not set(entry["requires"])
            <= {"history", "topic", "answer_unit_introduced", "place_established"}
        ):
            raise ValueError("invalid template semantics: " + tid)
        counts = entry["target_count"]
        if (
            set(counts) != {"min", "max"}
            or any(type(v) is not int for v in counts.values())
            or not 0 <= counts["min"] <= counts["max"] <= 5
            or (entry["kind"] == "answer") != (entry["query"] is not None)
            or (entry["kind"] == "transfer") != (entry["destination_mode"] is not None)
            or entry["inspection"] != (entry["query"] in {"current_room", "has_changed"})
            or entry["answer_unit"]
            != (
                "boolean"
                if entry["query"] == "has_changed"
                else "named_area"
                if entry["kind"] == "answer"
                else None
            )
        ):
            raise ValueError("inconsistent template semantics: " + tid)
        known_slots = {
            "target": "object_reference",
            "first_target": "object_reference",
            "second_target": "object_reference",
            **{f"target_{i}": "object_reference" for i in range(5)},
            "history_target": "historical_object_reference",
            "source_request": "request_reminder",
            "location_unit": "answer_unit",
            "destination": "area_reference",
            "place_name": "place_name",
            "description": "identity_description",
            "name": "name",
        }
        if not isinstance(entry["slots"], dict) or any(
            known_slots.get(k) != v for k, v in entry["slots"].items()
        ):
            raise ValueError("unknown typed slot: " + tid)
        if entry["answer_mapping"] not in {"identity", "not_has_changed"}:
            raise ValueError("unknown answer mapping")
        if entry["answer_mapping"] == "not_has_changed" and entry["query"] != "has_changed":
            raise ValueError("polarity applies only to change queries")
        if not entry["enabled"] and not entry["disabled_reason"]:
            raise ValueError("disabled template requires a reason")
        used = set()
        for variant, phrase in entry["variants"].items():
            if (
                (variant != "base" and variant not in {"list_3", "list_4", "list_5"})
                or not isinstance(phrase, str)
                or not phrase.strip()
            ):
                raise ValueError("invalid approved variant: " + tid)
            parsed = list(Formatter().parse(phrase))
            if any(spec or conv for _, name, spec, conv in parsed):
                raise ValueError("format conversion/specification is not a typed slot")
            fields = [name for _, name, spec, conv in parsed if name is not None]
            if any(not re.fullmatch(r"\w+", name) for name in fields):
                raise ValueError("untyped template field")
            used.update(fields)
            normalized = re.sub(r"[^a-z{} ]", "", re.sub(r"\{\w+\}", "{}", phrase.lower()))
            if normalized in phrases:
                raise ValueError("duplicate template expression: " + tid)
            phrases.add(normalized)
        if used != set(entry["slots"]) or any(not v for v in entry["slots"].values()):
            raise ValueError("missing/unused typed slots: " + tid)
    value["sha256"] = digest(value)
    if claimed is not None and claimed != value["sha256"]:
        raise ValueError("catalog hash mismatch")
    return value


def render(expression, slots):
    text = expression.format_map(slots)
    validate_public_wording(text)
    return text


def render_request(template, slots=None, introductions=None):
    slots = template["preview_slots"] if slots is None else slots
    introductions = template["introductions"] if introductions is None else introductions
    return " ".join(
        [
            *(render(i["snapshot"]["expression"], i["slots"]) for i in introductions),
            render(template["main"]["expression"], slots),
        ]
    )
