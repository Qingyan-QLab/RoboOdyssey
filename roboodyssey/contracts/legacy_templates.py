"""Legacy frozen template validation and replay comparison. See LICENSE."""

from __future__ import annotations

from copy import deepcopy
import math
import re

from .language import clauses_for, clause_options, validate_realization


VERSION = 1

SLOTS = {"target_reference", "destination_area", "place_name"}


def task_contract(proposal, truth):
    evidence = truth.get("memory_evidence", {})
    event = evidence.get("event")
    selector = evidence.get("selector", {})
    contract = {
        "task_id": proposal["task_id"],
        "task_type": proposal["task_type"],
        "reference_mode": "history" if event else "scene",
    }
    if event:
        contract["historical_reference"] = deepcopy(evidence.get("public_reference")) or {
            "session_number": int(event["session_id"].split("_")[-1]) + 1,
            "event_action": "moved" if selector["kind"] == "planned_transfer" else "visited",
            "ordinal": selector["ordinal"],
        }
    kind = proposal["task_type"]
    if kind == "navigation":
        contract["intent"] = "visit_object"
        if proposal.get("named_place"):
            named = proposal["named_place"]
            contract.update(
                intent="named_place", place_name=named["name"], place_mode=named["mode"]
            )
    elif kind == "manipulation":
        phase = proposal.get("destination_from_history")
        contract.update(
            intent="transfer_object",
            destination_mode="history" if phase else "explicit",
            destination_phase=phase,
            destination_room_id=proposal.get("destination_room_id"),
        )
    else:
        contract.update(intent="answer_question", query=proposal.get("query", "current_room"))
    return contract


def freeze_template(proposal, task, truth):
    """Extract slots from the bounded language grammar, preserving VLM wording.

    This also migrates existing archives without asking a model to reword tasks.
    Only reviewed clause alternatives and the original connective grammar pass.
    """
    contract = task_contract(proposal, truth)
    symbolic = deepcopy(contract)
    symbolic.update(
        reference_mode="scene",
        destination_mode="explicit",
        destination_area_name="ZZDESTINATION",
        destination_room_id="destination",
        place_name="ZZPLACE",
    )
    clauses = clauses_for(symbolic, object_phrase="ZZTARGET")
    substitutions = {
        "the ZZTARGET": "target_reference",
        "the ZZDESTINATION": "destination_area",
        "ZZPLACE": "place_name",
    }
    field = "question" if task["task_type"] == "qa" else "instruction"
    text = task[field]
    remainder, output, actual_clauses, slots = text, [], [], {}
    for index, clause in enumerate(clauses):
        patterns = []
        for option in clause_options(clause):
            pattern = re.escape(option)
            for token, slot in substitutions.items():
                pattern = pattern.replace(re.escape(token), "(?P<" + slot + ">.+?)")
            # Next semantic clause starts after a connective. A question ends
            # here; instruction final clauses consume the remaining sentence.
            patterns.append(pattern)
        match = None
        ending = r"(?=[.!?]?\s*$)"
        if index + 1 < len(clauses):
            prefixes = [
                re.split("ZZTARGET|ZZDESTINATION|ZZPLACE", option)[0]
                for option in clause_options(clauses[index + 1])
            ]
            ending = (
                r"(?=(?:\s+|[,;.!?]\s*)(?:(?:and|then)\s+)*(?:"
                + "|".join(map(re.escape, prefixes))
                + "))"
            )
        for pattern in patterns:
            candidate = re.search(pattern + ending, remainder, re.IGNORECASE)
            if candidate and (match is None or candidate.start() < match.start()):
                match = candidate
        if match is None:
            raise ValueError("cannot freeze task wording: " + task["task_id"])
        actual = match.group(0)
        rendered = actual
        for slot, value in match.groupdict().items():
            if value is not None:
                slots[slot] = value
                rendered = rendered.replace(value, "{" + slot + "}", 1)
        output.extend([remainder[: match.start()], rendered])
        actual_clauses.append(actual)
        remainder = remainder[match.end() :]
    output.append(remainder)
    legacy = (
        contract["reference_mode"] == "history"
        and "source_time" not in contract["historical_reference"]
    )
    validate_realization(text, actual_clauses, allow_legacy=legacy)
    evidence = deepcopy(truth.get("memory_evidence", {}))
    result = {
        "schema_version": VERSION,
        "task_id": task["task_id"],
        "field": field,
        "text": "".join(output),
        "preview_slots": slots,
        "contract": contract,
        "target_object_id": proposal["target_object_id"],
        "execution_dependencies": deepcopy(proposal.get("depends_on", [])),
        "history_binding": evidence.get("event"),
        "history_selector": evidence.get("selector"),
        "named_place": deepcopy(proposal.get("named_place")),
        "missing_history_policy": "dependency_unsatisfied",
    }
    if "issued_at" in task:
        result["issued_at"] = task["issued_at"]
    if task["task_type"] == "manipulation":
        result["held_text"] = "Place {target_reference} on the floor in {destination_area}."
    validate_template(result)
    return result


def validate_template(template):
    if template.get("schema_version") != VERSION:
        raise ValueError("unsupported task template version")
    if template.get("missing_history_policy") != "dependency_unsatisfied":
        raise ValueError("unsupported missing history policy")
    for key in ("text", "held_text"):
        if key not in template:
            continue
        text = template[key]
        fields = re.findall(r"\{([^{}]*)\}", text)
        if set(fields) - SLOTS or re.search(r"[{}]", re.sub(r"\{[^{}]*\}", "", text)):
            raise ValueError("invalid template slots")
        if set(fields) - set(template["preview_slots"]):
            raise ValueError("template slot has no preview value")
    return template


def replay_equal(first, second):
    """Allow only float roundoff across Python's summation implementations."""
    if isinstance(first, float) and isinstance(second, float):
        return math.isclose(first, second, rel_tol=1e-12, abs_tol=1e-12)
    if type(first) is not type(second):
        return False
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(
            replay_equal(first[k], second[k]) for k in first
        )
    if isinstance(first, list):
        return len(first) == len(second) and all(replay_equal(a, b) for a, b in zip(first, second))
    return first == second
