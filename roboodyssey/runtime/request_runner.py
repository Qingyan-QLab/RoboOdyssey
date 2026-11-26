"""Execute the current contextual request contract against actual world state."""

from copy import deepcopy
import json
import uuid
import time
from pathlib import Path
from ..data.frozen import SequenceData
from roboodyssey.contracts.contextual import (
    normalize_state,
    instantiate_request_v3,
    execution_events_v3,
    prepare_mutation_v3,
    RuntimeContractError,
)
from ..evaluation.requests import evaluate_request, request_totals
from ..evaluation.task_metrics import evaluate_metrics, prepare_reference
from ..evaluation.transfer import TransferReference, collision_digest
from .environment import RequestEnvironment, RequestEnvironmentFactory
from .session_runner import _jsonable, _merge_memory
from .observations import capture_observation
from .request_events import RequestEvents, state_changes, source_context
from ..reporting.request_audit import write_request_audit
from ..artifacts.layout import read_json as read, write_json as write
from ..artifacts.layout import (
    create_session,
    session_file,
    compact_outcome,
    compact_history,
)


def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def actual_state(env: RequestEnvironment, cache):
    raw = env.runtime_state_v5()
    raw = deepcopy(raw)
    raw.setdefault("robot_pose", env.get_agent_pose())
    return normalize_state(cache, raw)


def public_observation(observation, cache, ledger):
    result = {
        k: deepcopy(observation[k])
        for k in ("images", "depth", "pose", "reset_applied", "timing")
        if k in observation
    }
    held = observation.get("held_object")
    result["holding_object"] = bool(held)
    if held and "objects/" + held in ledger:
        result["held_object_name"] = cache["names"]["objects"][held]["en"]
    feedback = observation.get("action_feedback")
    if isinstance(feedback, dict):
        result["action_feedback"] = {
            k: deepcopy(feedback[k])
            for k in (
                "success",
                "reason",
                "collision",
                "executed_distance",
                "requested_distance",
            )
            if k in feedback
        }
    return result


def resolve_action(action, cache, ledger):
    if not isinstance(action, dict):
        return action
    action = deepcopy(action)
    if action.get("object_id") or action.get("room_id"):
        return None
    if action.get("target"):
        query = str(action["target"]).strip().casefold().removeprefix("the ")
        matches = [
            oid
            for oid, n in cache.get("names", {}).get("objects", {}).items()
            if n["en"].casefold() == query and "objects/" + oid in ledger
        ]
        if len(matches) != 1:
            return None
        action["object_id"] = matches[0]
        action.pop("target")
    return action


def load_runtime_context(root, scene=None):
    from ..data import load_sequence
    return load_sequence(root, scene=scene).mutation_context()


def apply_boundary(env: RequestEnvironment, context, mutations, cache, history, output, events=None):
    for mutation in mutations:
        before = actual_state(env, cache)
        result = prepare_mutation_v3(context, mutation, cache, before, history)
        if result["status"] == "ready":
            if not hasattr(env, "apply_mutation_v5"):
                raise RuntimeContractError(
                    "environment has no verified v5 mutation method"
                )
            env.apply_mutation_v5(result)
            after = actual_state(env, cache)
            oid = result["object_id"]
            intended = result["geometry"]["world_aabb"]
            actual = after["objects"][oid]["geometry"]["world_aabb"]
            confirmed = after["objects"][oid]["room_id"] == result[
                "destination_room_id"
            ] and all(
                (
                    abs(a - b) <= 0.05
                    for side in ("min", "max")
                    for a, b in zip(actual[side], intended[side])
                )
            )
            result.update(
                confirmed=confirmed, actual_geometry=after["objects"][oid]["geometry"]
            )
            if not confirmed:
                result.update(
                    status="not_applied",
                    reason="actual mutation readback differs from requested placement",
                )
        after = actual_state(env, cache)
        record = {
            "planned": mutation,
            "result": result,
            "state_changes": state_changes(before, after),
        }
        append(session_file(output, "runtime_mutations.jsonl"), record)
        if events:
            events.emit(
                "mutation", sample=True, reason_code=result["status"], mutation=record
            )


def run_requests(runner, scene_id, sessions, canonical_state, env_factory: RequestEnvironmentFactory):
    env = events = None
    history, ledger, memory = ([], {}, {})
    output, results, pending = (None, [], None)
    run_error = None
    resets = 0
    sequence_id = sessions[0].metadata["sequence_id"]
    sequence_context = {"sequence_id": sequence_id, "run_id": uuid.uuid4().hex}
    if any((s.metadata["sequence_id"] != sequence_id for s in sessions)):
        raise RuntimeContractError("cannot combine Sessions from different sequences")
    try:
        for session in sessions:
            output = create_session(runner.output_root, scene_id, session.session_id)
            results, pending, resets = ([], None, 0)
            runner._current_output, runner._current_session_id = (
                output,
                session.session_id,
            )
            runner._agent_io = session_file(output, "agent_io.jsonl").open("w")
            paths = session.private_paths
            if session.metadata["runtime_contract"] != {
                "name": "contextual_requests",
                "version": 3,
            }:
                raise RuntimeContractError("unsupported request runtime contract")
            cache = read(paths["runtime_scene_cache"])
            truths, templates = (read(paths["ground_truth"]), read(paths["templates"]))
            if len(truths) != len(session.tasks) or len(templates) != len(
                session.tasks
            ):
                raise RuntimeContractError("task/GT/template count mismatch")
            if env is None:
                env = env_factory(
                    scene_id, deepcopy(canonical_state), session.session_id
                )
                context = load_runtime_context(paths["dataset_root"], paths.get("scene_version_root"))
            events = RequestEvents(runner, env, output, scene_id, session.session_id)
            if runner.transition_callback:
                runner.transition_callback()
            events.emit(
                "session_start",
                sample=True,
                instruction=session.metadata["context"],
                session_instruction=session.metadata["context"],
                runtime_contract=session.metadata["runtime_contract"],
            )
            apply_boundary(
                env, context, read(paths["mutations"]), cache, history, output, events
            )
            lifecycle = {
                "sequence_context": {
                    **sequence_context,
                    "reset_memory": session is sessions[0],
                }
            }
            response = runner._request(
                {
                    "type": "session_start",
                    "scene_id": scene_id,
                    "session_id": session.session_id,
                    "instruction": session.metadata["context"],
                    "scene_briefing": session.metadata.get("scene_briefing", []),
                    "time_context": session.metadata["time_context"],
                    "task_ids": [t["task_id"] for t in session.tasks],
                    "memory": memory,
                    **lifecycle,
                }
            )
            if session is sessions[0] and (
                response.get("sequence_reset_ack") != sequence_context["run_id"]
            ):
                raise RuntimeContractError(
                    "Agent must acknowledge fresh sequence memory, including external service state"
                )
            _merge_memory(memory, response)
            for public, truth, template in zip(session.tasks, truths, templates):
                before = actual_state(env, cache)
                repair = instantiate_request_v3(
                    public, truth, template, cache, before, history, ledger
                )
                record = repair["repair_record"]
                decision_id = f"{session.session_id}:{events.counter + 1:06d}"
                metric_reference, metric_grid = ({"status": "not_required"}, None)
                if repair["status"] == "ready":
                    metric_reference, metric_grid = prepare_reference(
                        env, repair["runtime_ground_truth"], output
                    )
                append(
                    session_file(output, "runtime_tasks.jsonl"),
                    {
                        "original": public,
                        "original_ground_truth": truth,
                        "repair": repair,
                        "state_before": before,
                        "source_context": source_context(template, history),
                        "metric_reference": metric_reference,
                        "decision_event_id": decision_id,
                    },
                )
                events.emit(
                    "task_repair"
                    if record.get("changed") or record.get("conversions")
                    else "task_decision",
                    task_id=public["task_id"],
                    instruction=(repair.get("public_task") or public)["instruction"],
                    reason_code=repair["status"],
                    repair_record=record,
                )
                base = {
                    "task_id": public["task_id"],
                    "task_type": public["task_type"],
                    "task_family": public["task_family"],
                    "capabilities": (repair.get("public_task") or public)[
                        "capabilities"
                    ],
                    "repair_status": repair["status"],
                }
                stop_reason, agent_status = (None, None)
                if repair["status"] != "ready":
                    if repair["status"] == "unavailable" and (
                        not record.get("failure_causes")
                    ):
                        raise RuntimeContractError(
                            "unavailable lacks failure causality"
                        )
                    result = {
                        **base,
                        "scored": False,
                        "dispatched": False,
                        "success": None,
                        "reason_code": repair["status"],
                        "failure_causes": record.get("failure_causes", []),
                        "repair_record": record,
                    }
                    new = execution_events_v3(
                        session.session_id,
                        public,
                        truth,
                        before,
                        before,
                        dispatched=False,
                        success=None,
                        reason_code=repair["status"],
                        repair_record=record,
                    )
                    after = before
                    events.emit(
                        "task_excluded",
                        task_id=public["task_id"],
                        instruction=public["instruction"],
                        reason_code=repair["status"],
                    )
                else:
                    task, gt = (repair["public_task"], repair["runtime_ground_truth"])
                    task_started = time.monotonic()
                    request_seconds = action_seconds = 0.0
                    request_count = 0
                    env.set_task(task)
                    poses, actions, observations, answer = ([], [], [], None)
                    runner._begin_observations(
                        task["task_id"],
                        lambda: capture_observation(env),
                        lambda obs: public_observation(obs, cache, ledger),
                    )
                    if runner.observation_archive is not None:
                        runner._capture_context = lambda: {
                            'scene_id': scene_id, 'session_id': session.session_id,
                            'task_id': task['task_id'], 'state': actual_state(env, cache),
                            'camera_pose': env.get_camera_pose() if hasattr(env, 'get_camera_pose') else None,
                        }
                    stop_reason = "step_limit"
                    for step in range(runner.max_steps):
                        if (
                            runner.reset_requested_callback
                            and runner.reset_requested_callback()
                        ):
                            raise RuntimeContractError(
                                "operator reset requested; interrupt this continuous evaluation instead of silently altering state"
                            )
                        if step == 0:
                            events.emit(
                                "task_start",
                                sample=True,
                                task_id=task["task_id"],
                                instruction=task["instruction"],
                                task_type="long_horizon",
                            )
                        observation, evidence = runner.observations.take(
                            runner.wait_callback
                        )
                        if step == 0:
                            pending = (session, task, gt, before, record)
                            ledger = repair["name_ledger"]
                        observations.extend(evidence)
                        poses.append(deepcopy(env.get_agent_pose()))
                        events.emit(
                            "agent_request",
                            task_id=task["task_id"],
                            instruction=task["instruction"],
                            step=step,
                        )
                        request_started = time.monotonic()
                        if runner.private_policy:
                            runner._set_policy_context(
                                scene_id=scene_id, session_id=session.session_id,
                                task=deepcopy(task), ground_truth=deepcopy(gt),
                                state=actual_state(env, cache), history=deepcopy(history),
                                cache=deepcopy(cache), ledger=deepcopy(ledger),
                                poses=deepcopy(poses), actions=deepcopy(actions),
                                observations=deepcopy(observations),
                                feedback=deepcopy(actions[-1]['result']) if actions else None,
                                reference=metric_grid, metric_reference=deepcopy(metric_reference),
                                raw_public_observation=public_observation(observation, cache, ledger),
                                navigation_snapshot=deepcopy(env.evaluation_navigation())
                                if hasattr(env, 'evaluation_navigation') else None,
                            )
                        response = runner._request(
                            {
                                "type": "step",
                                "scene_id": scene_id,
                                "session_id": session.session_id,
                                "task_id": task["task_id"],
                                "task_type": "long_horizon",
                                "instruction": task["instruction"],
                                "capabilities": task["capabilities"],
                                "name_introductions": task.get(
                                    "name_introductions", []
                                ),
                                "response_mode": task["response_mode"],
                                "current_time": task["issued_at"],
                                "depends_on": task["depends_on"],
                                "observation": _jsonable(
                                    public_observation(observation, cache, ledger)
                                ),
                                "memory": memory,
                                "sequence_context": sequence_context,
                                "allowed_actions": [
                                    "no_op",
                                    "move_forward",
                                    "move_backward",
                                    "move_left",
                                    "move_right",
                                    "turn_left",
                                    "turn_right",
                                    "look_up",
                                    "look_down",
                                ]
                                if any(
                                    (g["goal_kind"] == "answer" for g in gt["goals"])
                                )
                                else None,
                            }
                        )
                        request_elapsed = time.monotonic() - request_started
                        request_seconds += request_elapsed
                        request_count += 1
                        observations.extend(runner._consumed_stream_evidence())
                        _merge_memory(memory, response)
                        agent_status = response.get("agent_status", "ok")
                        if agent_status == "unsupported":
                            stop_reason = response.get(
                                "reason_code", "unsupported_capability"
                            )
                            events.emit(
                                "agent_unsupported",
                                task_id=task["task_id"],
                                reason_code=stop_reason,
                                instruction=task["instruction"],
                            )
                            break
                        answer = response.get("answer")
                        action = response.get("action", "no_op")
                        name = (
                            action if isinstance(action, str) else action.get("action")
                        )
                        pose = deepcopy(env.get_agent_pose())
                        action_before = actual_state(env, cache)
                        collision_before = (
                            collision_digest(env.evaluation_navigation())
                            if isinstance(metric_grid, TransferReference)
                            else None
                        )
                        forbidden = any(
                            (g["goal_kind"] == "answer" for g in gt["goals"])
                        ) and name in {"pick", "place", "open", "close", "reset"}
                        resolved = resolve_action(action, cache, ledger)
                        terminated = truncated = False
                        action_started = time.monotonic()
                        if forbidden:
                            info = {
                                "success": False,
                                "reason": "question_action_not_allowed",
                            }
                        elif resolved is None:
                            info = {
                                "success": False,
                                "reason": "use_an_introduced_object_name",
                            }
                        else:
                            _, _, terminated, truncated, info = env.step(resolved)
                        action_elapsed = time.monotonic() - action_started
                        action_seconds += action_elapsed
                        reset_executed = (
                            name == "reset"
                            and (not forbidden)
                            and (resolved is not None)
                        )
                        if reset_executed:
                            resets += 1
                        state = actual_state(env, cache)
                        event = {
                            "task_id": task["task_id"],
                            "task_step": step,
                            "action": action,
                            "before": pose,
                            "after": deepcopy(env.get_agent_pose()),
                            "result": _jsonable(info),
                            "action_source": response.get("action_source", "agent"),
                            "observation_frame_id": runner._response_frame_id,
                            "request_seconds": request_elapsed,
                            "action_seconds": action_elapsed,
                            "state_changes": state_changes(action_before, state),
                        }
                        if collision_before is not None:
                            event.update(
                                transfer_collision_before_sha256=collision_before,
                                transfer_collision_after_sha256=collision_digest(
                                    env.evaluation_navigation()
                                ),
                            )
                        actions.append(event)
                        poses.append(event["after"])
                        event["event_id"] = (
                            f"{session.session_id}:{events.counter + 1:06d}"
                        )
                        append(session_file(output, "action_changelog.jsonl"), event)
                        events.emit(
                            "reset" if reset_executed else "step",
                            sample=True,
                            instruction=task["instruction"],
                            step=step,
                            reset_count=resets,
                            **{k: v for k, v in event.items() if k != "event_id"},
                        )
                        evaluated = evaluate_request(
                            gt, state, poses, actions, observations, answer
                        )
                        if evaluated["success"]:
                            stop_reason = "success"
                            break
                        if response.get("done") or terminated or truncated:
                            stop_reason = (
                                "agent_done"
                                if response.get("done")
                                else "environment_truncated"
                                if truncated
                                else "environment_terminated"
                            )
                            break
                        if runner.step_delay:
                            runner._pause()
                    runner._end_observations()
                    after = actual_state(env, cache)
                    evaluated = evaluate_request(
                        gt, after, poses, actions, observations, answer
                    )
                    result = {
                        **base,
                        "scored": True,
                        "dispatched": True,
                        "success": evaluated["success"]
                        and agent_status != "unsupported",
                        "reason_code": stop_reason,
                        "agent_status": agent_status,
                        "steps": len(actions),
                        "prediction": answer,
                        "goal_results": evaluated["goals"],
                    }
                    result["task_timing"] = {
                        "started_monotonic_s": task_started,
                        "finished_monotonic_s": time.monotonic(),
                        "request_count": request_count,
                        "agent_wait_seconds": request_seconds,
                        "action_seconds": action_seconds,
                    }
                    result["task_timing"]["elapsed_seconds"] = (
                        result["task_timing"]["finished_monotonic_s"] - task_started
                    )
                    result["evaluation"] = evaluate_metrics(
                        gt,
                        before,
                        after,
                        poses,
                        actions,
                        result,
                        reference=metric_reference,
                        grid=metric_grid,
                        timing=result["task_timing"],
                    )
                    evidence = {e["goal_id"]: e for e in evaluated["goals"]}
                    new = execution_events_v3(
                        session.session_id,
                        task,
                        gt,
                        before,
                        after,
                        dispatched=True,
                        success=result["success"],
                        goal_evidence=evidence,
                    )
                if repair["status"] != "ready":
                    result["evaluation"] = evaluate_metrics(
                        None, before, after, [], [], result, reference=metric_reference
                    )
                history.extend(new)
                for event in new:
                    append(
                        session_file(output, "runtime_history.jsonl"),
                        compact_history(event),
                    )
                results.append(compact_outcome(result))
                append(
                    session_file(output, "request_outcomes.jsonl"),
                    {
                        **compact_outcome(result),
                        "state_changes": state_changes(before, after),
                    },
                )
                pending = None
                events.emit(
                    "task_complete",
                    task_id=public["task_id"],
                    instruction=(repair.get("public_task") or public)["instruction"],
                    success=result["success"],
                    reason_code=result.get("reason_code"),
                    reset_count=resets,
                )
                totals = request_totals(results)
                write(
                    session_file(output, "metrics.json"),
                    {
                        "schema_version": 1,
                        "scene_id": scene_id,
                        "session_id": session.session_id,
                        "run_status": "running",
                        **totals,
                        "reset_count": resets,
                        "task_results": results,
                    },
                )
            events.emit("session_complete", reason_code="completed")
            events.close("completed")
            events = None
            write(
                session_file(output, "metrics.json"),
                {
                    "schema_version": 1,
                    "scene_id": scene_id,
                    "session_id": session.session_id,
                    "run_status": "completed",
                    **totals,
                    "reset_count": resets,
                    "task_results": results,
                },
            )
            write(session_file(output, "name_ledger.json"), ledger)
            runner._agent_io.close()
            runner._agent_io = None
    except BaseException as exc:
        run_error = exc
        if output:
            append(
                session_file(output, "integration_errors.jsonl"),
                {"error": str(exc), "type": type(exc).__name__},
            )
            if events:
                try:
                    events.emit(
                        "integration_error",
                        reason_code=type(exc).__name__,
                        error=str(exc),
                    )
                except Exception as event_error:
                    append(
                        session_file(output, "integration_errors.jsonl"),
                        {"error": str(event_error), "phase": "error_event"},
                    )
            if pending:
                session, task, gt, before, record = pending
                try:
                    after = actual_state(env, cache)
                    append(
                        session_file(output, "request_outcomes.jsonl"),
                        {
                            "task_id": task["task_id"],
                            "interrupted": True,
                            "success": None,
                            "reason_code": "integration_error",
                            "state_changes": state_changes(before, after),
                        },
                    )
                except Exception as history_error:
                    append(
                        session_file(output, "integration_errors.jsonl"),
                        {"error": str(history_error), "phase": "interrupted_state"},
                    )
            write(
                session_file(output, "metrics.json"),
                {
                    "schema_version": 1,
                    "scene_id": scene_id,
                    "run_status": "error",
                    "evaluation_valid": False,
                    "task_score": None,
                    "error": str(exc),
                    "reset_count": resets,
                    "task_results": results,
                },
            )
        raise
    finally:
        cleanup_errors = []
        for cleanup in (
            runner._end_observations,
            lambda: (
                events.close("interrupted" if run_error else "completed")
                if events
                else None
            ),
            lambda: runner._agent_io.close() if runner._agent_io else None,
            lambda: env.close() if env else None,
            lambda: write_request_audit(runner.output_root) if output else None,
        ):
            try:
                cleanup()
            except Exception as exc:
                cleanup_errors.append(exc)
                if output:
                    append(
                        session_file(output, "integration_errors.jsonl"),
                        {"error": str(exc), "phase": "cleanup"},
                    )
        runner._agent_io = None
        runner._current_output = runner._current_session_id = None
        if cleanup_errors and (not run_error):
            write(
                session_file(output, "metrics.json"),
                {
                    "run_status": "error",
                    "evaluation_valid": False,
                    "task_score": None,
                    "error": str(cleanup_errors[0]),
                    "task_results": results,
                },
            )
            raise cleanup_errors[0]
