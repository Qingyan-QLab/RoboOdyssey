"""Execute ordered benchmark tasks and persist auditable runtime logs."""
from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable

from ..agents.agent_api import AgentProcess
from ..agents.protocol import PROTOCOL_VERSION, OBSERVATION_STREAM
from ..scenes.dataset_loader import SessionRecord
from ..scenes.state_composer import apply_changes, compose_session_state
from ..evaluation.success import evaluate_embodied_task, evaluate_qa, already_satisfied
from ..evaluation.report import task_totals
from .instruction import plan_room_trajectory, session_start_pose
from .reference import resolve_runtime_reference
from .observations import ObservationClock, capture_observation


def _jsonable(value: Any) -> Any:
    if hasattr(value, "dtype") and hasattr(value, "shape") and hasattr(value, "tobytes"):
        return {"encoding": "base64", "dtype": str(value.dtype), "shape": list(value.shape),
                "data": base64.b64encode(value.tobytes()).decode("ascii")}
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _merge_memory(memory: dict[str, Any], response: dict[str, Any]) -> None:
    update = response.get("memory_update")
    if update is not None:
        if not isinstance(update, dict):
            raise ValueError("memory_update must be an object")
        memory.update(update)


def _runtime_instruction_rooms(session_instruction: dict[str, Any], env: Any) -> list[dict[str, Any]]:
    """Resolve compact room render hints into runtime-only camera trajectories."""
    rooms = session_instruction.get("rooms", [])
    if not isinstance(rooms, list):
        return []
    resolved = []
    for item in rooms:
        if not isinstance(item, dict):
            continue
        value = dict(item)
        region = value.get("render_region") or {}
        if region.get("type") == "room" and value.get("room_id"):
            trajectory = plan_room_trajectory(env, str(value["room_id"]))
            if trajectory is None:
                raise ValueError(
                    f"instruction room {value['room_id']!r} has no walkable runtime trajectory"
                )
            value["camera_trajectory"] = trajectory
        resolved.append(value)
    return resolved


def _audit_jsonable(value: Any) -> Any:
    """Make an agent audit record useful without duplicating RGB base64."""
    if isinstance(value, dict) and value.get("encoding") == "base64" and {
        "dtype", "shape", "data"
    } <= value.keys():
        raw = base64.b64decode(value["data"])
        return {"kind": "array", "dtype": str(value["dtype"]),
                "shape": list(value["shape"]), "sha256": hashlib.sha256(raw).hexdigest()}
    if hasattr(value, "dtype") and hasattr(value, "shape") and hasattr(value, "tobytes"):
        raw = value.tobytes()
        return {"kind": "array", "dtype": str(value.dtype), "shape": list(value.shape),
                "sha256": hashlib.sha256(raw).hexdigest()}
    if isinstance(value, dict):
        if "images" in value and isinstance(value["images"], dict):
            copied = {key: _audit_jsonable(item) for key, item in value.items() if key != "images"}
            copied["images"] = {str(name): _audit_jsonable(image)
                                for name, image in value["images"].items()}
            return copied
        return {str(key): _audit_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_audit_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


class SessionRunner:
    def __init__(
        self, output_root: str | Path, agent: AgentProcess, max_steps: int = 2000,
        qa_judge: Any | None = None, step_delay: float = 0.0, wait_callback=None,
        viewer_callback=None, transition_callback=None, reset_requested_callback=None,
        viewer_sync_callback=None, log_callback=None, session_start_pose_mode: str = "inherit",
        task_ids: set[str] | None = None, task_types: set[str] | None = None,
        video_options: dict | None = None, observation_clock: ObservationClock | None = None,
    ):
        self.output_root = Path(output_root)
        self.agent = agent
        self.max_steps = int(max_steps)
        if self.max_steps < 1:
            raise ValueError('max_steps must be positive')
        self.video_options = video_options
        self.qa_judge = qa_judge
        self.step_delay = max(0.0, float(step_delay))
        self.wait_callback = wait_callback
        self.observations = observation_clock or ObservationClock()
        self.observations.on_event = self._audit_observation
        self._observation_task_id = None
        self._observation_context = None
        self._public_observation = lambda value: value
        self._stream_evidence = []
        self._sent_frame_ids = set()
        self._request_number = 0
        self.viewer_callback = viewer_callback
        self.transition_callback = transition_callback
        self.reset_requested_callback = reset_requested_callback
        self.viewer_sync_callback = viewer_sync_callback
        self.log_callback = log_callback
        self.task_ids = task_ids
        self.task_types = task_types
        if session_start_pose_mode not in {"inherit", "explicit"}:
            raise ValueError("session_start_pose_mode must be 'inherit' or 'explicit'")
        self.session_start_pose_mode = session_start_pose_mode
        self._agent_io = None
        self._current_output: Path | None = None
        self._current_task_results: list[dict[str, Any]] = []
        self._current_events: list[dict[str, Any]] = []
        self._current_reset_count = 0
        self._current_session_id: str | None = None
        self._current_task_id: str | None = None
        self._current_task: dict[str, Any] | None = None

    def _write_checkpoint(self, *, status: str = "running", error: str | None = None) -> None:
        if self._current_output is None or self._current_session_id is None:
            return
        tasks = self._current_task_results
        checkpoint_tasks = list(tasks)
        if status == "interrupted" and self._current_task is not None and not any(
                item.get("task_id") == self._current_task.get("task_id") for item in checkpoint_tasks):
            checkpoint_tasks.append({"task_id": self._current_task.get("task_id"),
                                     "task_type": self._current_task.get("task_type"),
                                     "task_family": self._current_task.get("task_family"),
                                     "capabilities": self._current_task.get("capabilities", []),
                                     "success": False, "interrupted": True,
                                     "metrics": {"reason": "benchmark_interrupted"}})
        totals = task_totals(checkpoint_tasks)
        score = totals['task_score']
        value = {"schema_version": 1, "session_id": self._current_session_id,
                 "run_status": status, **totals, "reset_count": self._current_reset_count,
                 "reset_aware_score": score / (1 + self._current_reset_count) if score is not None else None,
                 "completed_tasks": [item.get("task_id") for item in tasks],
                 "task_results": checkpoint_tasks}
        if status == "interrupted" and self._current_task_id is not None:
            value["failed_task"] = self._current_task_id
        if error:
            value["error"] = error
        (self._current_output / "metrics.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _audit_observation(self, event, **values):
        if self._agent_io is not None:
            self._agent_io.write(json.dumps({
                'timestamp': time.time(), 'direction': 'observation', 'request_type': 'observation', 'event': event,
                'task_id': self._observation_task_id, **values,
            }, ensure_ascii=False) + '\n')
            self._agent_io.flush()

    def _begin_observations(self, task_id, source, public_observation=None):
        self.observations.set_source(None)
        self._observation_task_id = task_id
        self._observation_context = None
        self._public_observation = public_observation or (lambda value: value)
        self._stream_evidence = []
        self._sent_frame_ids = set()
        self.observations.set_source(source)

    def _end_observations(self):
        self.observations.set_source(None)
        self._observation_task_id = None
        self._observation_context = None
        self._stream_evidence = []

    def _push_observation(self):
        sample = self.observations.poll()
        if sample is None:
            return
        if self._observation_context is None:
            raise RuntimeError('observation stream has no active task context')
        observation, evidence = sample
        timing = observation['timing']
        timing['dispatched_monotonic_s'] = self.observations.clock()
        timing['age_at_dispatch_seconds'] = timing['dispatched_monotonic_s'] - timing['captured_monotonic_s']
        payload = {**self._observation_context, 'type': 'observation',
                   'observation': _jsonable(self._public_observation(observation))}
        self._audit_observation('send_attempt', payload=_audit_jsonable(payload))
        # Never queue/replace camera frames on the benchmark side. A blocked
        # receiver is an integration error, not a silently dropped observation.
        self.agent.send_observation(payload)
        self._audit_observation('sent', frame_id=timing['frame_id'],
                                sent_monotonic_s=self.observations.clock())
        self._sent_frame_ids.add(timing['frame_id'])
        self._stream_evidence.append((timing['frame_id'], evidence))

    def _consumed_stream_evidence(self):
        evidence = [item for frame, items in self._stream_evidence
                    if frame <= self._response_frame_id for item in items]
        self._stream_evidence = [(frame, items) for frame, items in self._stream_evidence
                                 if frame > self._response_frame_id]
        return evidence

    def _wait_for_agent(self):
        if self.wait_callback:
            self.wait_callback()
        self._push_observation()

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._request_number += 1
        payload = dict(payload, request_id=str(self._request_number), protocol_version=PROTOCOL_VERSION)
        if payload.get('type') == 'session_start':
            payload['observation_stream'] = dict(OBSERVATION_STREAM)
        def audit(direction, **values):
            if self._agent_io is not None:
                self._agent_io.write(json.dumps({
                    "timestamp": time.time(), "request_type": payload.get("type"),
                    "task_id": payload.get("task_id"), "direction": direction, **values,
                }, ensure_ascii=False) + "\n")
                self._agent_io.flush()
        # Keep attempted dispatches even if the server times out or returns an
        # integration error. The response is logged before status handling.
        timing = (payload.get('observation') or {}).get('timing')
        if timing is not None:
            self._observation_context = {key: payload[key] for key in (
                'scene_id', 'session_id', 'task_id', 'request_id', 'protocol_version', 'sequence_context') if key in payload}
            self._sent_frame_ids.add(timing['frame_id'])
            timing['dispatched_monotonic_s'] = self.observations.clock()
            timing['age_at_dispatch_seconds'] = timing['dispatched_monotonic_s'] - timing['captured_monotonic_s']
        audit('request', payload=_audit_jsonable(payload))
        try:
            response = self.agent.request(payload, on_wait=self._wait_for_agent)
            audit('response', payload=_audit_jsonable(response))
            if response.get('agent_status') == 'integration_error':
                raise RuntimeError('agent integration error: ' + str(response.get('error') or response.get('reason_code')))
            if timing is not None:
                used = response.get('observation_frame_id', timing['frame_id'])
                if type(used) is not int or used not in self._sent_frame_ids or used < timing['frame_id']:
                    raise ValueError('response observation_frame_id is not a sent frame for this request/task')
                self._response_frame_id = used
        except Exception as exc:
            audit('error', error={"type": type(exc).__name__, "message": str(exc)})
            raise
        return response

    def _pause(self, seconds: float | None = None) -> None:
        deadline = time.monotonic() + (self.step_delay if seconds is None else max(0.0, float(seconds)))
        while time.monotonic() < deadline:
            self._push_observation()
            # The environment already advances Kit once in ``env.step`` and
            # AgentProcess polling advances it while waiting for the next
            # model response.  Calling app.update repeatedly during the visual
            # delay can hit an uninitialized viewer backbuffer on Kit 6.x.
            time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))

    def _show(self, **values: Any) -> None:
        if self.viewer_callback is not None:
            self.viewer_callback(**values)

    def _log(self, message: str) -> None:
        if self.log_callback is not None:
            self.log_callback(message)

    def _run_qa(
        self, scene_id: str, session: SessionRecord, task: dict[str, Any],
        private: dict[str, Any], memory: dict[str, Any], env: Any,
    ) -> dict[str, Any]:
        self._begin_observations(task['task_id'], lambda: capture_observation(env))
        payload = {
            "type": "qa", "scene_id": scene_id, "session_id": session.session_id,
            "task_id": task["task_id"], "task_type": "qa", "question": task["question"],
            "answer_type": task.get("answer_type", "free_text"),
            "capabilities": task.get("capabilities", []),
            **({'current_time': task['issued_at']} if task.get('issued_at') else {}),
            "memory": memory,
        }
        inspect = private.get('state_assumption') == 'actual_task_start_state' and 'navigation' in task.get('capabilities', [])
        actions = []
        self._current_events = actions
        if inspect:
            env.set_task(task)
            payload['allow_navigation_actions'] = True
        for step in range(self.max_steps if inspect else 1):
            observation, _ = self.observations.take(self.wait_callback)
            payload['observation'] = _jsonable(observation)
            response = self._request(payload)
            _merge_memory(memory, response)
            if 'action' not in response:
                break
            if not inspect:
                raise ValueError('historical recall does not allow inspection actions')
            action = response['action']
            name = action if isinstance(action, str) else action.get('action')
            if name in {'pick', 'place', 'open', 'close'}:
                raise ValueError('inspection QA cannot change the world')
            before = env.get_agent_pose()
            _, _, terminated, truncated, info = env.step(action)
            event = {'task_id': task['task_id'], 'task_step': step, 'action': _jsonable(action),
                     'before': _jsonable(before), 'after': _jsonable(env.get_agent_pose()), 'result': _jsonable(info)}
            actions.append(event)
            if self._current_output:
                with (self._current_output / 'action_changelog.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + '\n')
            if self.viewer_sync_callback:
                self.viewer_sync_callback(env)
            payload.update(previous_action=action, previous_result=_jsonable(info))
            if terminated or truncated or response.get('done'):
                break
        self._end_observations()
        prediction = response.get("answer", response.get("prediction", ""))
        metrics = evaluate_qa(prediction, private)
        outcome = private.get("expected_outcome", {})
        if self.qa_judge is not None and task.get("answer_type") in {"free_text", "open_ended"}:
            judged = self.qa_judge.judge(task["question"], str(outcome.get("answer", "")), str(prediction))
            metrics["llm_judge"] = judged
            metrics["success"] = bool(judged.get("score", 0) >= 1)
        self._show(scene_id=scene_id, session_id=session.session_id, task_id=task["task_id"],
                   task_type="qa", question=task["question"], answer=prediction, event="qa")
        return {"task_id": task["task_id"], "task_type": "qa", "task_family": task["task_family"],
                "capabilities": task.get("capabilities", []), "success": metrics["success"],
                "prediction": prediction, "metrics": metrics, '_inspection_events': actions}

    def _run_embodied(
        self, scene_id: str, session: SessionRecord, task: dict[str, Any],
        private: dict[str, Any], memory: dict[str, Any], env: Any, event_index: int,
        reset_count: int,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
        env.set_task(task)
        self._begin_observations(task['task_id'], lambda: capture_observation(env))
        events: list[dict[str, Any]] = []
        self._current_events = events
        previous_action = previous_result = None
        termination_reason = 'step_limit_reached'
        for task_step in range(self.max_steps):
            if self.reset_requested_callback and self.reset_requested_callback():
                response = {"action": {"action": "reset"}, "source": "viewer_button"}
            else:
                observation, _ = self.observations.take(self.wait_callback)
                response = self._request({
                    "type": "step", "scene_id": scene_id, "session_id": session.session_id,
                    "task_id": task["task_id"], "task_type": task["task_type"],
                    "task_family": task["task_family"], "task_step": task_step,
                    "session_instruction": session.metadata.get("instruction"),
                    "instruction": task["instruction"], "observation": _jsonable(observation),
                    **({'current_time': task['issued_at']} if task.get('issued_at') else {}),
                    "memory": memory, "previous_action": _jsonable(previous_action),
                    "previous_result": _jsonable(previous_result),
                })
            _merge_memory(memory, response)
            action = response["action"]
            before = env.get_agent_pose()
            action_name = action if isinstance(action, str) else action.get('action')
            action_state = deepcopy(env.runtime_state()) if action_name in {'pick', 'place', 'open', 'close'} else None
            _, _, terminated, truncated, info = env.step(action)
            if self.viewer_sync_callback is not None:
                self.viewer_sync_callback(env)
            if (action if isinstance(action, str) else action.get("action")) == "reset":
                reset_count += 1
                info["reset_count"] = reset_count
            after = env.get_agent_pose()
            event = {"index": event_index + len(events), "task_id": task["task_id"],
                     "task_step": task_step, "action": _jsonable(action), "before": _jsonable(before),
                     "after": _jsonable(after), "result": _jsonable(info),
                     "action_source": response.get("action_source") or
                     (action.get("source") if isinstance(action, dict) else None) or "agent",
                     "agent_requested": bool(response.get("agent_requested", True))}
            if action_state is not None and info.get('object_id'):
                oid = str(info['object_id'])
                event['object_before'] = deepcopy(action_state.get('objects', {}).get(oid))
                event['object_after'] = deepcopy(env.runtime_state().get('objects', {}).get(oid))
            events.append(event)
            if task_step == 0 or (task_step + 1) % 10 == 0:
                action_name = action if isinstance(action, str) else action.get("action")
                self._log(
                    f"task_progress scene={scene_id} session={session.session_id} "
                    f"task={task['task_id']} step={task_step + 1}/{self.max_steps} "
                    f"action={action_name} success={info.get('success')} "
                    f"reason={info.get('reason', '')}"
                )
            if self._current_output is not None:
                with (self._current_output / "action_changelog.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._show(scene_id=scene_id, session_id=session.session_id,
                       task_id=task["task_id"], task_type=task["task_type"],
                       instruction=task["instruction"], task=task["instruction"], task_step=task_step,
                       action=action, result=info, before=before, after=after,
                       reset_count=reset_count, event="step")
            if self.step_delay:
                self._pause()
            result = evaluate_embodied_task(task, private, events, env.runtime_state())
            if result["success"]:
                termination_reason = "benchmark_success"
                break
            if response.get("done"):
                termination_reason = "agent_done_before_success"
                break
            if terminated or truncated:
                termination_reason = "environment_terminated"
                break
            previous_action, previous_result = action, info
        self._end_observations()
        result = evaluate_embodied_task(task, private, events, env.runtime_state())
        result.update(task_id=task["task_id"], task_type=task["task_type"],
                      task_family=task["task_family"], capabilities=task.get("capabilities", []),
                      termination_reason=termination_reason)
        return result, events, reset_count

    def run_scene(
        self, scene_id: str, sessions: list[SessionRecord], canonical_state: dict[str, Any],
        env_factory: Callable[[str, dict[str, Any], str], Any],
    ) -> None:
        if sessions and sessions[0].metadata.get('schema_version') == 5:
            from .request_runner import run_requests
            return run_requests(self, scene_id, sessions, canonical_state, env_factory)
        state = deepcopy(canonical_state)
        memory: dict[str, Any] = {}
        execution_history = []
        pending_history = None
        env = None
        try:
            for session in sessions:
                self._log(f"session_start scene={scene_id} session={session.session_id}")
                if self.transition_callback:
                    self.transition_callback()
                mutation = session.mutation()
                state = compose_session_state(state, mutation)
                if env is None:
                    env = env_factory(scene_id, state, session.session_id)
                else:
                    if hasattr(env, "apply_world_state"):
                        env.apply_world_state(state, preserve_runtime=True)
                    else:
                        env.apply_state(state)
                session_initial_state = deepcopy(state)
                session_instruction = session.metadata.get("instruction") or {}
                # Ground truth remains private to the runner.  New packages
                # normally provide an explicit start_pose; when they do not,
                # use the first reference point of the session's first task as
                # the reset/task start, never an instruction camera waypoint.
                gt = session.ground_truth()
                runtime_contract = session.runtime_contract()
                templates = {}
                if runtime_contract:
                    from ..legacy.task_repair import instantiate_task
                    from ..legacy.task_history import execution_events
                    from ..reporting.legacy import write_review
                    template_file, scene_cache = runtime_contract
                    templates = {t['task_id']: t for t in template_file['tasks']}
                tasks = [task for task in session.tasks
                         if (self.task_ids is None or str(task["task_id"]) in self.task_ids)
                         and (self.task_types is None or str(task["task_type"]) in self.task_types)]
                if not tasks:
                    self._log(f"session_skipped scene={scene_id} session={session.session_id} reason=no_selected_tasks")
                    continue
                explicit_pose = session_start_pose(session, gt, env)
                use_inherited_pose = (env is not None and self.session_start_pose_mode == "inherit"
                                      and session.metadata.get("start_pose_policy", "inherit") == "inherit"
                                      and self._current_session_id is not None)
                if use_inherited_pose:
                    start_pose = env.get_agent_pose()
                else:
                    start_pose = explicit_pose
                if hasattr(env, "begin_session"):
                    requested_start_pose = deepcopy(start_pose)
                    env.begin_session(start_pose, apply_pose=not use_inherited_pose)
                    start_pose = env.get_agent_pose()
                    requested_position = (requested_start_pose or {}).get("position")
                    if requested_position and start_pose.get("position") != requested_position:
                        self._log(
                            f"session_start_pose_adjusted scene={scene_id} "
                            f"session={session.session_id} requested={requested_position} "
                            f"actual={start_pose.get('position')}"
                        )
                elif start_pose is not None:
                    env.set_agent_pose(start_pose, update_reset_pose=True)
                output = self.output_root / "scenes" / scene_id / "sessions" / session.session_id
                output.mkdir(parents=True, exist_ok=True)
                if runtime_contract:
                    (output / 'runtime_tasks.jsonl').write_text('', encoding='utf-8')
                    (output / 'runtime_history.jsonl').write_text('', encoding='utf-8')
                    if mutation.get('changes'):
                        execution_history.append({'session_id': session.session_id, 'kind': 'hidden_change',
                                                  'agent_information': 'hidden', 'mutation': deepcopy(mutation)})
                self._current_output = output
                self._current_task_results = []
                self._current_events = []
                self._current_reset_count = 0
                self._current_session_id = session.session_id
                private_by_id = {str(item["task_id"]): item for item in gt["tasks"]}
                task_ids = [str(task["task_id"]) for task in tasks]
                if not set(task_ids).issubset(private_by_id):
                    raise ValueError(f"task/ground-truth mismatch in {session.session_id}")

                (output / "session_meta.json").write_text(json.dumps({
                    "schema_version": 1,
                    "scene_id": scene_id,
                    "session_id": session.session_id,
                    "parent_session_id": session.metadata.get("parent_session_id"),
                    "instruction": session.metadata.get("instruction"),
                    "start_pose": start_pose,
                    "task_order": [{"task_id": task["task_id"],
                                    "task_type": task["task_type"],
                                    "task_family": task.get("task_family"),
                                    "capabilities": task.get("capabilities", []),
                                    "instruction": task.get("instruction"),
                                    "question": task.get("question"),
                                    "depends_on": task.get("depends_on", [])}
                                   for task in tasks],
                    "mutation": mutation,
                    "initial_state": session_initial_state,
                }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

                self._agent_io = (output / "agent_io.jsonl").open("w", encoding="utf-8")
                self._agent_io.write(json.dumps({"schema_version": 1,
                                                 "scene_id": scene_id,
                                                 "session_id": session.session_id},
                                                ensure_ascii=False) + "\n")

                # Announce the compact session instruction before the first
                # room frame.  This initializes the Commander scene-memory
                # area; individual session_view events then update the main
                # body with the room currently being rendered.
                self._show(
                    scene_id=scene_id,
                    session_id=session.session_id,
                    session_instruction=session_instruction,
                    session_start_pose=start_pose,
                    task_summary=", ".join(
                        f"{task['task_id']} ({task['task_type']})" for task in tasks
                    ),
                    event="session_start",
                )

                # Plan and replay room trajectories at runtime.  Each rendered
                # observation plus its room text becomes initial visual memory;
                # the task starts only after returning to start_pose via reset.
                session_views = []
                room_instructions = (_runtime_instruction_rooms(session_instruction, env)
                                     if isinstance(session_instruction, dict) else [])
                frame_index = 0
                for room_instruction in room_instructions:
                    trajectory = room_instruction.get("camera_trajectory", {})
                    poses = trajectory.get("poses", []) if isinstance(trajectory, dict) else []
                    frame_interval = float(trajectory.get(
                        "frame_interval_s",
                        getattr(getattr(env, "config", None), "step_seconds", 1 / 30),
                    )) if isinstance(trajectory, dict) else float(
                        getattr(getattr(env, "config", None), "step_seconds", 1 / 30)
                    )
                    for pose in poses:
                        if not pose:
                            continue
                        env.set_agent_pose(pose, update_reset_pose=False)
                        if self.viewer_sync_callback is not None:
                            self.viewer_sync_callback(env)
                        if hasattr(env, "render_once"):
                            env.render_once()
                        observation = env.observe()
                        session_views.append({"text": room_instruction.get("text", ""),
                                              "room_id": room_instruction.get("room_id"),
                                              "camera_pose": pose,
                                              "trajectory_frame": frame_index,
                                              "observation": _jsonable(observation)})
                        self._show(scene_id=scene_id, session_id=session.session_id,
                                   session_instruction=session_instruction,
                                   view_index=frame_index,
                                   view_text=room_instruction.get("text", ""),
                                   view_observation=observation, event="session_view")
                        frame_index += 1
                        self._pause(frame_interval)
                if session_views:
                    env.reset()
                    if self.viewer_sync_callback is not None:
                        self.viewer_sync_callback(env)
                    self._show(scene_id=scene_id, session_id=session.session_id,
                               session_instruction=session_instruction,
                               task_summary=", ".join(f"{task['task_id']} ({task['task_type']})"
                                                       for task in tasks),
                               event="session_views_complete")

                start_response = self._request({
                    "type": "session_start", "scene_id": scene_id, "session_id": session.session_id,
                    "instruction": session.metadata.get("instruction"), "task_ids": task_ids,
                    **({'time_context': session.metadata['time_context']} if session.metadata.get('time_context') else {}),
                    "start_pose": start_pose,
                    "session_views": session_views,
                    "memory": memory,
                })
                _merge_memory(memory, start_response)
                instruction_poses = []
                for room_instruction in room_instructions:
                    trajectory = room_instruction.get("camera_trajectory", {})
                    instruction_poses.extend(trajectory.get("poses", [])
                                             if isinstance(trajectory, dict) else [])
                self._show(scene_id=scene_id, session_id=session.session_id,
                           instruction=session.metadata.get("instruction"),
                           session_instruction=session.metadata.get("instruction"),
                           session_start_pose=start_pose,
                           session_camera_poses=instruction_poses,
                           task_summary=", ".join(f"{task['task_id']} ({task['task_type']})" for task in tasks),
                           event="session_ready")

                all_events: list[dict[str, Any]] = []
                task_results = []
                reset_count = 0
                for task in tasks:
                    self._current_events = []
                    before = deepcopy(env.runtime_state())
                    task_start_pose = deepcopy(env.get_agent_pose())
                    repair = None
                    completion = None
                    events = []
                    if runtime_contract:
                        template = templates[task['task_id']]
                        repair = instantiate_task(template, scene_cache, before, execution_history,
                            task=task, ground_truth=private_by_id[task['task_id']],
                            completion_check=lambda truth: already_satisfied(task, truth, before, task_start_pose))
                        if repair['status'] == 'ready':
                            task = repair['public_task']
                    self._current_task_id = str(task["task_id"])
                    self._current_task = task
                    source_gt = repair['runtime_ground_truth'] if repair else private_by_id[str(task['task_id'])]
                    private = (resolve_runtime_reference(task, source_gt, env)
                               if source_gt and (not repair or repair['status'] == 'ready') else source_gt)
                    if not repair:
                        completion = already_satisfied(task, private, before, task_start_pose)
                    elif repair['status'] == 'already_satisfied':
                        completion = repair['repair_record']['completion_evidence']
                    if repair:
                        record = repair['repair_record']
                        record['runtime_ground_truth'] = private
                        # Snapshots make descriptor decisions reproducible even
                        # after an unexpected move of a non-target object.
                        record['state_before'] = before
                        record['agent_pose_before'] = task_start_pose
                        with (output / 'runtime_tasks.jsonl').open('a', encoding='utf-8') as stream:
                            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                        write_review(output)
                        if repair['status'] == 'ready':
                            pending_history = (session.session_id, template, deepcopy(task), before, output)
                    if completion:
                        result = {'task_id': task['task_id'], 'task_type': task['task_type'],
                                  'task_family': task['task_family'], 'capabilities': task.get('capabilities', []),
                                  'success': None, 'scored': False, 'dispatched': False, 'steps': 0,
                                  'termination_reason': 'already_satisfied', 'reason_code': 'already_satisfied',
                                  'completion_evidence': completion}
                    elif repair and repair['status'] != 'ready':
                        result = {'task_id': task['task_id'], 'task_type': task['task_type'],
                                  'task_family': task['task_family'], 'capabilities': task.get('capabilities', []),
                                  'success': None, 'scored': False, 'steps': 0, 'dispatched': False,
                                  'termination_reason': repair['status'],
                                  'reason_code': repair['repair_record']['reason_code'],
                                  'dependency': repair['repair_record'].get('dependency'),
                                  'reasons': repair['repair_record']['reasons']}
                    elif task["task_type"] == "qa":
                        result = self._run_qa(scene_id, session, task, private, memory, env)
                        events = result.pop('_inspection_events', [])
                        reset_count += sum((e['action'] if isinstance(e['action'], str) else e['action'].get('action')) == 'reset' for e in events)
                        all_events.extend(events)
                    else:
                        result, events, reset_count = self._run_embodied(
                            scene_id, session, task, private, memory, env, len(all_events), reset_count)
                        all_events.extend(events)
                    result.setdefault('scored', True)
                    result.setdefault('dispatched', True)
                    if repair:
                        result['repair_status'] = repair['status']
                        result['instruction_repaired'] = repair['repair_record']['changed']
                        new_events = execution_events(session.session_id, template, task, before, env.runtime_state(), result, events)
                        execution_history.extend(new_events)
                        with (output / 'runtime_history.jsonl').open('a', encoding='utf-8') as stream:
                            for item in new_events:
                                stream.write(json.dumps(item, ensure_ascii=False) + '\n')
                        pending_history = None
                    task_results.append(result)
                    self._current_task_results = task_results
                    self._current_reset_count = reset_count
                    self._write_checkpoint()
                    self._log("task_complete scene=%s session=%s task=%s success=%s" %
                              (scene_id, session.session_id, task["task_id"], result["success"]))
                    self._show(scene_id=scene_id, session_id=session.session_id,
                               task_id=task["task_id"], task_type=task["task_type"],
                               task=task.get("instruction", task.get("question", "")),
                               success=result["success"], event="task_complete", reset_count=reset_count)
                    self._current_task_id = None

                state_changes = env.state_changes(session_initial_state)
                state = deepcopy(env.runtime_state()) if runtime_contract else apply_changes(state, state_changes)
                (output / "action_changelog.jsonl").write_text(
                    "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in all_events), encoding="utf-8")
                (output / "state_changelog.json").write_text(
                    json.dumps(state_changes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                totals = task_totals(task_results)
                score = totals['task_score']
                metrics = {"schema_version": 1, "scene_id": scene_id, "session_id": session.session_id,
                           "run_status": "completed",
                           **totals, "reset_count": reset_count,
                           "reset_aware_score": score / (1 + reset_count) if score is not None else None,
                           "task_results": task_results}
                (output / "metrics.json").write_text(
                    json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                if self._agent_io is not None:
                    self._agent_io.close()
                    self._agent_io = None
                self._log(f"session_complete scene={scene_id} session={session.session_id} "
                          f"success={metrics['success']}")
        finally:
            self._end_observations()
            if pending_history is not None:
                # A timeout/interrupt after picking up still has a real partial
                # state. Preserve it without claiming the task succeeded.
                sid, template, interrupted_task, before, output = pending_history
                try:
                    after = env.runtime_state()
                except Exception as exc:
                    after = {'objects': {}}  # Missing evidence stays missing.
                    self._log(f'history_snapshot_unavailable session={sid} error={exc}')
                result = {'success': False, 'dispatched': True, 'termination_reason': 'benchmark_interrupted'}
                events = execution_events(sid, template, interrupted_task, before, after, result, self._current_events)
                with (output / 'runtime_history.jsonl').open('a', encoding='utf-8') as stream:
                    for item in events:
                        stream.write(json.dumps(item, ensure_ascii=False) + '\n')
            if self._current_output is not None and self._current_session_id is not None:
                # Preserve completed metrics; otherwise make an interrupted
                # checkpoint that can be included by the report aggregator.
                current = self._current_output / "metrics.json"
                try:
                    existing = json.loads(current.read_text(encoding="utf-8")) if current.exists() else {}
                except Exception:
                    existing = {}
                if existing.get("run_status") != "completed":
                    self._write_checkpoint(status="interrupted", error="benchmark interrupted")
            if self._agent_io is not None:
                self._agent_io.close()
                self._agent_io = None
            if env is not None:
                env.close()
            self._current_output = None
            self._current_session_id = None
            self._current_task_id = None
            self._current_task = None
