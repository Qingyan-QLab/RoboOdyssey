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
from ..scenes.dataset_loader import SessionRecord
from ..scenes.state_composer import apply_changes, compose_session_state
from ..evaluation.success import evaluate_embodied_task, evaluate_qa
from .instruction import plan_room_trajectory, session_start_pose
from .reference import resolve_runtime_reference


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
    ):
        self.output_root = Path(output_root)
        self.agent = agent
        self.max_steps = int(max_steps)
        self.qa_judge = qa_judge
        self.step_delay = max(0.0, float(step_delay))
        self.wait_callback = wait_callback
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
        score = sum(bool(item.get("success")) for item in tasks) / len(tasks) if tasks else 0.0
        checkpoint_tasks = list(tasks)
        if status == "interrupted" and self._current_task is not None and not any(
                item.get("task_id") == self._current_task.get("task_id") for item in checkpoint_tasks):
            checkpoint_tasks.append({"task_id": self._current_task.get("task_id"),
                                     "task_type": self._current_task.get("task_type"),
                                     "task_family": self._current_task.get("task_family"),
                                     "capabilities": self._current_task.get("capabilities", []),
                                     "success": False, "interrupted": True,
                                     "metrics": {"reason": "benchmark_interrupted"}})
        value = {"schema_version": 1, "session_id": self._current_session_id,
                 "run_status": status, "success": status == "completed" and bool(tasks) and all(
                     bool(item.get("success")) for item in tasks),
                 "task_score": score, "reset_count": self._current_reset_count,
                 "reset_aware_score": score / (1 + self._current_reset_count),
                 "completed_tasks": [item.get("task_id") for item in tasks],
                 "task_results": checkpoint_tasks}
        if status == "interrupted" and self._current_task_id is not None:
            value["failed_task"] = self._current_task_id
        if error:
            value["error"] = error
        (self._current_output / "metrics.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self.agent.request(payload, on_wait=self.wait_callback)
        except Exception as exc:
            if self._agent_io is not None:
                self._agent_io.write(json.dumps({
                    "timestamp": time.time(), "request_type": payload.get("type"),
                    "task_id": payload.get("task_id"), "direction": "error",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }, ensure_ascii=False) + "\n")
                self._agent_io.flush()
            raise
        if self._agent_io is not None:
            self._agent_io.write(json.dumps({
                "timestamp": time.time(), "request_type": payload.get("type"),
                "task_id": payload.get("task_id"), "direction": "request",
                "payload": _audit_jsonable(payload),
            }, ensure_ascii=False) + "\n")
            self._agent_io.write(json.dumps({
                "timestamp": time.time(), "request_type": payload.get("type"),
                "task_id": payload.get("task_id"), "direction": "response",
                "payload": _audit_jsonable(response),
            }, ensure_ascii=False) + "\n")
            self._agent_io.flush()
        return response

    def _pause(self, seconds: float | None = None) -> None:
        deadline = time.monotonic() + (self.step_delay if seconds is None else max(0.0, float(seconds)))
        while time.monotonic() < deadline:
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
        response = self._request({
            "type": "qa", "scene_id": scene_id, "session_id": session.session_id,
            "task_id": task["task_id"], "task_type": "qa", "question": task["question"],
            "answer_type": task.get("answer_type", "free_text"),
            "capabilities": task.get("capabilities", []), "observation": _jsonable(env.observe()),
            "memory": memory,
        })
        _merge_memory(memory, response)
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
                "prediction": prediction, "metrics": metrics}

    def _run_embodied(
        self, scene_id: str, session: SessionRecord, task: dict[str, Any],
        private: dict[str, Any], memory: dict[str, Any], env: Any, event_index: int,
        reset_count: int,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
        env.set_task(task)
        events: list[dict[str, Any]] = []
        previous_action = previous_result = None
        for task_step in range(self.max_steps):
            if self.reset_requested_callback and self.reset_requested_callback():
                response = {"action": {"action": "reset"}, "source": "viewer_button"}
            else:
                response = self._request({
                    "type": "step", "scene_id": scene_id, "session_id": session.session_id,
                    "task_id": task["task_id"], "task_type": task["task_type"],
                    "task_family": task["task_family"], "task_step": task_step,
                    "session_instruction": session.metadata.get("instruction"),
                    "instruction": task["instruction"], "observation": _jsonable(env.observe()),
                    "memory": memory, "previous_action": _jsonable(previous_action),
                    "previous_result": _jsonable(previous_result),
                })
            _merge_memory(memory, response)
            action = response["action"]
            before = env.get_agent_pose()
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
                result["termination_reason"] = "benchmark_success"
                break
            if response.get("done"):
                result["termination_reason"] = "agent_done_before_success"
                break
            if terminated or truncated:
                result["termination_reason"] = "environment_terminated"
                break
            previous_action, previous_result = action, info
        result = evaluate_embodied_task(task, private, events, env.runtime_state())
        result.update(task_id=task["task_id"], task_type=task["task_type"],
                      task_family=task["task_family"], capabilities=task.get("capabilities", []))
        return result, events, reset_count

    def run_scene(
        self, scene_id: str, sessions: list[SessionRecord], canonical_state: dict[str, Any],
        env_factory: Callable[[str, dict[str, Any], str], Any],
    ) -> None:
        state = deepcopy(canonical_state)
        memory: dict[str, Any] = {}
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
                    self._current_task_id = str(task["task_id"])
                    self._current_task = task
                    private = resolve_runtime_reference(
                        task, private_by_id[str(task["task_id"])], env)
                    if task["task_type"] == "qa":
                        result = self._run_qa(scene_id, session, task, private, memory, env)
                    else:
                        result, events, reset_count = self._run_embodied(
                            scene_id, session, task, private, memory, env, len(all_events), reset_count)
                        all_events.extend(events)
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
                state = apply_changes(state, state_changes)
                (output / "action_changelog.jsonl").write_text(
                    "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in all_events), encoding="utf-8")
                (output / "state_changelog.json").write_text(
                    json.dumps(state_changes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                score = sum(bool(item["success"]) for item in task_results) / len(task_results)
                metrics = {"schema_version": 1, "scene_id": scene_id, "session_id": session.session_id,
                           "run_status": "completed",
                           "success": all(bool(item["success"]) for item in task_results),
                           "task_score": score, "reset_count": reset_count,
                           "reset_aware_score": score / (1 + reset_count),
                           "task_results": task_results}
                (output / "metrics.json").write_text(
                    json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                if self._agent_io is not None:
                    self._agent_io.close()
                    self._agent_io = None
                self._log(f"session_complete scene={scene_id} session={session.session_id} "
                          f"success={metrics['success']}")
        finally:
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
