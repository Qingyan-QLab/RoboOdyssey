"""Execute ordered benchmark tasks and persist auditable runtime logs."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable

from ..agents.agent_api import AgentProcess
from ..agents.protocol import PROTOCOL_VERSION, OBSERVATION_STREAM
from ..scenes.dataset_loader import SessionRecord
from .observations import ObservationClock


def _jsonable(value: Any) -> Any:
    if (
        hasattr(value, "dtype")
        and hasattr(value, "shape")
        and hasattr(value, "tobytes")
    ):
        return {
            "encoding": "base64",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": base64.b64encode(value.tobytes()).decode("ascii"),
        }
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


def _audit_jsonable(value: Any) -> Any:
    """Make an agent audit record useful without duplicating RGB base64."""
    if (
        isinstance(value, dict)
        and value.get("encoding") == "base64"
        and {"dtype", "shape", "data"} <= value.keys()
    ):
        raw = base64.b64decode(value["data"])
        return {
            "kind": "array",
            "dtype": str(value["dtype"]),
            "shape": list(value["shape"]),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    if (
        hasattr(value, "dtype")
        and hasattr(value, "shape")
        and hasattr(value, "tobytes")
    ):
        raw = value.tobytes()
        return {
            "kind": "array",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    if isinstance(value, dict):
        if "images" in value and isinstance(value["images"], dict):
            copied = {
                key: _audit_jsonable(item)
                for key, item in value.items()
                if key != "images"
            }
            copied["images"] = {
                str(name): _audit_jsonable(image)
                for name, image in value["images"].items()
            }
            return copied
        return {str(key): _audit_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_audit_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


class SessionRunner:
    def __init__(
        self,
        output_root: str | Path,
        agent: AgentProcess,
        max_steps: int = 2000,
        step_delay: float = 0.0,
        wait_callback=None,
        viewer_callback=None,
        transition_callback=None,
        reset_requested_callback=None,
        viewer_sync_callback=None,
        log_callback=None,
        video_options: dict | None = None,
        observation_clock: ObservationClock | None = None,
        observation_archive=None,
    ):
        self.output_root = Path(output_root)
        self.agent = agent
        self.max_steps = int(max_steps)
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.video_options = video_options
        self.step_delay = max(0.0, float(step_delay))
        self.wait_callback = wait_callback
        self.observations = observation_clock or ObservationClock()
        self.observations.on_event = self._audit_observation
        self.observation_archive = observation_archive
        self._capture_context = None
        self.observations.on_sample = self._capture_observation if observation_archive is not None else None
        from .policy import InProcessPolicyAgent
        self.private_policy = isinstance(agent, InProcessPolicyAgent)
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
        self._agent_io = None
        self._current_output: Path | None = None
        self._current_session_id: str | None = None

    def _capture_observation(self, observation, evidence):
        if self._capture_context is None:
            raise RuntimeError('raw observation capture requires an actual state source')
        self.observation_archive.save(observation, **self._capture_context())

    def _set_policy_context(self, **context):
        if self.private_policy:
            self.agent.set_context(context)

    def _audit_observation(self, event, **values):
        if self._agent_io is not None:
            self._agent_io.write(
                json.dumps(
                    {
                        "timestamp": time.time(),
                        "direction": "observation",
                        "request_type": "observation",
                        "event": event,
                        "task_id": self._observation_task_id,
                        **values,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
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
        self._capture_context = None

    def _push_observation(self):
        sample = self.observations.poll()
        if sample is None:
            return
        if self._observation_context is None:
            raise RuntimeError("observation stream has no active task context")
        observation, evidence = sample
        timing = observation["timing"]
        timing["dispatched_monotonic_s"] = self.observations.clock()
        timing["age_at_dispatch_seconds"] = (
            timing["dispatched_monotonic_s"] - timing["captured_monotonic_s"]
        )
        payload = {
            **self._observation_context,
            "type": "observation",
            "observation": _jsonable(self._public_observation(observation)),
        }
        self._audit_observation("send_attempt", payload=_audit_jsonable(payload))
        # Never queue/replace camera frames on the benchmark side. A blocked
        # receiver is an integration error, not a silently dropped observation.
        self.agent.send_observation(payload)
        self._audit_observation(
            "sent",
            frame_id=timing["frame_id"],
            sent_monotonic_s=self.observations.clock(),
        )
        self._sent_frame_ids.add(timing["frame_id"])
        self._stream_evidence.append((timing["frame_id"], evidence))

    def _consumed_stream_evidence(self):
        evidence = [
            item
            for frame, items in self._stream_evidence
            if frame <= self._response_frame_id
            for item in items
        ]
        self._stream_evidence = [
            (frame, items)
            for frame, items in self._stream_evidence
            if frame > self._response_frame_id
        ]
        return evidence

    def _wait_for_agent(self):
        if self.wait_callback:
            self.wait_callback()
        self._push_observation()

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._request_number += 1
        payload = dict(
            payload,
            request_id=str(self._request_number),
            protocol_version=PROTOCOL_VERSION,
        )
        if payload.get("type") == "session_start":
            payload["observation_stream"] = dict(OBSERVATION_STREAM)

        def audit(direction, **values):
            if self._agent_io is not None:
                self._agent_io.write(
                    json.dumps(
                        {
                            "timestamp": time.time(),
                            "request_type": payload.get("type"),
                            "task_id": payload.get("task_id"),
                            "direction": direction,
                            **values,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                self._agent_io.flush()

        # Keep attempted dispatches even if the server times out or returns an
        # integration error. The response is logged before status handling.
        timing = (payload.get("observation") or {}).get("timing")
        if timing is not None:
            self._observation_context = {
                key: payload[key]
                for key in (
                    "scene_id",
                    "session_id",
                    "task_id",
                    "request_id",
                    "protocol_version",
                    "sequence_context",
                )
                if key in payload
            }
            self._sent_frame_ids.add(timing["frame_id"])
            timing["dispatched_monotonic_s"] = self.observations.clock()
            timing["age_at_dispatch_seconds"] = (
                timing["dispatched_monotonic_s"] - timing["captured_monotonic_s"]
            )
        audit("request", payload=_audit_jsonable(payload))
        try:
            response = self.agent.request(payload, on_wait=self._wait_for_agent)
            audit("response", payload=_audit_jsonable(response))
            if response.get("agent_status") == "integration_error":
                raise RuntimeError(
                    "agent integration error: "
                    + str(response.get("error") or response.get("reason_code"))
                )
            if timing is not None:
                used = response.get("observation_frame_id", timing["frame_id"])
                if (
                    type(used) is not int
                    or used not in self._sent_frame_ids
                    or used < timing["frame_id"]
                ):
                    raise ValueError(
                        "response observation_frame_id is not a sent frame for this request/task"
                    )
                self._response_frame_id = used
        except Exception as exc:
            audit("error", error={"type": type(exc).__name__, "message": str(exc)})
            raise
        return response

    def _pause(self, seconds: float | None = None) -> None:
        deadline = time.monotonic() + (
            self.step_delay if seconds is None else max(0.0, float(seconds))
        )
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

    def run_scene(
        self,
        scene_id: str,
        sessions: list[SessionRecord],
        canonical_state: dict[str, Any],
        env_factory: Callable[[str, dict[str, Any], str], Any],
    ) -> None:
        from ..artifacts.layout import manifest, initialize_run
        from ..scenes.dataset_loader import BenchmarkDataset
        from .request_runner import run_requests

        if not sessions:
            raise ValueError("a run requires at least one session")
        if manifest(self.output_root) is None:
            dataset = BenchmarkDataset(sessions[0].private_paths["dataset_root"],
                                       scene=sessions[0].private_paths.get("scene_version_root"))
            initialize_run(self.output_root, dataset)
        return run_requests(self, scene_id, sessions, canonical_state, env_factory)
