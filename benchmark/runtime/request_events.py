"""Private chronological events shared by the viewer, recorder and audit report."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import time

from .recording import SessionRecorder
from ..artifacts.layout import session_file, video_dir


class RequestEvents:
    def __init__(self, runner, env, output, scene_id, session_id):
        self.runner, self.env = runner, env
        self.scene_id, self.session_id = scene_id, session_id
        self.counter = 0
        self.stream = session_file(output, 'runtime_events.jsonl').open('x', encoding='utf-8')
        self.recorder = None
        try:
            if runner.video_options:
                self.recorder = SessionRecorder(video_dir(output), **runner.video_options)
        except BaseException:
            self.stream.close()
            raise

    def emit(self, kind, *, sample=False, **values):
        self.counter += 1
        event = {**deepcopy(values), 'event': kind, 'scene_id': self.scene_id,
                 'session_id': self.session_id, 'event_id': f'{self.session_id}:{self.counter:06d}',
                 'utc': datetime.now(timezone.utc).isoformat(), 'monotonic_ns': time.monotonic_ns()}
        self.stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n'); self.stream.flush()
        if self.runner.viewer_sync_callback and sample:
            self.runner.viewer_sync_callback(self.env)
        self.runner._show(**event)
        self.runner._log(f"{kind} {self.session_id}/{event.get('task_id', '-')} {event.get('reason_code', '')}")
        if self.recorder:
            frames = self.env.capture_review_frames(self.recorder.options['layout']) if sample else None
            self.recorder.capture(event, frames)
        return event['event_id']

    def close(self, status):
        try:
            if self.recorder:
                self.recorder.close(status)
        finally:
            self.stream.close()


def state_changes(before, after):
    """Capture actual object and robot differences, regardless of task success."""
    fields = ('room_id', 'held', 'support_object_id', 'geometry', 'door_open', 'pose', 'interaction_position')
    changes = []
    for oid in sorted(set(before.get('objects', {})) | set(after.get('objects', {}))):
        old, new = before.get('objects', {}).get(oid, {}), after.get('objects', {}).get(oid, {})
        delta = {k: {'before': deepcopy(old.get(k)), 'after': deepcopy(new.get(k))}
                 for k in fields if old.get(k) != new.get(k)}
        if delta:
            changes.append({'object_id': oid, 'changes': delta})
    return {'objects': changes, 'robot': {'before': before.get('robot_pose'), 'after': after.get('robot_pose')},
            'held_object': {'before': before.get('held_object'), 'after': after.get('held_object')}}


def source_context(template, history):
    """Exact bound sources, including failures; context does not prove causality."""
    sources = []
    for goal in template['goals']:
        binding = goal.get('history_binding') or goal.get('place_binding')
        if binding:
            for e in history:
                if all(e.get(k) == v for k, v in binding.items()):
                    sources.append({k: deepcopy(e.get(k)) for k in ('session_id', 'task_id', 'goal_id', 'object_id',
                        'dispatched', 'success', 'reason_code', 'location_experience', 'failure_causes')})
    return sources
