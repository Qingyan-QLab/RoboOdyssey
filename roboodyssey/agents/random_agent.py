#!/usr/bin/env python3
"""Deterministic protocol smoke agent; it is not a performance baseline."""
from roboodyssey.agents.stream import serve


class SmokeAgent:
    def __init__(self):
        self.task = None
        self.steps = 0

    def __call__(self, request):
        if request.get("type") == "session_start":
            self.task, self.steps = None, 0
            response = {"ok": True}
            if request.get('sequence_context', {}).get('reset_memory'):
                response['sequence_reset_ack'] = request['sequence_context']['run_id']
            return response
        # Current public requests identify the task but do not expose task_step.
        key = tuple(request.get(name) for name in ('scene_id', 'session_id', 'task_id'))
        if key != self.task:
            self.task, self.steps = key, 0
        self.steps += 1
        return {"action": {"action": "turn_left", "angle_degrees": 15}, "done": self.steps >= 5}


def main():
    serve(SmokeAgent(), history_frames=0)


if __name__ == '__main__':
    main()
