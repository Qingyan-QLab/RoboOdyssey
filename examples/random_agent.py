#!/usr/bin/env python3
"""Deterministic protocol smoke agent; it is not a performance baseline."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.agents.stream import serve


def handle(request):
    request_type = request.get("type")
    if request_type == "session_start":
        response = {"ok": True}
        if request.get('sequence_context', {}).get('reset_memory'):
            response['sequence_reset_ack'] = request['sequence_context']['run_id']
    elif request_type == "qa":
        response = {"answer": ""}
    else:
        step = int(request.get("task_step", 0))
        response = {"action": {"action": "turn_left", "angle_degrees": 15}, "done": step >= 4}
    return response


if __name__ == '__main__':
    serve(handle, history_frames=0)
