#!/usr/bin/env python3
"""Deterministic protocol smoke agent; it is not a performance baseline."""
import json
import sys


for line in sys.stdin:
    request = json.loads(line)
    request_type = request.get("type")
    if request_type == "session_start":
        response = {"ok": True}
    elif request_type == "qa":
        response = {"answer": ""}
    else:
        step = int(request.get("task_step", 0))
        response = {"action": {"action": "turn_left", "angle_degrees": 15}, "done": step >= 4}
    print(json.dumps(response), flush=True)
