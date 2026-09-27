"""Language-independent JSONL subprocess agent protocol.

Agents may return ``{"action": {"action": "reset"}}`` to respawn at the
environment's collision-free safe point. Reset events remain in the log and
are reflected by the reset-aware score.
"""
from __future__ import annotations

import json
import subprocess
import os
import select
import shlex
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from .protocol import PROTOCOL_VERSION, validate_response


class AgentProcess:
    def __init__(self, command: str, timeout: float = 120.0, stderr_callback=None):
        env = os.environ.copy()
        env["PATH"] = ":".join(p for p in env.get("PATH", "").split(":") if "/isaacsim/kit/python" not in p)
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)
        self.process = subprocess.Popen(
            shlex.split(command), shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
            cwd=Path(__file__).resolve().parents[1],
            env=env,
        )
        self.timeout = timeout
        self.stderr_callback = stderr_callback
        self._stderr_lines: deque[str] = deque(maxlen=80)
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            self._stderr_lines.append(line)
            sys.stderr.write(f"[agent] {line}")
            sys.stderr.flush()
            if self.stderr_callback is not None:
                self.stderr_callback(line)

    def _stderr_tail(self, limit: int = 2000) -> str:
        return "".join(self._stderr_lines)[-limit:]

    def request(self, payload: dict[str, Any], on_wait=None) -> dict[str, Any]:
        if self.process.poll() is not None:
            raise RuntimeError(
                f"agent exited with code {self.process.returncode}: {self._stderr_tail()}")
        assert self.process.stdin and self.process.stdout
        payload = dict(payload)
        payload.setdefault("protocol_version", PROTOCOL_VERSION)
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        deadline = time.monotonic() + self.timeout
        line = ""
        while time.monotonic() < deadline:
            ready, _, _ = select.select([self.process.stdout], [], [], 0.02)
            if ready:
                line = self.process.stdout.readline()
                break
            if on_wait is not None:
                on_wait()
            if self.process.poll() is not None:
                break
        if not line and self.process.poll() is None:
            details = self._stderr_tail()
            suffix = f"; recent stderr:\n{details}" if details else ""
            raise TimeoutError(
                f"agent did not respond within {self.timeout:.1f}s{suffix}")
        if not line:
            raise RuntimeError(f"agent closed stdout: {self._stderr_tail(1000)}")
        value = json.loads(line)
        return validate_response(str(payload.get("type", "")), value)

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self._stderr_thread.join(timeout=1)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
