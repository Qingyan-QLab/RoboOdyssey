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
from typing import Any

from .protocol import PROTOCOL_VERSION, OBSERVATION_STREAM, OBSERVATION_SEND_TIMEOUT, validate_response


class AgentProcess:
    def __init__(self, command: str, timeout: float = 120.0, stderr_callback=None):
        env = os.environ.copy()
        env["PATH"] = ":".join(p for p in env.get("PATH", "").split(":") if "/isaacsim/kit/python" not in p)
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)
        self.process = subprocess.Popen(
            shlex.split(command), shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
            cwd=None,
            env=env,
        )
        self.timeout = timeout
        # A non-reading agent must not block Isaac indefinitely. Writes are
        # immediate and bounded; there is no benchmark-side frame queue.
        os.set_blocking(self.process.stdin.fileno(), False)
        self._response_bytes = bytearray()
        self._request_number = 0
        self._stream_ready = False
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

    def _send(self, payload, timeout):
        if self.process.poll() is not None:
            raise RuntimeError(
                f"agent exited with code {self.process.returncode}: {self._stderr_tail()}")
        assert self.process.stdin
        payload = dict(payload)
        payload.setdefault("protocol_version", PROTOCOL_VERSION)
        data = memoryview((json.dumps(payload, ensure_ascii=False) + '\n').encode('utf-8'))
        deadline = time.monotonic() + timeout
        while data:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('agent stdin blocked; observation delivery failed (no frame was dropped or queued)')
            _, writable, _ = select.select([], [self.process.stdin], [], min(.02, remaining))
            if writable:
                try:
                    sent = os.write(self.process.stdin.fileno(), data)
                except BlockingIOError:
                    continue
                data = data[sent:]

    def send_observation(self, payload):
        if not self._stream_ready:
            raise RuntimeError('Agent must acknowledge observation_stream before receiving frames')
        if payload.get('type') != 'observation':
            raise ValueError('expected observation notification')
        self._send(payload, timeout=OBSERVATION_SEND_TIMEOUT)

    def request(self, payload: dict[str, Any], on_wait=None) -> dict[str, Any]:
        payload = dict(payload)
        self._request_number += 1
        payload.setdefault('request_id', str(self._request_number))
        if payload.get('type') == 'session_start':
            payload.setdefault('observation_stream', dict(OBSERVATION_STREAM))
        elif not self._stream_ready:
            raise RuntimeError('Agent must acknowledge observation_stream in session_start')
        self._send(payload, timeout=self.timeout if payload.get('type') == 'session_start' else OBSERVATION_SEND_TIMEOUT)
        assert self.process.stdout
        deadline = time.monotonic() + self.timeout
        line = ""
        while time.monotonic() < deadline:
            if b'\n' in self._response_bytes:
                raw, _, rest = self._response_bytes.partition(b'\n')
                self._response_bytes = bytearray(rest)
                line = raw.decode('utf-8')
                break
            ready, _, _ = select.select([self.process.stdout], [], [], 0.02)
            if ready:
                raw = os.read(self.process.stdout.fileno(), 65536)
                if not raw:
                    break
                self._response_bytes.extend(raw)
                # Check complete responses before taking a new camera frame;
                # partial lines still run the sampling pump and timeout clock.
                if b'\n' in self._response_bytes:
                    continue
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
        if not isinstance(value, dict) or value.get('request_id') != payload['request_id']:
            raise ValueError('Agent response request_id does not match the active request')
        if payload.get('type') == 'session_start':
            if value.get('observation_stream_ack') is not True:
                raise ValueError('Agent does not support continuous observation messages; update its receiver')
            self._stream_ready = True
        elif (payload.get('observation') or {}).get('timing') and type(value.get('observation_frame_id')) is not int:
            raise ValueError('Agent must identify the observation_frame_id used for its response')
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
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream is not None:
                stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
