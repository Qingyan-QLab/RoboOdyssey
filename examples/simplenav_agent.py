#!/usr/bin/env python3
"""SimpleNav JSONL adapter for :mod:`roboodyssey.run`.

The adapter keeps the benchmark protocol independent from SimpleNav.  Navigation
requests are converted to SimpleNav waypoint queries; unsupported QA and
manipulation requests are answered as failures so the benchmark can continue.
"""
from __future__ import annotations

import argparse
import base64
import sys
from typing import Any
from urllib.parse import urlparse

import numpy as np


def _decode(value: Any) -> Any:
    """Decode arrays produced by roboodyssey.runtime.session_runner._jsonable."""
    if isinstance(value, dict):
        if value.get("encoding") == "base64" and {"dtype", "shape", "data"} <= value.keys():
            raw = base64.b64decode(value["data"])
            return np.frombuffer(raw, dtype=np.dtype(value["dtype"])).reshape(value["shape"]).copy()
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _url_parts(url: str) -> tuple[str, int]:
    parsed = urlparse(url)
    if (parsed.scheme != "ws" or not parsed.hostname or parsed.path not in ('', '/')
            or parsed.query or parsed.fragment or parsed.username or parsed.password
            or (parsed.port is not None and not 0 < parsed.port < 65536)):
        raise ValueError("--url must be ws://host:port without credentials or a path")
    return parsed.hostname, int(parsed.port or 10093)


# The pure state machine is importable without the SimpleNav model stack.
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roboodyssey.agents.simplenav import SimpleNavAgent, simple_action as _simple_action, instruction_text as _instruction_text
from roboodyssey.agents.stream import serve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:10093")
    parser.add_argument("--checkpoint", required=True,
                        help="SimpleNav checkpoint; its dataset_statistics.json is used for unnormalization")
    parser.add_argument("--simplenav-root", type=Path, help="SimpleNav source root, containing deployment and NavVLAeval")
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument("--history-frames", type=int, default=8)
    parser.add_argument("--collision-reset-threshold", type=int, default=5,
                        help="reset after this many consecutive blocked moves")
    args = parser.parse_args()

    if args.simplenav_root:
        sys.path.insert(0, str(args.simplenav_root.resolve()))

    # These imports intentionally happen after argument parsing: this script is
    # run by the SimpleNav environment, not by Isaac Sim's Python.
    from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
    from NavVLAeval.common.data.runtime_dataset import (
        OnlineNavVLARuntimeDatasetAdapter,
        VLNCE_PLATFORM_TEXT,
    )
    from NavVLAeval.common.model.action_codec import ActionCodec
    from NavVLAeval.common.types import EpisodeHistory

    host, port = _url_parts(args.url)
    policy = WebsocketClientPolicy(host=host, port=port)
    codec = ActionCodec.from_checkpoint(args.checkpoint, unnorm_key=args.unnorm_key)
    adapter = OnlineNavVLARuntimeDatasetAdapter(
        required_cameras=("front",),
        history_image_frames=args.history_frames,
        history_selection="recent",
        platform_text=VLNCE_PLATFORM_TEXT,
    )
    agent = SimpleNavAgent(policy, codec, adapter, EpisodeHistory,
                           collision_reset_threshold=args.collision_reset_threshold)
    try:
        serve(agent.handle, decode=_decode, history_frames=args.history_frames)
    finally:
        policy.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
