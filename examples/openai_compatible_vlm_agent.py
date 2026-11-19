#!/usr/bin/env python3
"""Protocol-v1 adapter for an OpenAI-compatible vision chat endpoint.

The endpoint is expected to return one JSON object: ``{"action": ...}`` for a
step request or ``{"answer": ...}`` for QA. Session-start observations are
summarized into the adapter's private conversation state.
"""
from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roboodyssey.agents.stream import serve


SYSTEM = """You are HoloAgent, an embodied agent in an indoor simulator. Return JSON only.
For step requests return {"action":{"action":"...", ...},"done":false}.
Legal actions: no_op, move_forward, move_backward, move_left, move_right,
turn_left, turn_right, look_up, look_down, pick, place, open, close, reset.
For pick/open/close, provide object_id when known; otherwise provide a visible
semantic label in target. For place, provide room_id only when known from your
own observations. nearby_manipulable_objects lists local physical affordances;
pick an exact reachable object_id from it. If held_object is non-null, navigate
to the requested destination room and place it there. A blocked movement must
be followed by a turn or lateral move, never the same movement repeatedly.
Use previous_result to recover from failed actions.
For QA return {"answer":"..."}. You may return memory_update containing only
facts inferred from supplied observations and instructions. Never request or
assume scene ground truth, hidden target coordinates, reference trajectories,
or expected answers. Never include markdown."""


def _image_url(value: Any) -> str | None:
    if not isinstance(value, dict) or value.get("encoding") != "base64":
        return None
    array = np.frombuffer(base64.b64decode(value["data"]), dtype=np.dtype(value["dtype"]))
    array = array.reshape(value["shape"])
    if array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        maximum = float(array.max()) if array.size else 1.0
        array = np.clip(array * (255.0 if maximum <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _images(payload: dict[str, Any], limit: int = 4) -> list[str]:
    candidates = []
    for observation in payload.get('observation_history', []):
        candidates.extend((observation.get('images') or {}).values())
    observation = payload.get("observation", {})
    candidates.extend((observation.get("images") or {}).values())
    for view in payload.get("session_views", [])[-limit:]:
        candidates.extend((view.get("observation", {}).get("images") or {}).values())
    return [url for item in candidates if (url := _image_url(item)) is not None][-limit:]


def _text(payload: dict[str, Any]) -> str:
    public = {key: value for key, value in payload.items()
              if key not in {"observation", "observation_history", "session_views", "memory"}}
    memory = payload.get("memory", {})
    return "Request:\n" + json.dumps(public, ensure_ascii=False) + \
           "\nMemory supplied by benchmark:\n" + json.dumps(memory, ensure_ascii=False)


class Adapter:
    def __init__(self, endpoint: str, model: str, api_key: str | None, timeout: float):
        self.endpoint = endpoint.rstrip("/")
        if not self.endpoint.endswith("/chat/completions"):
            self.endpoint += "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": _text(payload)}]
        content.extend({"type": "image_url", "image_url": {"url": url}}
                       for url in _images(payload))
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": content}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(self.endpoint, data=body, headers=headers, method="POST")
        with urlopen(request, timeout=self.timeout) as response:  # noqa: S310
            result = json.loads(response.read().decode("utf-8"))
        content_value = result["choices"][0]["message"]["content"]
        return json.loads(content_value) if isinstance(content_value, str) else content_value


def _pil_images(payload: dict[str, Any], limit: int = 4) -> list[Image.Image]:
    result = []
    for url in _images(payload, limit):
        encoded = url.split(",", 1)[1]
        result.append(Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB"))
    return result


class LocalTransformersAdapter:
    """Run a local multimodal Hugging Face checkpoint without an API server."""

    def __init__(self, model_path: str, device: str = "auto",
                 max_new_tokens: int = 256) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if device == "auto":
            if not torch.cuda.is_available():
                device = "cpu"
            else:
                device = f"cuda:{min(1, torch.cuda.device_count() - 1)}"
        if device.startswith("cuda"):
            index = int(device.split(":", 1)[1]) if ":" in device else 0
            if index >= torch.cuda.device_count():
                raise ValueError(
                    f"requested {device}, but only {torch.cuda.device_count()} CUDA device(s) "
                    "are visible; use --local-vlm-device auto or cuda:0"
                )
        print(f"local VLM device: {device}", flush=True)
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=dtype, trust_remote_code=True,
        ).to(device).eval()
        self.device = device
        self.max_new_tokens = max(32, int(max_new_tokens))

    @staticmethod
    def _json_object(text: str) -> dict[str, Any]:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                continue
        raise ValueError(f"local VLM did not return JSON: {text[-500:]}")

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {"type": "image", "image": image} for image in _pil_images(payload)
        ]
        content.append({"type": "text", "text": _text(payload)})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": content},
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", enable_thinking=False,
        )
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value
                  for key, value in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
            )
        prompt_length = inputs["input_ids"].shape[1]
        text = self.processor.batch_decode(
            generated[:, prompt_length:], skip_special_tokens=True,
        )[0]
        return self._json_object(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default=os.environ.get("VLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("VLM_MODEL"), required=os.environ.get("VLM_MODEL") is None)
    parser.add_argument("--api-key", default=os.environ.get("VLM_API_KEY"))
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    adapter = Adapter(args.endpoint, args.model, args.api_key, args.timeout)
    def handle(payload):
        response = adapter.request(payload)
        if payload.get("type") == "session_start" and not isinstance(response, dict):
            response = {"ok": True}
        if payload.get('type') == 'session_start' and payload.get('sequence_context', {}).get('reset_memory'):
            response['sequence_reset_ack'] = payload['sequence_context']['run_id']
        return response
    serve(handle, history_frames=3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
