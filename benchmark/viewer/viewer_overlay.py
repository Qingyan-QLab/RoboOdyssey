"""Movable translucent commander dialog for viewer runs."""
from __future__ import annotations
import math
import time

def _english(value: object, fallback: str = "Complete the current task.") -> str:
    text = str(value or "")
    return text if text else fallback

class ViewerOverlay:
    def __init__(self):
        import omni.ui as ui
        self._ui = ui
        self._reset_requested = False
        self._operation_count = 0
        self._movement_count = 0
        self._reset_count = 0
        self.window = ui.Window("Commander / User", width=560, height=190)
        try:
            self.window.position_x = 24
            self.window.position_y = 42
        except Exception: pass
        self.window.visible = True
        with self.window.frame:
            with ui.VStack(spacing=7, padding=12, style={"background_color": 0xB8202633}):
                with ui.HStack(height=28):
                    self._title = ui.Label("COMMANDER / USER", style={"font_size": 18, "color": 0xFF70D6FF})
                    ui.Spacer()
                    self._reset = ui.Button("RESET (0)", width=105, clicked_fn=self._on_reset,
                        style={"Button": {"background_color": 0xFF8F3F45, "color": 0xFFFFFFFF}})
                self._body = ui.Label("Waiting for session...", word_wrap=True, style={"font_size": 16, "color": 0xFFFFFFFF})
                self._session = ui.Label("Scene memory: —", word_wrap=True, style={"font_size": 14, "color": 0xFFB8E6A0})
                self._status = ui.Label("", style={"font_size": 13, "color": 0xFFB8C4D6})

    def _on_reset(self): self._reset_requested = True
    def consume_reset(self) -> bool:
        requested, self._reset_requested = self._reset_requested, False
        return requested

    def update(self, **values):
        event = values.get("event", "")
        task = values.get("task") or values.get("instruction")
        if event == "session_start":
            self._operation_count = 0
            self._movement_count = 0
            self._reset_count = 0
            self._reset.enabled = False
            text = _english(values.get("session_instruction") or values.get("instruction"))
            self._session.text = "Scene memory: " + text
        elif event == "task_start":
            self._operation_count = self._movement_count = 0
        elif event in {"step", "reset"}:
            self._operation_count = int(values.get("step", self._operation_count)) + 1
            before = values.get("before") or {}
            after = values.get("after") or {}
            before_position = before.get("position") or []
            after_position = after.get("position") or []
            if len(before_position) >= 2 and len(after_position) >= 2:
                if math.dist([float(v) for v in before_position[:2]], [float(v) for v in after_position[:2]]) > 1e-5:
                    self._movement_count += 1
        if event == "reset":
            self._title.text = "COMMANDER  •  RESET"
            self._body.text = "Agent reset to a safe walkable location. Continue the current task."
        elif event == "step":
            action = values.get("action") or {}
            action_name = action.get("action", action) if isinstance(action, dict) else action
            self._title.text = f"AGENT  •  {action_name}"
            self._body.text = _english(task)
        elif event == "task_complete":
            self._title.text = "TASK RESULT  •  " + ('SUCCESS' if values.get('success') else 'FAILED' if values.get('success') is False else 'EXCLUDED')
            self._body.text = _english(task)
        elif event in {'task_repair', 'task_excluded', 'integration_error', 'agent_unsupported', 'agent_request'}:
            self._title.text = {'task_repair':'REPAIRED REQUEST', 'task_excluded':'NOT DISPATCHED',
                                'integration_error':'INTEGRATION ERROR','agent_unsupported':'UNSUPPORTED CAPABILITY',
                                'agent_request':'WAITING FOR AGENT'}[event]
            self._body.text = _english(task, str(values.get('error') or values.get('reason_code') or event))
        elif task:
            self._title.text = "COMMANDER / USER"
            self._body.text = _english(task)
        self._reset_count = int(values.get("reset_count", self._reset_count))
        self._reset.text = f"RESET ({self._reset_count})"
        after = values.get("after") or {}
        position = after.get("position") or []
        pose_text = ""
        if len(position) >= 2:
            pose_text = f" | pos=({float(position[0]):.2f},{float(position[1]):.2f})"
        self._status.text = f"{values.get('scene_id','')} | {values.get('session_id','')} | task {values.get('task_id','')} ({values.get('task_type','')}) | steps {self._operation_count} | moves {self._movement_count} | resets {self._reset_count}{pose_text}"

    def transition(self, seconds=3.0, app=None):
        import omni.ui as ui
        card = ui.Window("Time Transition", width=4096, height=2160, flags=ui.WINDOW_FLAGS_NO_RESIZE | ui.WINDOW_FLAGS_NO_SCROLLBAR | ui.WINDOW_FLAGS_NO_TITLE_BAR | ui.WINDOW_FLAGS_NO_MOVE)
        with card.frame:
            with ui.VStack(style={"background_color": 0xFFFFFFFF}):
                ui.Spacer(); ui.Label("a few seconds later", alignment=ui.Alignment.CENTER, style={"color": 0xFF000000, "font_size": 28}); ui.Spacer()
        card.visible = True; end = time.monotonic() + max(0.0, float(seconds))
        while time.monotonic() < end:
            if app is not None: app.update()
            time.sleep(0.02)
        card.visible = False

    def close(self):
        if self.window is not None: self.window.visible = False
