"""Simulator interface for the v5 request runner."""
from __future__ import annotations

from typing import Any, Callable, Protocol, Union

State = dict[str, Any]
Action = Union[str, dict[str, Any]]


class RequestEnvironment(Protocol):
    """Required operations for running a complete request sequence.

    The factory returns a ready environment. The runner owns it until close,
    and keeps the same instance across Sessions. Raw observations may carry
    private fields. Only the runner's curated observation is sent to the Agent.

    Scoring can require navigation and transfer evidence. Review runs
    can also request rendered frames.
    """

    def observe(self) -> State: ...
    def runtime_state_v5(self) -> State: ...
    def apply_mutation_v5(self, prepared: State) -> None: ...
    def get_agent_pose(self) -> State: ...
    def set_task(self, task: State) -> None: ...
    def step(self, action: Action) -> tuple[Any, Any, bool, bool, State]: ...
    def close(self) -> None: ...


RequestEnvironmentFactory = Callable[[str, State, str], RequestEnvironment]
