"""Typed, versioned state branches; planned facts cannot enter actual history."""
from copy import deepcopy
from dataclasses import dataclass

from .contextual import normalize_state, RuntimeContractError
from ..artifacts.layout import json_digest


def state_version(state, *, source):
    """Content identity of a normalized state and its actual/planned source."""
    if source not in {'planned', 'actual'}:
        raise ValueError('state source must be planned or actual')
    return json_digest({'source': source, 'state': state}, allow_nan=False)


@dataclass(frozen=True)
class StateSnapshot:
    source: str
    version: str
    event_id: str
    _state: dict

    @property
    def state(self):
        return deepcopy(self._state)


class StateBranch:
    """One run/compile branch with explicit source and revision-bound cache.

    Only caller-supplied environment readback enters an actual branch. Planned
    commits cannot be used on it. Snapshots own state copies; asset files are
    never copied. Large-state copy costs remain a separate performance concern.
    """
    def __init__(self, cache, state, *, source, event_id='initial'):
        if source not in {'planned', 'actual'}:
            raise ValueError('state source must be planned or actual')
        self._cache = cache
        self.source = source
        self._history = []
        self._derived = {}
        self._snapshot = self._make(state, event_id)

    def _make(self, state, event_id):
        if not isinstance(event_id, str) or not event_id:
            raise ValueError('state changes require a source event')
        normalized = normalize_state(self._cache, state)
        version = state_version(normalized, source=self.source)
        return StateSnapshot(self.source, version, event_id, normalized)

    def snapshot(self):
        value = self._snapshot
        return StateSnapshot(value.source, value.version, value.event_id, value.state)

    def fork(self):
        if self.source != 'planned':
            raise RuntimeContractError('actual state cannot branch into assumed execution')
        branch = StateBranch(self._cache, self._snapshot.state, source='planned', event_id=self._snapshot.event_id)
        branch._history = deepcopy(self._history)
        return branch

    def _replace(self, state, *, expected_version, event_id):
        if expected_version != self._snapshot.version:
            raise RuntimeContractError('stale state commit')
        next_snapshot = self._make(state, event_id)
        if next_snapshot.version != self._snapshot.version:
            self._derived.clear()
        self._snapshot = next_snapshot
        return self.snapshot()

    def commit(self, state, *, expected_version, event_id):
        if self.source != 'planned':
            raise RuntimeContractError('actual state requires environment readback')
        return self._replace(state, expected_version=expected_version, event_id=event_id)

    def readback(self, state, *, event_id):
        if self.source != 'actual':
            raise RuntimeContractError('environment readback belongs to an actual branch')
        return self._replace(state, expected_version=self._snapshot.version, event_id=event_id)

    def append_history(self, event, *, source):
        if source != self.source or (self.source == 'actual' and (
                event.get('planned_success_assumption') or event.get('source') == 'planned'
                or event.get('state_source') == 'planned'
                or event.get('evidence_status') == 'planned_success_assumption'
                or any(key.startswith('assumed_') for key in event))):
            raise RuntimeContractError('planned and actual histories cannot be mixed')
        self._history.append(deepcopy(event))

    def history(self):
        return deepcopy(self._history)

    def cached(self, key, compute):
        """Cache only within this source/state revision; every change invalidates."""
        if key not in self._derived:
            self._derived[key] = compute(self.snapshot())
        return self._derived[key]
