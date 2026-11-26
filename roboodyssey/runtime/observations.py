"""Main-thread wall-clock observation ticks; frames are returned, never buffered."""
from copy import deepcopy
import math
import time
from ..interaction.protocol import OBSERVATION_STREAM, OBSERVATION_SEND_TIMEOUT


OBSERVATION_POLICY = {
    'fps': OBSERVATION_STREAM['fps'], 'timebase': 'wall_monotonic', 'delivery': 'push_every_frame',
    'benchmark_frame_buffer': False, 'in_flight_action_requests': 1,
    'send_timeout_seconds': OBSERVATION_SEND_TIMEOUT, 'missed_deadlines': 'skip_and_log',
}


def capture_observation(env):
    """Real environments render a fresh frame; CPU fixtures may implement observe."""
    capture = getattr(env, 'capture_observation', env.observe)
    return capture(), []


class ObservationClock:
    """Pump on Isaac's main thread, including while AgentProcess awaits a reply.

    All deadlines stay on the original 0.5-second grid. A blocked render/event
    may miss deadlines; skip those slots rather than replaying stale captures.
    The injectable clock/sleep keep CPU tests independent of real elapsed time.
    """

    period = 1.0 / OBSERVATION_POLICY['fps']

    def __init__(self, *, clock=time.monotonic, sleep=time.sleep):
        self.clock, self.sleep = clock, sleep
        self.on_event = None
        self.source = self.origin = None
        self.next_slot = self.frame_id = 0
        self.missed = 0

    def _emit(self, event, **values):
        if self.on_event:
            self.on_event(event, **values)

    def set_source(self, source):
        self.source = source
        if source is not None:
            now = self.clock()
            if self.origin is None:
                self.origin = now
            else:
                # Idle task/session transitions are not acquisition failures.
                # Keep the same grid and wait for the next eligible slot.
                self.next_slot = max(self.next_slot, math.ceil((now - self.origin) / self.period))

    def poll(self):
        if self.source is None:
            return
        started = self.clock()
        deadline = self.origin + self.next_slot * self.period
        if started < deadline:
            return
        slot = max(self.next_slot, math.floor((started - self.origin) / self.period))
        missed = slot - self.next_slot
        self.next_slot = slot + 1
        observation, evidence = self.source()
        # Camera buffers can be reused by the following render. Freeze RGB,
        # depth, pose and evidence together before yielding to another update.
        observation, evidence = deepcopy(observation), deepcopy(evidence)
        completed = self.clock()
        after_render = math.floor((completed - self.origin) / self.period) + 1
        missed += max(0, after_render - self.next_slot)
        self.next_slot = max(self.next_slot, after_render)
        self.missed += missed
        self.frame_id += 1
        timing = {
            'frame_id': self.frame_id, 'fps': OBSERVATION_POLICY['fps'],
            'timebase': OBSERVATION_POLICY['timebase'], 'slot': slot,
            'scheduled_monotonic_s': self.origin + slot * self.period,
            'sample_started_monotonic_s': started, 'captured_monotonic_s': completed,
            'capture_seconds': completed - started,
            'lateness_seconds': started - (self.origin + slot * self.period),
            'missed_slots': missed, 'missed_slots_total': self.missed,
        }
        observation['timing'] = timing
        self._emit('sample', **timing)
        return observation, evidence

    def take(self, on_wait=None):
        if self.source is None:
            raise RuntimeError('observation source is not active')
        while True:
            sample = self.poll()
            if sample is not None:
                return sample
            if on_wait:
                on_wait()
            remaining = self.origin + self.next_slot * self.period - self.clock()
            if remaining > 0:
                self.sleep(min(.02, remaining))
