"""SimpleNav protocol state machine. Backends are injected; importing is CPU-only."""
import math
import numpy as np


def instruction_text(value):
    if isinstance(value, str):
        return value
    return str(value or '')


def simple_action(raw):
    values = np.asarray(raw, dtype=np.float32)
    if not values.size or values.size % 4:
        raise ValueError('SimpleNav returned an invalid waypoint shape')
    dx, dy, _dz, dyaw = (float(v) for v in values.reshape(-1, 4)[0])
    if not all(math.isfinite(v) for v in (dx, dy, _dz, dyaw)):
        raise ValueError('SimpleNav returned a non-finite waypoint')
    if abs(dyaw) >= .05 and abs(dyaw) >= max(abs(dx), abs(dy)):
        return {'action': 'turn_right' if dyaw > 0 else 'turn_left', 'angle_degrees': max(1, math.degrees(abs(dyaw)))}
    if abs(dx) >= .05:
        return {'action': 'move_forward' if dx > 0 else 'move_backward', 'distance': abs(dx)}
    if abs(dy) >= .05:
        return {'action': 'move_right' if dy > 0 else 'move_left', 'distance': abs(dy)}
    return {'action': 'no_op'}


class SimpleNavAgent:
    def __init__(self, policy, codec, adapter, history_factory, *, collision_reset_threshold=5):
        self.policy, self.codec, self.adapter = policy, codec, adapter
        self.history_factory = history_factory
        self.history = history_factory()
        self.scene_id = None
        self.task_key = None
        self.step = 0
        self.collision_streak = 0
        self.collision_reset_threshold = collision_reset_threshold
        self.definitions, self.requests = [], []
        self.context = ''

    def handle(self, request):
        kind = request.get('type')
        if kind == 'session_start':
            sequence = request.get('sequence_context', {})
            if request.get('scene_id') != self.scene_id or sequence.get('reset_memory'):
                self.history = self.history_factory()
                self.scene_id = request.get('scene_id')
                self.definitions, self.requests = [], []
                self.step = 0
            self.task_key = None
            self.collision_streak = 0
            self.context = instruction_text(request.get('instruction'))
            for text in request.get('scene_briefing', []):
                if text not in self.definitions:
                    self.definitions.append(text)
            return {'memory_update': {}, **({'sequence_reset_ack': sequence['run_id']} if sequence.get('reset_memory') else {})}
        if kind != 'step':
            raise ValueError('unknown request type: ' + str(kind))
        key = (request.get('session_id'), request.get('task_id'))
        if key != self.task_key:
            self.task_key = key
            self.collision_streak = 0
            for text in request.get('name_introductions', []):
                if text not in self.definitions:
                    self.definitions.append(text)
            self.requests.append({'time': request.get('current_time'), 'instruction': request.get('instruction', '')})
            self.requests = self.requests[-32:]
        task_type = request.get('task_type')
        if task_type != 'long_horizon':
            raise ValueError('expected a long_horizon request')
        caps = set(request.get('capabilities', []))
        if not caps:
            raise ValueError('v5 request is missing public capabilities')
        unsupported = ('qa' if 'qa' in caps else
                       'manipulation' if 'manipulation' in caps else
                       None)
        if unsupported or 'navigation' not in caps:
            return {'action': {'action': 'no_op'}, 'answer': '', 'done': True,
                    'agent_status': 'unsupported', 'reason_code': 'simplenav_unsupported_' + (unsupported or 'capability')}
        observation_input = request.get('observation') or {}
        if observation_input.get('images', {}).get('front') is None:
            raise ValueError('SimpleNav requires a front RGB image')
        feedback = observation_input.get('action_feedback') or {}
        self.collision_streak = self.collision_streak + 1 if feedback.get('collision') else 0
        allowed = request.get('allowed_actions')
        if (self.collision_reset_threshold > 0 and self.collision_streak >= self.collision_reset_threshold
                and (allowed is None or 'reset' in allowed)):
            self.collision_streak = 0
            return {'action': {'action': 'reset'}, 'done': False, 'action_source': 'adapter_recovery'}
        def prepare(value):
            timing = value.get('timing') or {}
            prepared = self.adapter.prepare_observation_for_step(
                observation=value, step=timing.get('frame_id', self.step))
            if timing:
                prepared = dict(prepared)
                prepared['navvla_eval'] = {**prepared.get('navvla_eval', {}),
                    'frame_index': timing['frame_id'], 'timestamp': timing['captured_monotonic_s']}
            return prepared
        observation = prepare(observation_input)
        visual_history = self.history
        if 'observation_history' in request:
            # Incoming frames reach Agent memory while predict_action blocks.
            # This snapshot excludes the current frame and later arrivals.
            visual_history = self.history_factory()
            for item in request['observation_history']:
                if item.get('images', {}).get('front') is not None:
                    visual_history.observations.append(prepare(item))
                    visual_history.images.append(item['images']['front'])
                    visual_history.instructions.append('')
            if hasattr(visual_history, 'raw_actions'):
                visual_history.raw_actions = list(self.history.raw_actions)
        instruction = '\n'.join([
            # SimpleNav's Qwen wrapper keeps the first max_text_tokens tokens.
            # The active request must survive when accumulated memory is long.
            'Current task: ' + str(request.get('instruction', '')),
            'Current scenario time: ' + str(request.get('current_time', '')),
            'Public name definitions: ' + ' '.join(self.definitions),
            'Session context: ' + self.context,
            'Earlier requests, newest first (not evidence of success): ' + str(list(reversed(self.requests[:-1]))),
        ])
        if feedback.get('collision'):
            instruction += '\nPrevious movement was blocked; choose a different walkable direction or distance.'
        example = self.adapter.build_example(observation=observation, history=visual_history, instruction=instruction)
        response = self.policy.predict_action({'examples': [example]})
        if response.get('ok') is False:
            raise RuntimeError('SimpleNav service error: ' + str(response.get('error')))
        prediction = self.codec.decode(response.get('data', response))
        action = simple_action(prediction.raw_actions)
        if allowed is not None and action['action'] not in allowed:
            raise ValueError('predicted action is outside the public allowed_actions')
        self.adapter.update_history(history=self.history, observation=observation, prediction=prediction, instruction=instruction)
        self.step += 1
        return {'action': action, 'done': False, 'agent_status': 'ok', 'action_source': 'model'}
