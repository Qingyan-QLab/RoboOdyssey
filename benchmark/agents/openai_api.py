"""API agent using a server-side response chain and one new RGB per decision."""
from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import re
import socket
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import uuid

import numpy as np
from PIL import Image

from .protocol import ACTION_TYPES, validate_response
from ..artifacts.layout import write_json, json_digest


SYSTEM = """You control an indoor simulated robot from public task messages and front RGB.
Remember names, places, experiences and robot state yourself using this conversation.
The initial robot state is supplied once. Estimate subsequent state from images,
your actions and the available execution flags; a requested action may not succeed.
Never assume hidden target coordinates, object/room IDs, maps or ground truth.
Each new image requests one decision. Follow the current task and its allowed_actions.
Task goals may be ordered: do not finish after only the first of several goals.
Distances are metres, turns are degrees; null uses defaults of 0.25 m and 15 degrees.
pick uses an introduced public name in target; place releases at the current position.
A collision flag reports a blocked translation, not the exact executed distance.
Other execution success flags concern the action, not task success. After a collision,
change direction or viewpoint. Missing feedback means unavailable, not success.
reset recovers the robot pose; world state, task budget and conversation continue.
For actions_and_answer, explore with allowed actions, then answer briefly using no_op
and done=true. Use introduced area names for location answers. For legacy qa an answer
may use action=null. Scenario time, not wall time, determines terms such as yesterday.
Return JSON only with exactly these fields:
{"action":{"action":"no_op","distance":null,"angle_degrees":null,"target":null},"answer":null,"done":false}
Choose one legal action and use null for unused arguments. Keep answer=null until
answering. Set done=true when you decide to finish the current task; the environment
judges actual success. Do not output memory summaries, facts or extra commentary.
"""

DEFAULTS = {
    'endpoint': 'https://api.openai.com/v1', 'api': 'responses',
    'api_key_env': 'OPENAI_API_KEY', 'request_budget_seconds': 115.0,
    'max_output_tokens': 2048, 'reasoning_effort': None, 'output_format': 'json_schema',
    'image_detail': 'auto', 'feedback': 'minimal',
}
TRANSLATIONS = {'move_forward', 'move_backward', 'move_left', 'move_right'}
EXECUTION_FLAGS = {'pick', 'place', 'open', 'close', 'reset'}
INPUT_POLICY = {
    'context': 'server_response_chain', 'system_prompt': 'first_response_only',
    'images': 'one_current_rgb_per_decision', 'robot_state': 'sequence_initial_only',
    'explicit_memory': False, 'task_text': 'on_change_only', 'automatic_retries': False,
}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _digest(value):
    return json_digest(value, allow_nan=False)


def _loads(value):
    def invalid_constant(value):
        raise ValueError('non-finite JSON value')
    return json.loads(value, parse_constant=invalid_constant)


def load_config(path):
    """Validate without reading credentials, contacting APIs or creating outputs."""
    path = Path(path).expanduser().resolve()
    raw = _loads(path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict) or set(raw) - (set(DEFAULTS) | {'model', 'log_dir'}):
        raise ValueError('unknown API agent fields; history/memory/retry fields have been removed')
    config = {**DEFAULTS, **raw}
    model = config.get('model')
    if not isinstance(model, str) or not model.strip() or model.startswith('REPLACE_'):
        raise ValueError('set model to the verified API model ID')
    endpoint = config['endpoint']
    if not isinstance(endpoint, str):
        raise ValueError('endpoint must be an API base URL')
    parsed = urlsplit(endpoint)
    if (parsed.scheme not in {'https', 'http'} or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ValueError('endpoint must be an API base URL without credentials/query/fragment')
    if parsed.port is not None and not 0 < parsed.port < 65536:
        raise ValueError('invalid endpoint port')
    if parsed.path.rstrip('/').endswith(('/responses', '/chat/completions')):
        raise ValueError('endpoint is the base URL, without the API operation suffix')
    config['endpoint'] = endpoint.rstrip('/')
    if config['api'] != 'responses':
        raise ValueError('single-upload conversation mode requires the Responses API')
    for key, choices in [('output_format', {'json_schema', 'json_object'}),
                         ('image_detail', {'auto', 'low', 'high'}), ('feedback', {'minimal', 'none'})]:
        if not isinstance(config[key], str) or config[key] not in choices:
            raise ValueError('invalid ' + key)
    if not isinstance(config['api_key_env'], str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', config['api_key_env']):
        raise ValueError('api_key_env must name an environment variable')
    value = config['max_output_tokens']
    if type(value) is not int or not 1 <= value <= 1000000:
        raise ValueError('invalid max_output_tokens')
    value = config['request_budget_seconds']
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 115:
        raise ValueError('invalid request budget')
    effort = config['reasoning_effort']
    if effort is not None and (not isinstance(effort, str) or not effort.strip()):
        raise ValueError('reasoning_effort must be null or a supported effort string')
    if not isinstance(config.get('log_dir'), str) or not config['log_dir'].strip():
        raise ValueError('log_dir is required')
    config['log_dir'] = str((path.parent / Path(config['log_dir']).expanduser()).resolve())
    return config


class APIError(RuntimeError):
    """Contains no raw server body, URL credentials or API key."""
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def post_json(url, body, api_key, timeout):
    """Injectable HTTP transport; no automatic request/image replay."""
    request = Request(url, data=_json(body).encode('utf-8'), method='POST', headers={
        'Content-Type': 'application/json', 'Authorization': 'Bearer ' + api_key})
    try:
        with urlopen(request, timeout=timeout) as response:
            value = _loads(response.read().decode('utf-8'))
            if not isinstance(value, dict):
                raise APIError('api_response_not_object')
            value['_http_request_id'] = response.headers.get('x-request-id')
            return value
    except HTTPError as exc:
        status = exc.code
        exc.close()
        raise APIError('http_' + str(status)) from None
    except (TimeoutError, socket.timeout):
        raise APIError('api_timeout') from None
    except URLError as exc:
        code = 'api_timeout' if isinstance(exc.reason, (TimeoutError, socket.timeout)) else 'api_connection_error'
        raise APIError(code) from None
    except (ValueError, UnicodeError):
        raise APIError('api_invalid_json') from None


def _strings(value):
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError('expected a list of public strings')
    return list(value)


def _initial_state(observation):
    # Only robot proprioception at sequence start, never target/room state.
    pose = observation.get('pose') or {}
    result = {'pose': {k: deepcopy(pose[k]) for k in ('position', 'orientation')
                      if isinstance(pose.get(k), list) and all(type(x) in (int, float) and math.isfinite(x) for x in pose[k])}}
    for key in ('yaw_degrees', 'pitch_degrees'):
        if type(pose.get(key)) in (int, float) and math.isfinite(pose[key]):
            result['pose'][key] = pose[key]
    if type(observation.get('holding_object')) is bool:
        result['holding_object'] = observation['holding_object']
    if isinstance(observation.get('held_object_name'), str):
        result['held_object_name'] = observation['held_object_name']
    return result


def _frame(observation):
    timing = observation.get('timing') or {}
    frame_id = timing.get('frame_id')
    if type(frame_id) is not int or frame_id < 1:
        raise ValueError('front image requires a positive observation frame_id')
    image = (observation.get('images') or {}).get('front')
    if not isinstance(image, dict) or image.get('encoding') != 'base64' or image.get('dtype') != 'uint8':
        raise ValueError('front RGB must be a serialized uint8 array')
    shape = image.get('shape')
    if (not isinstance(shape, list) or len(shape) != 3 or shape[2] not in (3, 4)
            or any(type(n) is not int or not 0 < n <= 4096 for n in shape[:2])):
        raise ValueError('invalid front RGB shape')
    pixels = base64.b64decode(image['data'], validate=True)
    rgb = np.frombuffer(pixels, dtype=np.uint8).reshape(shape)[..., :3]
    output = BytesIO()
    Image.fromarray(rgb).save(output, format='PNG')
    metadata = {'frame_id': frame_id, 'rgb_sha256': hashlib.sha256(pixels).hexdigest()}
    for key in ('captured_monotonic_s', 'scheduled_monotonic_s'):
        if type(timing.get(key)) in (int, float) and math.isfinite(timing[key]):
            metadata[key] = timing[key]
    return metadata, 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode('ascii')


def _schema(allowed):
    action = {'type': 'object', 'additionalProperties': False, 'properties': {
        'action': {'type': 'string', 'enum': allowed}, 'distance': {'type': ['number', 'null']},
        'angle_degrees': {'type': ['number', 'null']}, 'target': {'type': ['string', 'null']}},
        'required': ['action', 'distance', 'angle_degrees', 'target']}
    return {'type': 'object', 'additionalProperties': False, 'properties': {
        'action': {'anyOf': [action, {'type': 'null'}]}, 'answer': {'type': ['string', 'null']},
        'done': {'type': 'boolean'}}, 'required': ['action', 'answer', 'done']}


def _usage(value):
    if not isinstance(value, dict):
        return None
    result = {}
    for name in ('input_tokens', 'output_tokens', 'total_tokens'):
        count = value.get(name)
        if type(count) is int and count >= 0:
            result[name] = count
    for source, key, name in [('input_tokens_details', 'cached_tokens', 'cached_input_tokens'),
                              ('output_tokens_details', 'reasoning_tokens', 'reasoning_tokens')]:
        details = value.get(source)
        count = details.get(key) if isinstance(details, dict) else None
        if type(count) is int and count >= 0:
            result[name] = count
    return result or None


class OpenAIAPIAgent:
    """A sequence shares a response chain; local state tracks delivery, not memories."""
    def __init__(self, config, *, transport=post_json, clock=time.monotonic):
        self.config, self.transport, self.clock = deepcopy(config), transport, clock
        self.api_key = os.environ.get(config['api_key_env'])
        if not self.api_key:
            raise ValueError('missing API credential environment variable: ' + config['api_key_env'])
        self.instance_id = uuid.uuid4().hex
        self.log_dir = Path(config['log_dir']) / self.instance_id
        self.log_dir.mkdir(parents=True, exist_ok=False)
        self.config_sha256, self.prompt_sha256 = _digest(config), hashlib.sha256(SYSTEM.encode()).hexdigest()
        write_json(self.log_dir / 'config.json', config)
        (self.log_dir / 'prompt.txt').write_text(SYSTEM, encoding='utf-8')
        self.scene_id = self.run_id = None
        self.failed = False
        self._reset_chain()
        self.summary = {'schema_version': 2, 'instance_id': self.instance_id, 'client_status': 'ready',
            'input_policy': {**INPUT_POLICY, 'feedback': config['feedback']},
            'config_sha256': self.config_sha256, 'prompt_sha256': self.prompt_sha256,
            'attempts': 0, 'api_successes': 0, 'usage_missing_attempts': 0,
            'uploaded_images': 0, 'system_prompt_uploads': 0, 'initial_state_uploads': 0,
            'usage_totals': {}, 'model_ids': [], 'run_ids': [], 'latency_seconds_total': 0.0,
            'estimated_cost': None,
            'cost_note': 'Stored history may be billed again; prices/cache behavior unverified. Failed calls may be billed.'}
        self._save_summary()

    def _reset_chain(self):
        self.previous_response_id = None
        self.task_contract = None
        self.last_frame_id = None
        self.last_action_kind = None
        self.pending_sessions = []
        self.sent_introductions = set()

    def _save_summary(self):
        write_json(self.log_dir / 'summary.json', self.summary)

    def _append(self, name, value):
        with (self.log_dir / name).open('a', encoding='utf-8') as stream:
            stream.write(_json(value) + '\n')

    def _allowed(self, payload):
        allowed = payload.get('allowed_actions')
        if allowed is None:
            allowed = sorted(ACTION_TYPES)
            if payload['type'] == 'qa':
                allowed = [x for x in allowed if x not in {'pick', 'place', 'open', 'close'}] if payload.get('allow_navigation_actions') else ['no_op']
        if not isinstance(allowed, list) or not allowed or any(not isinstance(x, str) or x not in ACTION_TYPES for x in allowed):
            raise APIError('invalid_allowed_actions')
        return sorted(set(allowed))

    def _feedback(self, observation):
        if self.config['feedback'] == 'none' or self.last_action_kind is None:
            return {}
        feedback = observation.get('action_feedback') or {}
        if self.last_action_kind in TRANSLATIONS and type(feedback.get('collision')) is bool:
            return {'collision': feedback['collision']}
        if self.last_action_kind == 'reset' and type(observation.get('reset_applied')) is bool:
            return {'success': observation['reset_applied']}
        if self.last_action_kind in EXECUTION_FLAGS and type(feedback.get('success')) is bool:
            return {'success': feedback['success']}
        return {}

    def _input(self, payload, allowed):
        observation = payload.get('observation') or {}
        metadata, image = _frame(observation)
        if self.last_frame_id is not None and metadata['frame_id'] <= self.last_frame_id:
            raise APIError('observation_frame_reused')
        first = self.previous_response_id is None
        delta, introduction_hashes = {}, set(self.sent_introductions)
        sessions = []
        for pending in self.pending_sessions:
            session = {k: deepcopy(v) for k, v in pending.items() if k != 'scene_briefing'}
            new = []
            for text in pending['scene_briefing']:
                digest = _digest(text)
                if digest not in introduction_hashes:
                    new.append(text)
                    introduction_hashes.add(digest)
            if new:
                session['scene_briefing'] = new
            sessions.append(session)
        if sessions:
            delta['sessions'] = sessions
        contract = {k: payload[k] for k in ('type', 'instruction', 'question', 'answer_type', 'response_mode', 'current_time')
                    if isinstance(payload.get(k), str)}
        contract['allowed_actions'] = allowed
        task_key = (payload.get('scene_id'), payload.get('session_id'), payload.get('task_id'))
        contract_id = _digest([task_key, contract])
        if contract_id != self.task_contract:
            delta['task'] = contract
        introductions = []
        for text in _strings(payload.get('name_introductions', [])):
            digest = _digest(text)
            if digest not in introduction_hashes:
                introductions.append(text)
                introduction_hashes.add(digest)
        if introductions:
            delta['name_introductions'] = introductions
        if first:
            delta['initial_state'] = _initial_state(observation)
        flags = self._feedback(observation)
        if flags:
            delta['feedback'] = flags
        content = []
        if delta:
            content.append({'type': 'input_text', 'text': _json(delta)})
        content.append({'type': 'input_image', 'image_url': image, 'detail': self.config['image_detail']})
        items = []
        if first:
            # instructions= is intentionally absent: it is not an inherited
            # conversation item. A developer message is sent only at chain start.
            items.append({'role': 'developer', 'content': [{'type': 'input_text', 'text': SYSTEM}]})
        items.append({'role': 'user', 'content': content})
        delivery = {'contract': contract_id, 'introductions': introduction_hashes,
                    'frame_id': metadata['frame_id']}
        return items, metadata, delivery

    def _body(self, items, allowed):
        config = self.config
        fmt = {'type': config['output_format']}
        if config['output_format'] == 'json_schema':
            fmt.update(name='benchmark_action', strict=True, schema=_schema(allowed))
        body = {'model': config['model'], 'store': True, 'truncation': 'disabled',
                'input': items, 'text': {'format': fmt}, 'max_output_tokens': config['max_output_tokens']}
        if self.previous_response_id is not None:
            body['previous_response_id'] = self.previous_response_id
        if config['reasoning_effort'] is not None:
            body['reasoning'] = {'effort': config['reasoning_effort']}
        return body

    def _invoke(self, body, deadline):
        remaining = deadline - self.clock()
        if remaining <= 0:
            raise APIError('api_deadline_exceeded')
        result = Queue(maxsize=1)
        def send():
            try:
                result.put((True, self.transport(self.config['endpoint'] + '/responses', body, self.api_key, remaining)))
            except Exception as exc:
                result.put((False, exc))
        # A socket timeout is not an end-to-end deadline. On expiry fail closed;
        # never replay an image or use a late response from the daemon transport.
        threading.Thread(target=send, name='api-http', daemon=True).start()
        try:
            ok, value = result.get(timeout=remaining)
        except Empty:
            raise APIError('api_deadline_exceeded') from None
        if self.clock() >= deadline:
            raise APIError('api_deadline_exceeded')
        if not ok:
            raise value if isinstance(value, APIError) else APIError('api_transport_error')
        return value

    def _decode(self, result):
        if not isinstance(result, dict):
            raise APIError('api_response_not_object')
        if result.get('status') != 'completed' or result.get('error'):
            raise APIError('api_response_incomplete')
        response_id = result.get('id')
        if not isinstance(response_id, str) or not response_id or response_id == self.previous_response_id:
            raise APIError('api_response_identity')
        parts = [item for output in result.get('output', []) if output.get('type') == 'message'
                 for item in output.get('content', [])]
        if any(p.get('type') == 'refusal' for p in parts):
            raise APIError('api_refusal')
        return _loads(''.join(p['text'] for p in parts if p.get('type') == 'output_text'))

    def _validate(self, payload, value, allowed):
        if not isinstance(value, dict) or set(value) != {'action', 'answer', 'done'}:
            raise APIError('model_output_schema')
        if type(value['done']) is not bool or (value['answer'] is not None and not isinstance(value['answer'], str)):
            raise APIError('model_output_schema')
        response = {'done': value['done']}
        action = value['action']
        if action is not None:
            if not isinstance(action, dict) or set(action) != {'action', 'distance', 'angle_degrees', 'target'}:
                raise APIError('model_action_schema')
            name = action['action']
            if name not in allowed:
                raise APIError('model_forbidden_action')
            action = {k: v for k, v in action.items() if v is not None}
            for key, maximum, kinds in [('distance', 10, TRANSLATIONS),
                                         ('angle_degrees', 720, {'turn_left', 'turn_right', 'look_up', 'look_down'})]:
                if key in action and (name not in kinds or type(action[key]) not in (int, float)
                                      or not math.isfinite(action[key]) or not 0 < action[key] <= maximum):
                    raise APIError('model_action_argument')
            if 'target' in action and (name not in {'pick', 'open', 'close'} or not isinstance(action['target'], str) or not action['target'].strip()):
                raise APIError('model_action_target')
            if name == 'pick' and not action.get('target'):
                raise APIError('model_action_target')
            response['action'] = action
        elif payload['type'] == 'step':
            raise APIError('model_missing_action')
        if value['answer'] is not None:
            response['answer'] = value['answer']
        if payload.get('response_mode') == 'actions' and 'answer' in response:
            raise APIError('model_unrequested_answer')
        if (payload['type'] == 'qa' or payload.get('response_mode') == 'actions_and_answer') and value['done'] and 'answer' not in response:
            raise APIError('model_missing_answer')
        return validate_response(payload['type'], response)

    def _record(self, record):
        self._append('calls.jsonl', record)
        self.summary['attempts'] += 1
        self.summary['api_successes'] += int(record['status'] == 'ok')
        self.summary['uploaded_images'] += len(record['input_frames'])
        self.summary['system_prompt_uploads'] += int(record['system_prompt_sent'])
        self.summary['initial_state_uploads'] += int(record['initial_state_sent'])
        self.summary['latency_seconds_total'] += record['elapsed_seconds']
        usage = record.get('usage') or {}
        self.summary['usage_missing_attempts'] += int(not {'input_tokens', 'output_tokens'} <= usage.keys())
        for key, value in usage.items():
            totals = self.summary['usage_totals']
            totals[key] = totals.get(key, 0) + value
        model = record.get('model')
        if model and model not in self.summary['model_ids']:
            self.summary['model_ids'].append(model)
        self._save_summary()

    def handle(self, payload):
        """Protocol state is local; the API owns the conversation and its memories."""
        deadline = self.clock() + self.config['request_budget_seconds']
        identity = {k: payload.get(k) for k in ('scene_id', 'session_id', 'task_id', 'request_id')}
        try:
            if self.failed:
                raise APIError('agent_already_failed')
            sequence = payload.get('sequence_context') or {}
            run_id = sequence.get('run_id')
            if payload['type'] == 'session_start':
                if self.scene_id != payload.get('scene_id') or sequence.get('reset_memory'):
                    self._reset_chain()
                elif run_id != self.run_id:
                    raise APIError('sequence_reset_missing')
                self.scene_id, self.run_id = payload.get('scene_id'), run_id
                pending = {'scene_briefing': _strings(payload.get('scene_briefing', []))}
                if isinstance(payload.get('instruction'), str):
                    pending['instruction'] = payload['instruction']
                context = payload.get('time_context') or {}
                pending['time_context'] = {k: context[k] for k in ('kind', 'starts_at') if isinstance(context.get(k), str)}
                self.pending_sessions.append(pending)
                response = {}
                if sequence.get('reset_memory'):
                    if not isinstance(run_id, str) or not run_id:
                        raise APIError('invalid_sequence_identity')
                    response['sequence_reset_ack'] = run_id
                self._append('events.jsonl', {'event': 'session_start', **identity, 'run_id': run_id,
                                            'previous_response_id': self.previous_response_id})
                self.summary['client_status'] = 'active'
                if run_id is not None and run_id not in self.summary['run_ids']:
                    self.summary['run_ids'].append(run_id)
                self._save_summary()
                return response
            if payload['type'] not in {'step', 'qa'} or self.scene_id != payload.get('scene_id') or run_id != self.run_id:
                raise APIError('request_sequence_mismatch')
            allowed = self._allowed(payload)
            items, frame, delivery = self._input(payload, allowed)
            body = self._body(items, allowed)
            first = self.previous_response_id is None
            record = {**identity, 'run_id': self.run_id, 'instance_id': self.instance_id,
                'attempt': 1, 'api': 'responses', 'started_monotonic_s': self.clock(),
                'previous_response_id': self.previous_response_id, 'input_sha256': _digest(body),
                'config_sha256': self.config_sha256, 'prompt_sha256': self.prompt_sha256,
                'system_prompt_sent': first, 'initial_state_sent': first, 'input_frames': [frame],
                'observation_frame_id': frame['frame_id'],
                'uploaded_text_chars': sum(len(c['text']) for item in items for c in item['content'] if c['type'] == 'input_text')}
            result, response, error = None, None, None
            try:
                result = self._invoke(body, deadline)
                response = self._validate(payload, self._decode(result), allowed)
                if self.clock() >= deadline:
                    raise APIError('api_deadline_exceeded')
            except APIError as exc:
                error = exc
            except Exception:
                error = APIError('api_invalid_output')
            record.update(status='error' if error else 'ok', error_code=error.code if error else None,
                          elapsed_seconds=max(0, self.clock() - record['started_monotonic_s']))
            if isinstance(result, dict):
                record.update({name: result.get(key) if isinstance(result.get(key), str) else None
                               for name, key in [('model', 'model'), ('response_id', 'id'), ('http_request_id', '_http_request_id')]})
                record['usage'] = _usage(result.get('usage'))
            if response is not None and error is None:
                record['response'] = response
            self._record(record)
            if error is not None:
                raise error
            self.previous_response_id = result['id']
            self.last_frame_id = delivery['frame_id']
            self.task_contract = delivery['contract']
            self.sent_introductions = delivery['introductions']
            self.pending_sessions.clear()
            action = response.get('action')
            self.last_action_kind = action.get('action') if isinstance(action, dict) else action
            return response
        except Exception as exc:
            code = exc.code if isinstance(exc, APIError) else 'agent_invalid_input'
            self.failed = True
            self.summary.update(client_status='failed', error_code=code)
            self._append('events.jsonl', {'event': 'integration_error', **identity, 'run_id': self.run_id, 'reason_code': code})
            self._save_summary()
            return {'action': 'no_op', 'done': True, 'agent_status': 'integration_error', 'reason_code': code}

    def close(self):
        self.summary['client_status'] = 'failed' if self.failed else 'closed'
        self._save_summary()
