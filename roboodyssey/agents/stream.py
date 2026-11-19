"""Agent-side JSONL receiver: continue reading frames while a worker infers.

This is an optional agent implementation, not benchmark observation filtering.
Every incoming frame reaches on_observation immediately. The supplied examples
keep a configurable visual history for their next inference; no actions are
queued for observation notifications.
"""
from collections import deque
import json
from queue import Queue
import sys
import threading


def serve(handler, *, history_frames=8, decode=None, on_observation=None,
          input_stream=None, output_stream=None):
    """Run one inference worker and a separate continuously reading receiver.

    handler gets a stable observation_history snapshot with earlier received
    frames. on_observation is an optional fast callback on the reader thread.
    A custom streaming model may implement its own receiver instead.
    """
    if history_frames < 0:
        raise ValueError('history_frames must be nonnegative')
    incoming, outgoing = input_stream or sys.stdin, output_stream or sys.stdout
    history = deque(maxlen=history_frames)
    work = Queue(maxsize=1)
    busy = threading.Event()
    scene_id = None
    task_key = None

    def worker():
        while True:
            payload = work.get()
            if payload is None:
                return
            try:
                response = dict(handler(payload))
            except Exception as exc:
                print(f'agent integration error: {exc}', file=sys.stderr, flush=True)
                response = {'action': 'no_op', 'done': True, 'agent_status': 'integration_error',
                            'reason_code': 'agent_handler_error', 'error': str(exc)}
            if 'request_id' in payload:
                response['request_id'] = payload['request_id']
            if payload.get('type') == 'session_start' and payload.get('observation_stream'):
                response['observation_stream_ack'] = True
            timing = (payload.get('observation') or {}).get('timing', {})
            if 'frame_id' in timing:
                # This worker consumes a snapshot taken at request receipt.
                # Later frames cannot retroactively justify the current answer.
                response['observation_frame_id'] = timing['frame_id']
            try:
                line = json.dumps(response, ensure_ascii=False, allow_nan=False)
            except (TypeError, ValueError) as exc:
                line = json.dumps({'request_id': payload.get('request_id'),
                    'observation_frame_id': timing.get('frame_id'),
                    'action': 'no_op', 'agent_status': 'integration_error',
                    'reason_code': 'non_json_response', 'error': str(exc)})
            busy.clear()
            outgoing.write(line + '\n')
            outgoing.flush()

    thread = threading.Thread(target=worker, name='agent-inference', daemon=True)
    thread.start()
    try:
        for line in incoming:
            if not line.strip():
                continue
            payload = json.loads(line)
            if decode:
                payload = decode(payload)
            kind = payload.get('type')
            if kind not in {'session_start', 'step', 'qa', 'observation'}:
                raise ValueError('unknown request type: ' + str(kind))
            if kind == 'session_start':
                if payload.get('scene_id') != scene_id or payload.get('sequence_context', {}).get('reset_memory'):
                    history.clear()
                scene_id, task_key = payload.get('scene_id'), None
                # Legacy introductions are delivered as a session_views batch;
                # preserve them in Agent memory for subsequent model calls.
                for view in payload.get('session_views') or []:
                    if view.get('observation'):
                        history.append(view['observation'])
            key = (payload.get('scene_id'), payload.get('session_id'), payload.get('task_id'))
            if kind == 'observation' and key != task_key:
                raise ValueError('observation belongs to a different task')
            if kind != 'observation':
                if busy.is_set():
                    raise ValueError('only one action request may be in flight')
                if kind in {'step', 'qa'}:
                    task_key = key
                payload['observation_history'] = list(history)
            observation = payload.get('observation')
            if observation is not None:
                if on_observation:
                    on_observation(payload)
                history.append(observation)
            if kind != 'observation':
                busy.set()
                work.put_nowait(payload)
    finally:
        work.put(None)
        thread.join()
