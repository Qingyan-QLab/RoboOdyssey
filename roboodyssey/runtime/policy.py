"""Privileged in-process policies using the same execution and success rules.

This adapter is opt-in and never used for a public subprocess model. A policy
is trusted Python code; context is private audit information, not model input.
"""
from copy import deepcopy
from pathlib import Path

from ..agents.protocol import validate_response


class InProcessPolicyAgent:
    def __init__(self, policy):
        self.policy = policy
        self._context = None
        self.permission_profile = getattr(policy, 'permission_profile', 'private_state_oracle')
        if self.permission_profile not in {'private_state_oracle', 'perception_policy'}:
            raise ValueError('unknown policy permission_profile')

    def set_context(self, context):
        self._context = context

    def send_observation(self, payload):
        # Synchronous policies consume the captured request frame. They do not
        # advertise a concurrently updated image stream during decide().
        return None

    def request(self, payload, on_wait=None):
        if payload['type'] == 'session_start':
            lifecycle = payload['sequence_context']
            if lifecycle.get('reset_memory'):
                reset = getattr(self.policy, 'reset', None)
                if reset:
                    reset()
            return {'sequence_reset_ack': lifecycle['run_id'], 'observation_stream_ack': True}
        if payload['type'] != 'step' or self._context is None:
            raise ValueError('policy requires an actual task context')
        context = self._context
        task = deepcopy(context['task'])
        observation = deepcopy(context.get('raw_public_observation', payload.get('observation', {})))
        if self.permission_profile == 'perception_policy':
            context = {key: context[key] for key in ('scene_id', 'session_id')}
            context['feedback'] = deepcopy(observation.get('action_feedback'))
        else:
            context = {key: value for key, value in context.items() if key != 'raw_public_observation'}
        response = self.policy.decide(task, observation, context)
        if not isinstance(response, dict):
            raise ValueError('policy response must be a dictionary')
        response = dict(response)
        response.setdefault('request_id', payload['request_id'])
        response.setdefault('observation_frame_id', observation['timing']['frame_id'])
        response.setdefault('action_source', 'oracle' if self.permission_profile == 'private_state_oracle' else 'perception_policy')
        return validate_response('step', response)


def make_isaac_env_factory(dataset, app, config=None):
    """Use benchmark run's scene, initial pose and lighting in an existing app."""
    from ..scenes.isaac_backend import make_env_factory
    from ..scenes.dataset_loader import BenchmarkDataset
    if not isinstance(dataset, BenchmarkDataset):
        dataset = BenchmarkDataset(dataset)
    return make_env_factory(dataset, app, camera_config=config)


def run_sequence(sequence, policy, output, *, env_factory, scene=None,
                 max_steps=2000, capture=True, observation_clock=None):
    """Run one whole sequence; write standard benchmark records plus raw frames.

    env_factory(scene_id, canonical_state, first_session_id) returns an
    environment implementing the Benchmark runtime methods. The common runner
    owns its lifecycle. This function always creates a fresh result directory.
    """
    from ..scenes.dataset_loader import BenchmarkDataset
    from ..artifacts.layout import initialize_run, write_json, update_status, sha256_file
    from ..evaluation.report import write_summary
    from .session_runner import SessionRunner
    from .capture import ObservationArchive

    agent = InProcessPolicyAgent(policy)
    dataset = BenchmarkDataset(sequence, scene=scene)
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    initialize_run(root, dataset)
    from ..evaluation.task_metrics import METRIC_POLICY
    from .observations import OBSERVATION_POLICY
    config = {'schema_version': 1, 'official': False,
        'execution_profile': agent.permission_profile, 'max_steps': max_steps,
        'capture_original_observations': bool(capture), 'sequence_identity': dataset.sequence_identity(),
        'dataset_manifest_sha256': sha256_file(dataset.root/'dataset_manifest.json'),
        'metric_policy': deepcopy(METRIC_POLICY), 'observation_policy': deepcopy(OBSERVATION_POLICY),
        'environment': {'status': 'not_created'},
        'policy_class': type(policy).__module__+'.'+type(policy).__qualname__}
    write_json(root/'run_config.json', config)

    def recorded_factory(*args):
        import dataclasses
        import inspect
        from .session_runner import _jsonable
        from ..scenes.isaac_env import IsaacVLNEnv, AVATAR_INTERACTION
        env = env_factory(*args)
        try:
            source = inspect.getsourcefile(type(env))
            details = {'status': 'created', 'class': type(env).__module__+'.'+type(env).__qualname__,
                       'backend': 'isaac_sim' if isinstance(env, IsaacVLNEnv) else 'custom_or_test',
                       'source_sha256': sha256_file(source) if source and Path(source).is_file() else None}
            env_config = getattr(env, 'config', None)
            if dataclasses.is_dataclass(env_config):
                def paths(value):
                    if isinstance(value, Path):
                        return str(value)
                    if isinstance(value, dict):
                        return {key: paths(item) for key, item in value.items()}
                    if isinstance(value, (list, tuple)):
                        return [paths(item) for item in value]
                    return value
                details['config'] = _jsonable(paths(dataclasses.asdict(env_config)))
            if isinstance(env, IsaacVLNEnv):
                from ..scenes.robot_body import robot_identity_for_config, validate_scene_robot
                details['robot_identity'] = robot_identity_for_config(env.config)
                validate_scene_robot(dataset, details['robot_identity'])
                details['interaction_model'] = deepcopy(AVATAR_INTERACTION)
                import importlib.metadata
                details['sdk_packages'] = {name: dist.version for name in ('isaacsim', 'isaacsim-kernel')
                                          for dist in importlib.metadata.distributions()
                                          if dist.metadata.get('Name') == name}
            config['environment'] = details
            write_json(root/'run_config.json', config)
        except BaseException:
            env.close()
            raise
        return env
    runner = SessionRunner(root, agent, max_steps=max_steps, observation_clock=observation_clock,
                           observation_archive=ObservationArchive(root/'observations') if capture else None)
    update_status(root, phase='running', execution='running')
    try:
        scene_id = dataset.manifest['scene_id']
        runner.run_scene(scene_id, dataset.sessions(scene_id), dataset.canonical_state(scene_id), recorded_factory)
        summary = write_summary(dataset, root)
        update_status(root, phase='completed' if summary['evaluation_valid'] else 'failed',
                      execution='completed', summary='completed', cleanup='completed')
        return summary
    except BaseException as exc:
        write_summary(dataset, root)
        update_status(root, phase='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                      execution='failed', summary='completed', error=f'{type(exc).__name__}: {exc}')
        raise
