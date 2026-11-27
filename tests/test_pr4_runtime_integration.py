"""Runtime compatibility between standalone resources and the shared execution API."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from roboodyssey.data.sources import SceneSpec
from roboodyssey.scenes.robot_body import (
    default_robot_identity, scene_robot_identity, validate_scene_robot,
)


class PR4RuntimeIntegrationTests(unittest.TestCase):
    def test_standalone_scene_identity_and_navigation_are_checked(self):
        identity = default_robot_identity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / 'scene_resources.json'
            manifest.write_text(json.dumps({'robot_profile': identity}))
            navigation = root / 'navigation.npz'
            np.savez(navigation, robot_radius=identity['robot_radius_m'],
                     robot_profile_json=json.dumps(identity))
            scene = SceneSpec('scene', root / 'scene.usda', {'navigation': navigation}, manifest)
            dataset = SimpleNamespace(_requests=SimpleNamespace(scene=scene))
            self.assertEqual(scene_robot_identity(scene), identity)
            self.assertEqual(scene_robot_identity(dataset), identity)
            validate_scene_robot(dataset, identity)
            with self.assertRaisesRegex(ValueError, 'does not match'):
                validate_scene_robot(dataset, dict(identity, robot_radius_m=.1))
            np.savez(navigation, robot_radius=.1)
            with self.assertRaisesRegex(ValueError, 'navigation robot_radius'):
                validate_scene_robot(dataset, identity)
            manifest.write_text('{}')
            self.assertIsNone(scene_robot_identity(dataset))
            validate_scene_robot(dataset, identity)
            with self.assertRaisesRegex(ValueError, 'requires a declared'):
                validate_scene_robot(dataset, identity, require_declared=True)

    def test_shared_environment_factory_accepts_both_loaded_dataset_contracts(self):
        from roboodyssey.data.sources import ExternalDataset
        from roboodyssey.scenes.dataset_loader import BenchmarkDataset
        from roboodyssey.runtime.policy import make_isaac_env_factory
        app, config = object(), object()
        for dataset_class in (BenchmarkDataset, ExternalDataset):
            with self.subTest(dataset_class=dataset_class.__name__):
                dataset = dataset_class.__new__(dataset_class)
                with patch.object(BenchmarkDataset, '__init__', side_effect=AssertionError('reloading input')), \
                     patch('roboodyssey.scenes.isaac_backend.make_env_factory') as factory:
                    result = make_isaac_env_factory(dataset, app, config=config)
                factory.assert_called_once_with(dataset, app, camera_config=config)
                self.assertIs(result, factory.return_value)

    def test_external_run_records_verified_provenance_before_starting_simulator(self):
        from roboodyssey.artifacts.layout import sha256_file
        from roboodyssey.scenes.dataset_loader import BenchmarkDataset
        from roboodyssey.launch.provenance import capture_provenance
        from roboodyssey.run import main
        from tests.frozen_data import fixture_v3, goal, select_scenario, task, write_dataset

        class SimulatorReached(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compiler, inputs, plan = fixture_v3([task('visit', goal('cabinet_dark'))])
            published_root = root / 'published'
            write_dataset(published_root, select_scenario(compiler, plan), compiler, inputs)
            published = BenchmarkDataset(published_root)
            scene_id = published.scene_ids()[0]
            record = published.sessions(scene_id)[0]
            scene, session = root / 'scene', root / 'session'
            scene.mkdir()
            session.mkdir()
            (scene / 'scene.usda').write_text('#usda 1.0\n')
            scene_paths = {}
            for name in ('scene_metainfo', 'navigation', 'manipulation', 'environment_setup',
                         'initial_state', 'runtime_scene_cache'):
                source = (published._requests.version.path(name) if name == 'initial_state'
                          else published._requests.path(name))
                destination = scene / (name + source.suffix)
                shutil.copy2(source, destination)
                scene_paths[name] = destination.name
            (scene / 'scene_resources.json').write_text(json.dumps({
                'schema_version': 1, 'kind': 'roboodyssey_scene', 'scene_id': scene_id,
                'asset': 'scene.usda', 'asset_sha256': sha256_file(scene / 'scene.usda'),
                'resources': scene_paths,
                'sha256': {name: sha256_file(scene / path) for name, path in scene_paths.items()},
            }))
            shutil.copy2(record.path / 'session.json', session / 'session.json')
            session_paths = {}
            for name in ('ground_truth', 'templates', 'mutations'):
                destination = session / (name + '.json')
                shutil.copy2(record.private_paths[name], destination)
                session_paths[name] = destination.name
            (session / 'session_resources.json').write_text(json.dumps({
                'schema_version': 1, 'kind': 'roboodyssey_session', 'scene_id': scene_id,
                'sequence_id': record.metadata['sequence_id'], 'session_id': record.session_id,
                'session': 'session.json', 'session_sha256': sha256_file(session / 'session.json'),
                'resources': session_paths,
                'sha256': {name: sha256_file(session / path) for name, path in session_paths.items()},
            }))
            output = root / 'run'
            with patch('roboodyssey.launch.provenance.repository', return_value={}), \
                 patch('roboodyssey.launch.provenance.distributions', return_value=[]), \
                 patch('roboodyssey.scenes.isaac_backend.IsaacBackend', side_effect=SimulatorReached) as backend, \
                 redirect_stdout(StringIO()):
                with self.assertRaises(SimulatorReached):
                    main(['--scene-source', str(scene), '--session-source', str(session),
                          '--agent-cmd', 'python -m external_agent', '--run-directory', str(output)])
            backend.assert_called_once()
            provenance = json.loads((output / 'provenance.json').read_text())['dataset']
            self.assertEqual(provenance['kind'], 'external_scene_session')
            self.assertEqual(provenance['scene']['manifest']['sha256'], sha256_file(scene / 'scene_resources.json'))
            self.assertEqual(provenance['sessions'][0]['session_sha256'], sha256_file(session / 'session.json'))
            with patch('roboodyssey.launch.provenance.repository', return_value={}), \
                 patch('roboodyssey.launch.provenance.distributions', return_value=[]):
                frozen = capture_provenance(root / 'frozen.json', dataset_root=published_root)
                self.assertIn('scene_manifest.json', frozen['dataset']['manifests'])
                # Resource support must not turn invalid frozen paths into a fallback.
                with self.assertRaisesRegex(ValueError, 'no scene_manifest'):
                    capture_provenance(root / 'invalid.json', dataset_root=scene)
                (session / 'session.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                    capture_provenance(root / 'tampered.json', scene_source=scene, session_sources=[session])

    def test_real_execution_requires_explicit_external_agent(self):
        from roboodyssey.run import main
        with redirect_stderr(StringIO()) as errors, patch('roboodyssey.run.BenchmarkDataset') as dataset:
            with self.assertRaises(SystemExit) as caught:
                main(['--dataset', 'unused'])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn('real execution requires --agent-cmd', errors.getvalue())
        dataset.assert_not_called()

    def test_sensor_plugin_and_stretch_body_are_both_initialized(self):
        from roboodyssey.scenes.isaac_env import EnvConfig, IsaacVLNEnv
        with tempfile.TemporaryDirectory() as temporary:
            asset = Path(temporary) / 'scene.usda'
            asset.write_text('#usda 1.0\n')
            camera, stage, contract = Mock(), Mock(), object()
            bridge_factory = Mock()
            modules = {'omni': SimpleNamespace(), 'omni.usd': SimpleNamespace(),
                       'isaacsim.sensors.camera': SimpleNamespace(Camera=Mock())}
            with patch.dict('sys.modules', modules), \
                 patch('roboodyssey.scenes.camera.open_scene_stage', return_value=(stage, Mock())), \
                 patch('roboodyssey.scenes.camera.create_camera', return_value=camera), \
                 patch.object(IsaacVLNEnv, '_configure_camera_intrinsics'), \
                 patch.object(IsaacVLNEnv, '_configure_scene_rendering'), \
                 patch('roboodyssey.scenes.robot_body.KinematicRobotBody') as body:
                config = EnvConfig()
                env = IsaacVLNEnv(asset, config, sensor_bridge_factory=bridge_factory,
                                  sensor_contract=contract)
            body.assert_called_once_with(stage, config.robot_body)
            bridge_factory.assert_called_once_with(camera, config.resolution, contract)
            self.assertIs(env._robot_body, body.return_value)
            self.assertIs(env._sensor_bridge, bridge_factory.return_value)

    def test_in_process_policy_remains_usable_after_protocol_migration(self):
        from roboodyssey.runtime.policy import InProcessPolicyAgent
        from roboodyssey.runtime.session_runner import SessionRunner
        class Policy:
            def decide(self, task, observation, context):
                return {'action': 'no_op', 'done': True}
        policy = InProcessPolicyAgent(Policy())
        with tempfile.TemporaryDirectory() as temporary:
            runner = SessionRunner(Path(temporary), policy, max_steps=1)
            self.assertIs(runner.agent, policy)
        policy.set_context({'scene_id': 's', 'session_id': 'session', 'task': {},
                            'raw_public_observation': {'timing': {'frame_id': 3}}})
        response = policy.request({'type': 'step', 'request_id': 'req'})
        self.assertEqual(response['observation_frame_id'], 3)
        self.assertEqual(response['action'], 'no_op')


if __name__ == '__main__':
    unittest.main()
