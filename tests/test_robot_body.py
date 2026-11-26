"""Robot asset integrity, launch identity, and avatar/camera separation on CPU."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from roboodyssey.scenes.robot_body import ROBOT_NAMES, load_robot_profile, resolve_robot
from roboodyssey.scenes.isaac_env import EnvConfig, IsaacVLNEnv


class RobotBodyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.asset = self.root / 'usd' / 'body.usdc'
        self.asset.parent.mkdir()
        self.asset.write_bytes(b'CPU fixture; no USD parser is imported')
        self.texture = self.asset.parent / 'texture.png'
        self.texture.write_bytes(b'texture fixture')
        self.path = self.root / 'profile.json'
        self.data = {'schema_version': 1, 'model_id': 'example', 'mode': 'kinematic_avatar',
                     'asset': 'usd/body.usdc', 'robot_radius_m': .3, 'camera_height_m': 1.5,
                     'base_offset_z_m': .08, 'asset_files': [
                         {'path': str(p.relative_to(self.root)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                         for p in (self.asset, self.texture)]}
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.data))

    def test_texture_mutation_is_rejected_even_when_root_asset_is_unchanged(self):
        first = load_robot_profile(self.path)
        self.assertEqual(first.robot_radius_m, .3)
        self.texture.write_bytes(b'modified')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            load_robot_profile(self.path)

    def named_profile(self, name):
        folder = self.root / 'profiles'
        folder.mkdir(exist_ok=True)
        data = {**self.data, 'model_id': name, 'asset': '../usd/body.usdc',
                'asset_files': [{**entry, 'path': '../' + entry['path']} for entry in self.data['asset_files']]}
        (folder / (name + '.json')).write_text(json.dumps(data))

    def test_stretch_default_is_bundled_and_explicit_cylinder_remains_asset_free(self):
        from roboodyssey.scenes.robot_body import default_robot_identity
        default = resolve_robot()
        self.assertEqual(default.name, 'stretch3')
        self.assertEqual(default.identity(), default_robot_identity())
        self.assertEqual(default.body, EnvConfig().robot_body)
        self.assertEqual(default.identity()['robot_radius_m'], EnvConfig().robot_radius)
        with patch.dict('os.environ', {'BENCHMARK_ROBOT_ASSETS': str(self.root/'missing')}):
            explicit = resolve_robot('cylinder')
            # The canonical body used by direct EnvConfig never depends on an environment override.
            self.assertEqual(EnvConfig().robot_body.model_id, 'stretch3')
        self.assertIsNone(explicit.body)
        self.assertEqual(explicit.identity()['robot_radius_m'], EnvConfig(robot_body=None).robot_radius)
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            EnvConfig(robot_radius=.01)

    def test_each_named_robot_resolves_and_sets_both_runtime_parameters(self):
        for name in ROBOT_NAMES[1:]:
            with self.subTest(name=name):
                self.named_profile(name)
                selected = resolve_robot(name, assets=self.root)
                self.assertEqual(selected.body.model_id, name)
                config = EnvConfig(robot_body=selected.body)
                self.assertEqual(config.robot_radius, self.data['robot_radius_m'])
                self.assertEqual(config.camera_height, self.data['camera_height_m'])
                self.assertEqual(config.move_step, .25)
                self.assertEqual(config.turn_step_degrees, 15.)

    def test_explicit_assets_override_environment_and_wrong_model_is_rejected(self):
        self.named_profile('stretch3')
        with patch.dict('os.environ', {'BENCHMARK_ROBOT_ASSETS': str(self.root)}):
            self.assertEqual(resolve_robot('stretch3').body.model_id, 'stretch3')
            with self.assertRaises(FileNotFoundError):
                resolve_robot('stretch3', assets=self.root/'missing')
        path = self.root/'profiles'/'stretch3.json'
        data = json.loads(path.read_text()); data['model_id'] = 'go2'
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'does not match'):
            resolve_robot('stretch3', assets=self.root)

    def test_ambiguous_selection_and_missing_assets_do_not_fall_back_to_cylinder(self):
        for options in ({'robot': 'stretch3', 'profile': self.path},
                        {'profile': self.path, 'assets': self.root}, {'robot': 'unknown'}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                resolve_robot(**options)
        with self.assertRaises(FileNotFoundError):
            resolve_robot('go2', assets=self.root)

    def test_list_and_check_include_cylinder_and_report_corrupt_dependencies(self):
        from roboodyssey.cli import main
        for name in ROBOT_NAMES[1:]: self.named_profile(name)
        with redirect_stdout(StringIO()) as output:
            self.assertEqual(main(['robots', '--robot-assets', str(self.root), '--check']), 0)
        self.assertEqual([r['robot'] for r in json.loads(output.getvalue())['robots']], list(ROBOT_NAMES))
        self.texture.write_bytes(b'broken')
        with redirect_stdout(StringIO()) as output:
            self.assertEqual(main(['robots', '--robot-assets', str(self.root), '--check']), 1)
        self.assertEqual(json.loads(output.getvalue())['robots'][1]['status'], 'error')

    def test_relocation_preserves_evaluation_identity(self):
        import shutil
        first = load_robot_profile(self.path)
        moved = self.root / 'moved'
        moved.mkdir()
        shutil.copytree(self.asset.parent, moved / 'usd')
        shutil.copy(self.path, moved / 'profile.json')
        second = load_robot_profile(moved / 'profile.json')
        self.assertNotEqual(first.asset, second.asset)
        self.assertEqual(first.identity(), second.identity())

    def test_missing_asset_and_invalid_parameters_fail_before_simulation(self):
        for field, value in [('robot_radius_m', 0), ('robot_radius_m', float('nan')),
                             ('camera_height_m', -1), ('base_offset_z_m', float('inf')),
                             ('mode', 'dynamic_robot')]:
            with self.subTest(field=field, value=value):
                data = dict(self.data, **{field: value})
                self.path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    load_robot_profile(self.path)
        self.save()
        self.asset.unlink()
        with self.assertRaises(FileNotFoundError):
            load_robot_profile(self.path)

    def test_camera_pitch_does_not_tilt_the_body_and_floor_height_is_preserved(self):
        body = load_robot_profile(self.path)
        self.assertEqual(body.body_position([2, 3, 2.5]), (2, 3, 1.08))
        env = IsaacVLNEnv.__new__(IsaacVLNEnv)
        env.camera = SimpleNamespace(set_world_pose=Mock())
        env._robot_body = SimpleNamespace(sync=Mock())
        env._agent_position = np.array([2., 3., 2.5])
        env._yaw, env._pitch = .7, 0.
        env._author_camera_pose()
        first_orientation = env.camera.set_world_pose.call_args.kwargs['orientation']
        env._pitch = .4
        env._author_camera_pose()
        self.assertNotEqual(first_orientation, env.camera.set_world_pose.call_args.kwargs['orientation'])
        for call in env._robot_body.sync.call_args_list:
            np.testing.assert_array_equal(call.args[0], [2, 3, 2.5])
            self.assertEqual(call.args[1], .7)

    def test_larger_body_blocks_a_narrow_passage_and_scoring_uses_same_radius(self):
        from roboodyssey.evaluation.navigation import GridDistance
        env = IsaacVLNEnv.__new__(IsaacVLNEnv)
        env._np = np
        env.config = EnvConfig(robot_body=None, robot_radius=.3)
        env._origin, env._resolution = np.array([0., 0.]), .1
        env._static_base = np.ones((20, 20), dtype=np.uint8)
        env._static_base[:, 8:12] = 0
        env._static = env._static_base
        env._dynamic = np.zeros_like(env._static)
        env._door_open, env._door_cells = set(), {}
        self.assertFalse(env._walkable([1., 1.]))
        grid = GridDistance(env._static_base, env._origin, env._resolution, radius=env.config.robot_radius)
        self.assertFalse(grid.walkable([1., 1.]))
        env.config = EnvConfig(robot_body=None, robot_radius=.1)
        self.assertTrue(env._walkable([1., 1.]))
        grid = GridDistance(env._static_base, env._origin, env._resolution, radius=env.config.robot_radius)
        self.assertTrue(grid.walkable([1., 1.]))

    def test_custom_body_cannot_be_mislabeled_as_original_official_profile(self):
        from roboodyssey.run import main
        with redirect_stderr(StringIO()), patch('roboodyssey.run.BenchmarkDataset') as dataset:
            with self.assertRaises(SystemExit) as caught:
                main(['--dataset', 'unused', '--robot-profile', str(self.path), '--official', '--dry-run'])
            self.assertEqual(caught.exception.code, 2)
            dataset.assert_not_called()

    def test_custom_profile_is_checked_in_cpu_dry_run(self):
        from roboodyssey.run import main
        dataset = SimpleNamespace(manifest={}, scene_ids=lambda: [])
        with redirect_stdout(StringIO()) as output, patch('roboodyssey.run.BenchmarkDataset', return_value=dataset):
            self.assertEqual(main(['--dataset', 'unused', '--robot-profile', str(self.path), '--dry-run']), 0)
        identity = json.loads(output.getvalue().removeprefix('Robot: '))
        self.assertEqual(identity['model_id'], 'example')
        self.assertEqual(identity['robot_radius_m'], .3)

    def test_named_launch_selection_survives_sequence_child_arguments(self):
        from roboodyssey.launch.sequences import child_arguments
        command = child_arguments(['--dataset', 'original', '--robot', 'stretch3',
                                   '--robot-assets', str(self.root/'assets'),
                                   '--agent-cmd', 'python -m external_agent',
                                   '--sensor-plugin', 'external_sensors'],
                                  'next_dataset', 'next_output')
        self.assertEqual(command[command.index('--robot')+1], 'stretch3')
        self.assertEqual(command[command.index('--robot-assets')+1], str(self.root/'assets'))
        self.assertEqual(command[command.index('--agent-cmd')+1], 'python -m external_agent')
        self.assertEqual(command[command.index('--sensor-plugin')+1], 'external_sensors')

    def test_official_rejects_cylinder_before_loading_data(self):
        from roboodyssey.run import main
        with redirect_stderr(StringIO()), patch('roboodyssey.run.BenchmarkDataset') as dataset:
            with self.assertRaises(SystemExit):
                main(['--dataset', str(self.root), '--robot', 'cylinder', '--official', '--dry-run'])
            dataset.assert_not_called()

    def test_scene_declaration_and_navigation_radius_are_checked(self):
        from roboodyssey.scenes.robot_body import default_robot_identity, validate_scene_robot
        identity = default_robot_identity()
        nav = self.root/'navigation.npz'
        np.savez(nav, robot_radius=identity['robot_radius_m'])
        scene = SimpleNamespace(manifest={'robot_profile': identity}, path=lambda key: nav)
        validate_scene_robot(scene, identity, require_declared=True)
        changed = dict(identity, robot_radius_m=.1)
        with self.assertRaisesRegex(ValueError, 'does not match'):
            validate_scene_robot(scene, changed)
        np.savez(nav, robot_radius=.1)
        with self.assertRaisesRegex(ValueError, 'navigation robot_radius'):
            validate_scene_robot(scene, identity)
        validate_scene_robot({}, identity)
        with self.assertRaisesRegex(ValueError, 'requires a declared'):
            validate_scene_robot({}, identity, require_declared=True)

    def test_public_raster_default_and_npz_preserve_stretch_identity(self):
        from roboodyssey.contracts.raster import derive_navigation
        from roboodyssey.contracts.navigation_io import read_navigation_npz, write_navigation_npz
        from roboodyssey.scenes.robot_body import default_robot_identity, validate_scene_robot
        rooms = [{'room_id': 'room', 'boundary_polygon': [[0,0], [3,0], [3,3], [0,3]]}]
        grid = derive_navigation(rooms, [], resolution=.5)
        identity = default_robot_identity()
        self.assertEqual(grid['robot_radius'], .29)
        self.assertEqual(grid['robot_profile'], identity)
        grid['scene_id'] = 'test'
        path = self.root/'nav.npz'
        write_navigation_npz(path, grid)
        self.assertEqual(read_navigation_npz(path)['robot_profile'], identity)
        scene = SimpleNamespace(manifest={'robot_profile': identity}, path=lambda key: path)
        validate_scene_robot(scene, identity)
        grid['robot_profile'] = dict(identity, profile_version='modified')
        write_navigation_npz(path, grid)
        with self.assertRaisesRegex(ValueError, 'differs from scene manifest'):
            validate_scene_robot(scene, identity)
        # Choosing a scalar radius does not imply a body, even when equal to the default.
        self.assertNotIn('robot_profile', derive_navigation(rooms, [], resolution=.5, robot_radius=.29))

    def test_stretch_circle_uses_same_runtime_and_reference_clearance(self):
        from roboodyssey.evaluation.navigation import GridDistance
        env = IsaacVLNEnv.__new__(IsaacVLNEnv)
        env._np = np
        env._origin, env._resolution = np.array([0., 0.]), .1
        env._static_base = np.ones((20, 20), dtype=np.uint8)
        env._static_base[:, 5:12] = 0  # 0.7 m corridor; circle is centered in it.
        env._static = env._static_base
        env._dynamic = np.zeros_like(env._static)
        env._door_open, env._door_cells, env._held_object = set(), {}, None
        for config, expected in ((EnvConfig(), True), (EnvConfig(robot_body=None, robot_radius=.36), False)):
            env.config = config
            self.assertEqual(env._walkable([.85, 1.05]), expected)
            snapshot = env.evaluation_navigation()
            grid = GridDistance(snapshot['blocked'], snapshot['origin'], snapshot['resolution'], radius=snapshot['radius'])
            self.assertEqual(grid.walkable([.85, 1.05]), expected)
            self.assertEqual(snapshot['radius'], config.robot_radius)

    def test_bundled_profile_is_pinned_independently_of_mutable_json(self):
        from dataclasses import replace
        from roboodyssey.scenes.robot_body import default_robot_body
        canonical = default_robot_body()
        changed = replace(canonical, profile_sha256='0'*64)
        with patch('roboodyssey.scenes.robot_body.load_robot_profile', return_value=changed):
            with self.assertRaisesRegex(ValueError, 'pinned'):
                default_robot_body()

    def test_official_accepts_only_canonical_named_stretch(self):
        from roboodyssey.scenes.robot_body import validate_official_robot
        validate_official_robot(resolve_robot())
        with self.assertRaisesRegex(ValueError, 'bundled'):
            validate_official_robot(resolve_robot('cylinder'))
        self.named_profile('stretch3')
        with self.assertRaisesRegex(ValueError, 'bundled'):
            validate_official_robot(resolve_robot('stretch3', assets=self.root))



if __name__ == '__main__':
    unittest.main()
