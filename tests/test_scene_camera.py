import math
import json
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
import numpy as np

from roboodyssey.scenes.camera import SceneCamera


class SceneCameraTests(unittest.TestCase):
    def test_rectangular_resolution_preserves_horizontal_fov_and_closes_once(self):
        camera = Mock()
        with patch('roboodyssey.scenes.camera.sha256_file', return_value='asset_hash'), \
             patch('roboodyssey.scenes.camera.open_scene_stage', return_value=(object(), Mock())), \
             patch('roboodyssey.scenes.camera.create_camera', return_value=camera), \
             patch('roboodyssey.scenes.camera.apply_scene_lighting', return_value={'mode': 'benchmark_fill'}) as lighting, \
             patch('roboodyssey.contracts.doors.apply_open_doors', return_value={}):
            sensor = SceneCamera('/unused.usdz', resolution=(640, 480), horizontal_fov_degrees=100)
        horizontal = camera.set_horizontal_aperture.call_args.args[0]
        vertical = camera.set_vertical_aperture.call_args.args[0]
        self.assertAlmostEqual(horizontal, 48*math.tan(math.radians(50)))
        self.assertAlmostEqual(vertical/horizontal, 480/640)
        lighting.assert_called_once()
        camera.get_rgb.return_value = np.ones((480, 640, 3), np.uint8)
        sensor.capture([1, 2, 3], [0, 0, 0, 1])
        first = sensor.last_capture_metadata
        self.assertEqual(first['state_source'], 'preparation')
        self.assertEqual(first['state_identity']['scene_asset_sha256'], 'asset_hash')
        self.assertIn('lighting_sha256', first['state_identity'])
        self.assertEqual(first['lighting']['mode'], 'benchmark_fill')
        sensor.capture([2, 2, 3], [0, 0, 0, 1])
        second = sensor.last_capture_metadata
        self.assertEqual(second['frame_id'], first['frame_id']+1)
        self.assertEqual(second['state_version'], first['state_version'])
        self.assertNotEqual(second['camera_pose'], first['camera_pose'])
        self.assertEqual(second['image_sha256'], first['image_sha256'])
        camera.get_rgb.return_value = np.zeros((480, 640, 3), np.uint8)
        with self.assertRaisesRegex(RuntimeError, 'all-black'):
            sensor.capture([1, 2, 3], [0, 0, 0, 1])
        self.assertEqual(sensor.last_capture_metadata['frame_id'], 2)
        sensor.close()
        sensor.close()
        camera.destroy.assert_called_once()

    def test_shared_default_lighting_matches_original_runtime_and_authored_is_explicit(self):
        from roboodyssey.scenes.lighting import LightingConfig, apply_scene_lighting
        from roboodyssey.scenes.isaac_env import EnvConfig
        self.assertEqual(EnvConfig().lighting, LightingConfig())
        usd_lux = SimpleNamespace(DomeLight=Mock(), DistantLight=Mock())
        usd_geom = SimpleNamespace(XformCommonAPI=Mock())
        gf = SimpleNamespace(Vec3f=lambda *xyz: xyz)
        with patch.dict('sys.modules', {'pxr': SimpleNamespace(Gf=gf, UsdGeom=usd_geom, UsdLux=usd_lux)}):
            value = apply_scene_lighting('stage')
        self.assertEqual(value['dome_intensity'], 3000.)
        self.assertEqual(value['fill_intensity'], 800.)
        self.assertEqual(value, json.loads(json.dumps(value)))
        usd_lux.DomeLight.Define.assert_called_once_with('stage', '/World/BenchmarkRuntime/DomeLight')
        usd_lux.DomeLight.Define.return_value.CreateIntensityAttr.assert_called_once_with(3000.)
        usd_geom.XformCommonAPI.return_value.SetRotate.assert_called_once_with((25., -20., -35.))
        self.assertEqual(apply_scene_lighting(None, LightingConfig(mode='authored'))['mode'], 'authored')
        with self.assertRaises(ValueError):
            LightingConfig(dome_intensity=-1)


if __name__ == '__main__':
    unittest.main()
