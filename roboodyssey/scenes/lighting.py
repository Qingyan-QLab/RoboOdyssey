"""Explicit baseline illumination shared by scene preparation and execution."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class LightingConfig:
    mode: str = 'benchmark_fill'
    dome_intensity: float = 3000.0
    fill_intensity: float = 800.0
    fill_rotation_xyz_degrees: tuple = (25.0, -20.0, -35.0)

    def __post_init__(self):
        if self.mode not in {'benchmark_fill', 'authored'}:
            raise ValueError('lighting mode must be benchmark_fill or authored')
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
               for value in (self.dome_intensity, self.fill_intensity)):
            raise ValueError('light intensities must be finite and nonnegative')
        if len(self.fill_rotation_xyz_degrees) != 3 or any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in self.fill_rotation_xyz_degrees):
            raise ValueError('fill rotation requires three finite XYZ degrees')

    def as_dict(self):
        return {'version': 'benchmark_fill_lighting_v1', **asdict(self),
                'fill_rotation_xyz_degrees': list(self.fill_rotation_xyz_degrees)}


def apply_scene_lighting(stage, config=None):
    """Apply the declared policy; failures are explicit, never silent fallback."""
    config = config or LightingConfig()
    if not isinstance(config, LightingConfig):
        raise TypeError('lighting must be a LightingConfig')
    if config.mode == 'authored':
        return config.as_dict()
    from pxr import Gf, UsdGeom, UsdLux
    dome = UsdLux.DomeLight.Define(stage, '/World/BenchmarkRuntime/DomeLight')
    dome.CreateIntensityAttr(config.dome_intensity)
    fill = UsdLux.DistantLight.Define(stage, '/World/BenchmarkRuntime/FillLight')
    fill.CreateIntensityAttr(config.fill_intensity)
    UsdGeom.XformCommonAPI(fill).SetRotate(Gf.Vec3f(*config.fill_rotation_xyz_degrees))
    return config.as_dict()
