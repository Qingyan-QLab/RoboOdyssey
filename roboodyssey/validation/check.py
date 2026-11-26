"""Source regression suite or installed-wheel smoke checks, always CPU-only."""
import argparse
import importlib.abc
import os
from pathlib import Path
import sys
import unittest


class NoAcceleratorImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
                'torch', 'isaacsim', 'omni', 'carb', 'cupy', 'tensorflow', 'jax',
                'CL_task_builder', 'CL_session_builder', 'roboodyssey_datasynth',
                'odyssey_task_builder', 'synthesis', 'oracle_policies'}:
            raise RuntimeError('CPU validation forbids importing '+fullname)
        return None


def installed_smoke():
    """Check public CPU modules, protocol resources and a collision query."""
    import importlib
    import importlib.resources
    import json
    import numpy as np
    from ..agents.protocol import validate_response
    from ..evaluation.navigation import GridDistance
    from ..data import CURRENT_FORMAT
    for module in ('contracts.planning', 'contracts.snapshots', 'runtime.policy',
                   'runtime.capture', 'scenes.camera', 'scenes.isaac_env',
                   'agents.simplenav_cli', 'agents.openai_api_cli', 'run'):
        importlib.import_module('roboodyssey.'+module)
    schema = importlib.resources.files('roboodyssey.agents').joinpath('protocol_schema.json')
    json.loads(schema.read_text())
    if CURRENT_FORMAT['layout_version'] != 2:
        raise AssertionError('unexpected current layout')
    validate_response('step', {'action': 'no_op', 'done': True})
    blocked = np.zeros((4, 4), bool)
    blocked[1, 1] = True
    grid = GridDistance(blocked, [0, 0], 1, 0)
    if grid.walkable([1.5, 1.5]) or not grid.walkable([2.5, 2.5]):
        raise AssertionError('collision query failed')
    print('Installed package CPU smoke checks passed. Full regression tests require the source distribution.')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', action='store_true', help='source tree: select the fast regression subset')
    args = parser.parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    guard = NoAcceleratorImports()
    sys.meta_path.insert(0, guard)
    try:
        root = Path(__file__).resolve().parents[2]
        if not (root/'pyproject.toml').is_file() or not (root/'tests/test_benchmark.py').is_file():
            return installed_smoke()
        sys.path.insert(0, str(root))
        if args.core:
            patterns = ('test_benchmark.py', 'test_task_diagnostics.py', 'test_observations.py',
                        'test_evaluation_metrics.py', 'test_transfer_metrics.py', 'test_openai_api_agent.py')
            suite = unittest.TestSuite(unittest.defaultTestLoader.discover(str(root/'tests'), pattern=p)
                                       for p in patterns)
            print('Core suite only; omit --core for complete source regression tests.', flush=True)
        else:
            suite = unittest.defaultTestLoader.discover(str(root/'tests'))
        return 0 if unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful() else 1
    finally:
        sys.meta_path.remove(guard)


if __name__ == '__main__':
    raise SystemExit(main())
