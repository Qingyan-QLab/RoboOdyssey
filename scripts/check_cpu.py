#!/usr/bin/env python3
"""Run the CPU regression suite; fail if any model/GPU/simulator SDK is imported."""
import argparse
import importlib.abc
import os
from pathlib import Path
import sys
import unittest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--core', action='store_true', help='fast subset; the default also runs standalone frozen-data integration tests')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=''
class NoAcceleratorImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','isaacsim','omni','carb','cupy','tensorflow','jax','CL_task_builder','CL_session_builder'}:
            raise RuntimeError('CPU validation forbids importing '+fullname)
        return None
sys.meta_path.insert(0,NoAcceleratorImports())
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
if args.core:
    patterns = ('test_benchmark.py', 'test_task_diagnostics.py', 'test_observations.py', 'test_evaluation_metrics.py', 'test_transfer_metrics.py', 'test_contextual_requests.py')
    suite = unittest.TestSuite(unittest.defaultTestLoader.discover(str(root/'tests'), pattern=p) for p in patterns)
    print('Core suite only: use the default check for frozen-data and launch integration tests.', flush=True)
else:
    suite = unittest.defaultTestLoader.discover(str(root/'tests'))
result=unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
