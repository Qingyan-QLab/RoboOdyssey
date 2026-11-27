"""Current artifacts are readable independently, without a generating repository."""
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import unittest

from roboodyssey.data import load_scene, load_sequence, validate_binding
from roboodyssey.scenes.dataset_loader import BenchmarkDataset
from tests.frozen_data import fixture_v3, select_scenario, write_dataset, prepare_scene


class PublicDataTests(unittest.TestCase):
    def test_scene_before_sequences_and_detached_binding(self):
        context, inputs, plan = fixture_v3()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = prepare_scene(root/'prepared', context, inputs, plan['names'])
            scene = load_scene(root/'prepared')
            self.assertEqual(scene.reference(), prepared.reference())
            write_dataset(root/'collection', select_scenario(context, plan), context, inputs)
            sequence = load_sequence(root/'collection')
            shutil.copytree(sequence.root, root/'detached_sequence')
            shutil.copytree(sequence.version.directory, root/'detached_scene')
            detached = load_sequence(root/'detached_sequence', scene=root/'detached_scene')
            self.assertEqual(validate_binding(scene, detached)['sequence_id'], sequence.manifest['sequence_id'])
            dataset = BenchmarkDataset(detached)
            self.assertEqual(dataset.sessions('synthetic_demo')[0].tasks, BenchmarkDataset(sequence).sessions('synthetic_demo')[0].tasks)

    def test_explicit_scene_cannot_bypass_hash_binding(self):
        context, inputs, plan = fixture_v3()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root/'collection', select_scenario(context, plan), context, inputs)
            sequence = load_sequence(root/'collection')
            wrong = deepcopy(sequence.version)
            wrong.manifest_hash = '0'*64
            with self.assertRaisesRegex(ValueError, 'revision/hash'):
                load_sequence(sequence.root, scene=wrong)
            with self.assertRaisesRegex(ValueError, 'sequence_id'):
                load_sequence(sequence.root, sequence_id='wrong')


if __name__ == '__main__':
    unittest.main()
