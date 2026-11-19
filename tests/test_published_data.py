"""Standalone release-data ingestion, integrity and boundary mutation checks."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from benchmark.artifacts.layout import sha256_file, write_json
from benchmark.data.archive import dataset_root
from benchmark.data.frozen import select_sequences
from benchmark.scenes.dataset_loader import BenchmarkDataset
from benchmark.validation.dataset import validate
from tests.frozen_data import fixture_v3, select_scenario, write_dataset


class PublishedDataTests(unittest.TestCase):
    def archive(self, folder, output, prefix='session/preparation/collection'):
        with tarfile.open(output, 'w') as archive:
            archive.add(folder, arcname=prefix)

    def test_published_wrapper_selects_sequences_and_reuses_separate_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from tests.test_scene_sequences import SequenceIntegrationTests
            collection = SequenceIntegrationTests().collection(root/'collection')
            archive = root/'session.tar'
            self.archive(collection, archive)
            original = sha256_file(archive)
            with patch.dict('os.environ', {'BENCHMARK_DATA_CACHE': str(root/'cache')}):
                self.assertEqual(validate(archive), [])
                with self.assertRaisesRegex(ValueError, 'select'):
                    BenchmarkDataset(archive)
                data = BenchmarkDataset(archive, sequence_id='sequence_A')
                self.assertTrue(data.root.is_relative_to(root/'cache'))
                self.assertEqual(len(data.sessions('synthetic_demo')), 2)
                self.assertEqual(len(select_sequences(archive)), 2)
                resolved = dataset_root(archive)
                self.assertEqual(dataset_root(archive), resolved)
                self.assertEqual(sha256_file(archive), original)
                source = collection/'scene_versions/r_fixture/scene_metainfo.json'
                self.assertEqual(source.read_bytes(), data._requests.path('scene_metainfo').read_bytes())
                data._requests.path('scene_metainfo').write_text('{}')
                self.assertTrue(any('checksum' in error for error in validate(archive)))
                self.assertEqual(sha256_file(archive), original)

    def test_archive_path_links_and_duplicate_members_are_rejected(self):
        for mode in ('parent', 'absolute', 'symlink', 'duplicate'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); archive = root/'session.tar'
                with tarfile.open(archive, 'w') as output:
                    name = '../escape' if mode == 'parent' else '/escape' if mode == 'absolute' else 'entry'
                    member = tarfile.TarInfo(name)
                    if mode == 'symlink':
                        member.type, member.linkname = tarfile.SYMTYPE, '../escape'
                        output.addfile(member)
                    else:
                        member.size = 1
                        output.addfile(member, io.BytesIO(b'x'))
                        if mode == 'duplicate':
                            output.addfile(member, io.BytesIO(b'x'))
                with patch.dict('os.environ', {'BENCHMARK_DATA_CACHE': str(root/'cache')}):
                    with self.assertRaises(ValueError):
                        dataset_root(archive)
                self.assertFalse((root/'escape').exists())
                self.assertEqual(list((root/'cache').iterdir()), [])

    def test_manifest_order_is_checked_after_valid_file_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'data'
            c, inputs, plan = fixture_v3(second=[{'task_id': 'ask', 'objectives': [
                {'goal_id': 'answer_book_red', 'kind': 'answer', 'object_id': 'book_red',
                 'query': 'room_after', 'history': {'session_id': 'session_000', 'task_id': 'move', 'goal_id': 'transfer_book_red'}}]}])
            manifest = write_dataset(root, select_scenario(c, plan), c, inputs)
            manifest['sessions'].reverse()
            write_json(root/'sequences/seq_fixture/dataset_manifest.json', manifest)
            index = json.loads((root/'scene_manifest.json').read_text())
            index['sequences']['seq_fixture']['sha256'] = sha256_file(root/'sequences/seq_fixture/dataset_manifest.json')
            write_json(root/'scene_manifest.json', index)
            with self.assertRaisesRegex(ValueError, 'session order'):
                BenchmarkDataset(root/'sequences/seq_fixture')

    def test_download_accepts_published_wrapper_and_verifies_before_copy(self):
        from benchmark.cli import _download
        from tests.test_scene_sequences import SequenceIntegrationTests
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = SequenceIntegrationTests().collection(root/'source')
            archive = root/'session.tar'
            self.archive(source, archive)
            with patch.dict('os.environ', {'BENCHMARK_DATA_CACHE': str(root/'cache')}):
                with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                    _download(archive.as_uri(), root/'bad', '0'*64)
                self.assertFalse((root/'bad').exists())
                self.assertEqual(_download(archive.as_uri(), root/'downloaded', sha256_file(archive)), 0)
                self.assertEqual(len(select_sequences(root/'downloaded')), 2)
                with self.assertRaises(FileExistsError):
                    _download(archive.as_uri(), root/'downloaded', sha256_file(archive))

    def test_runtime_mutation_uses_published_geometry_and_actual_state(self):
        from benchmark.contracts.requests import prepare_mutation
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'data'
            c, inputs, plan = fixture_v3()
            write_dataset(root, select_scenario(c, plan), c, inputs)
            context = BenchmarkDataset(root)._requests.mutation_context()
            state = {'objects': {o['object_id']: deepcopy(o) for o in inputs['metadata']['objects']},
                     'held_object': None, 'robot_pose': {'position': [.4, .4, 0]}}
            mutation = {'changes': [{'object_id': 'book_red', 'relation_after': {'room_id': 'kitchen'}}]}
            history = [{'object_id': 'book_red', 'dispatched': True, 'after': deepcopy(state)}]
            before = deepcopy(state)
            result = prepare_mutation(context, mutation, state, history)
            self.assertEqual(result['status'], 'ready', result)
            self.assertEqual(result['destination_room_id'], 'kitchen')
            self.assertEqual(state, before)
            state['objects']['book_red']['held'] = True
            state['held_object'] = 'book_red'
            skipped = prepare_mutation(context, mutation, state, history)
            self.assertEqual(skipped['status'], 'skipped')

    def test_removed_input_versions_are_rejected_before_environment_start(self):
        from benchmark.data.frozen import read
        for field, value in (('schema_version', 1), ('schema_version', 4), ('layout_version', 1),
                             ('runtime_contract', {'name': 'contextual_requests', 'version': 1}),
                             ('runtime_contract', {'name': 'contextual_requests', 'version': 2})):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)/'data'
                c, inputs, plan = fixture_v3()
                write_dataset(root, select_scenario(c, plan), c, inputs)
                sequence = root/'sequences/seq_fixture'
                manifest = read(sequence/'dataset_manifest.json')
                manifest[field] = value
                write_json(sequence/'dataset_manifest.json', manifest)
                with self.assertRaisesRegex(ValueError, 'unsupported'):
                    BenchmarkDataset(sequence)
                self.assertTrue(validate(sequence))

    def test_removed_import_paths_are_unavailable(self):
        import importlib
        for name in ('benchmark.legacy.task_repair', 'benchmark.runtime.task_repair',
                     'benchmark.runtime.sequence_runs', 'benchmark.contracts.calendar',
                     'benchmark.evaluation.accounting', 'benchmark.reporting.legacy'):
            with self.subTest(module=name), self.assertRaises(ModuleNotFoundError):
                importlib.import_module(name)

    def test_no_source_module_imports_generation_packages(self):
        import ast
        root = Path(__file__).resolve().parents[1]
        for folder in ('benchmark', 'examples', 'scripts', 'tests'):
            for path in (root/folder).rglob('*.py'):
                tree = ast.parse(path.read_text(encoding='utf-8'))
                for node in ast.walk(tree):
                    imports = ([node.module or ''] if isinstance(node, ast.ImportFrom)
                               else [a.name for a in node.names] if isinstance(node, ast.Import) else [])
                    self.assertFalse(any(name.split('.')[0] in {'CL_task_builder', 'CL_session_builder'}
                                         for name in imports), str(path))


if __name__ == '__main__':
    unittest.main()
