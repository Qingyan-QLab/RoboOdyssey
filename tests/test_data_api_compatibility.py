"""Both public scene contracts retain their identities and validation boundaries."""
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import roboodyssey
from roboodyssey.artifacts.layout import sha256_file
from roboodyssey.data import (
    SceneSpec, SceneVersion, load_scene, load_scene_spec, load_scene_version,
    load_sequence, validate_binding,
)
from roboodyssey.scenes.dataset_loader import BenchmarkDataset
from tests.frozen_data import fixture_v3, goal, select_scenario, task, write_dataset


class SceneApiCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.collection = self.root / 'collection'
        context, inputs, plan = fixture_v3([task('visit', goal('cabinet_dark'))])
        write_dataset(self.collection, select_scenario(context, plan), context, inputs)
        published = BenchmarkDataset(self.collection)
        self.bundle = self.root / 'resource'
        self.bundle.mkdir()
        (self.bundle / 'scene.usda').write_text('#usda 1.0\n')
        resources = {}
        for name in ('scene_metainfo', 'navigation', 'manipulation', 'environment_setup',
                     'initial_state', 'runtime_scene_cache'):
            source = (published._requests.version.path(name) if name == 'initial_state'
                      else published._requests.path(name))
            target = self.bundle / (name + source.suffix)
            shutil.copy2(source, target)
            resources[name] = target.name
        self.manifest = {
            'schema_version': 1, 'kind': 'roboodyssey_scene',
            'scene_id': published.scene_ids()[0], 'asset': 'scene.usda',
            'asset_sha256': sha256_file(self.bundle / 'scene.usda'),
            'resources': resources,
            'sha256': {name: sha256_file(self.bundle / path) for name, path in resources.items()},
        }
        (self.bundle / 'scene_resources.json').write_text(json.dumps(self.manifest))
        cache = patch.dict(os.environ, {'BENCHMARK_DATA_CACHE': str(self.root / 'cache')})
        cache.start()
        self.addCleanup(cache.stop)

    def test_public_entries_preserve_both_contracts(self):
        for entry in (load_scene, roboodyssey.load_scene):
            resource = entry(self.bundle)
            self.assertIsInstance(resource, SceneSpec)
            self.assertEqual(resource.manifest_path, self.bundle / 'scene_resources.json')
            frozen = entry(self.collection, revision='r_fixture')
            self.assertIsInstance(frozen, SceneVersion)
            sequence = load_sequence(self.collection, scene=frozen)
            self.assertEqual(validate_binding(frozen, sequence)['scene_reference'], frozen.reference())
            self.assertIs(entry(resource), resource)
            self.assertIs(entry(frozen, revision='r_fixture'), frozen)
        self.assertIsInstance(load_scene_spec(self.bundle), SceneSpec)
        self.assertIsInstance(load_scene_version(self.collection), SceneVersion)

    def test_both_archive_formats_use_their_existing_loaders(self):
        for name, directory, arcname, expected in (
            ('resource.tar', self.bundle, 'scene_bundle', SceneSpec),
            ('published.tar', self.collection, 'session/preparation/collection', SceneVersion),
        ):
            target = self.root / name
            with tarfile.open(target, 'w') as archive:
                archive.add(directory, arcname=arcname)
            self.assertIsInstance(load_scene(target), expected)
            self.assertIsInstance(load_scene(target), expected)  # Cached reads keep the contract.

    def test_revision_does_not_silently_change_the_resource_contract(self):
        for source in (self.bundle, load_scene_spec(self.bundle)):
            with self.assertRaisesRegex(ValueError, 'revision.*SceneVersion'):
                load_scene(source, revision='r_fixture')
        with self.assertRaisesRegex(ValueError, 'unknown scene revision'):
            load_scene(self.collection, revision='missing')
        with self.assertRaisesRegex(ValueError, 'differs from input'):
            load_scene(load_scene_version(self.collection), revision='missing')

    def test_mixed_contracts_require_an_explicit_loader(self):
        for source in self.bundle.iterdir():
            shutil.copy2(source, self.collection / source.name)
        with self.assertRaisesRegex(ValueError, 'ambiguous scene source'):
            load_scene(self.collection)
        self.assertIsInstance(load_scene_spec(self.collection), SceneSpec)
        self.assertIsInstance(load_scene_version(self.collection), SceneVersion)
        target = self.root / 'mixed.tar'
        with tarfile.open(target, 'w') as archive:
            archive.add(self.collection, arcname='.')
        with self.assertRaisesRegex(ValueError, 'ambiguous scene source'):
            load_scene(target)

    def test_resource_bundle_cannot_bypass_frozen_binding(self):
        resource = load_scene(self.bundle)
        sequence = load_sequence(self.collection)
        with self.assertRaisesRegex(ValueError, 'SceneVersion.*BenchmarkSuite'):
            load_sequence(self.collection, scene=resource)
        with self.assertRaisesRegex(ValueError, 'SceneVersion.*BenchmarkSuite'):
            validate_binding(resource, sequence)

    def test_validation_errors_do_not_fall_back_to_another_contract(self):
        (self.bundle / 'scene_metainfo.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            load_scene(self.bundle)
        version = load_scene_version(self.collection)
        version.path('scene_metainfo').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            load_scene(self.collection)


if __name__ == '__main__':
    unittest.main()
