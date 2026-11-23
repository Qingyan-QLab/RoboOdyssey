"""Standalone Scene/Session bundle loading and core runner integration."""
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from roboodyssey import BenchmarkSuite, load_scene, load_session
from roboodyssey.artifacts.layout import sha256_file
from roboodyssey.scenes.dataset_loader import BenchmarkDataset
from tests.frozen_data import fixture_v3, goal, select_scenario, task, write_dataset
from tests.test_contextual_requests import Agent, Env


class ExternalSourceTests(unittest.TestCase):
    def test_directory_and_tar_sources_run_through_the_core(self):
        compiler, inputs, plan = fixture_v3([task("visit", goal("cabinet_dark"))])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_dataset(root / "published", select_scenario(compiler, plan), compiler, inputs)
            published = BenchmarkDataset(root / "published")
            BenchmarkSuite.from_published(root / "published").validate("interaction")
            scene_id = published.scene_ids()[0]
            old_session = published.sessions(scene_id)[0]
            asset = root / "scene.usd"
            asset.write_text("#usda 1.0\n", encoding="utf-8")
            names = ("scene_metainfo", "navigation", "manipulation", "environment_setup",
                     "initial_state", "runtime_scene_cache")
            paths = {name: published._requests.version.path(name) if name == "initial_state"
                     else published._requests.path(name) for name in names}
            scene_dir = root / "scene_bundle"
            scene_dir.mkdir()
            shutil.copy2(asset, scene_dir / asset.name)
            scene_files = {}
            for name, source in paths.items():
                target = scene_dir / f"{name}{Path(source).suffix}"
                shutil.copy2(source, target)
                scene_files[name] = target.name
            (scene_dir / "scene_resources.json").write_text(json.dumps({
                "schema_version": 1, "kind": "roboodyssey_scene", "scene_id": scene_id,
                "asset": asset.name, "asset_sha256": sha256_file(scene_dir / asset.name),
                "resources": scene_files,
                "sha256": {name: sha256_file(scene_dir / path) for name, path in scene_files.items()},
            }))
            scene = load_scene(scene_dir)
            session_dir = root / "session_bundle"
            session_dir.mkdir()
            shutil.copy2(old_session.path / "session.json", session_dir / "session.json")
            session_files = {}
            for name in ("ground_truth", "templates", "mutations"):
                target = session_dir / f"{name}.json"
                shutil.copy2(old_session.private_paths[name], target)
                session_files[name] = target.name
            (session_dir / "session_resources.json").write_text(json.dumps({
                "schema_version": 1, "kind": "roboodyssey_session", "scene_id": scene_id,
                "sequence_id": old_session.metadata["sequence_id"],
                "session_id": old_session.session_id, "session": "session.json",
                "session_sha256": sha256_file(session_dir / "session.json"),
                "runtime_contract": {"name": "contextual_requests", "version": 3},
                "resources": session_files,
                "sha256": {name: sha256_file(session_dir / path) for name, path in session_files.items()},
            }))
            session = load_session(session_dir)
            with tarfile.open(root / "scene.tar", "w") as archive:
                archive.add(root / "scene_bundle", arcname="scene_bundle")
            with tarfile.open(root / "session.tar", "w") as archive:
                archive.add(root / "session_bundle", arcname="session_bundle")
            self.assertEqual(load_scene(root / "scene_bundle").scene_id, scene_id)
            with patch.dict(os.environ, {"BENCHMARK_DATA_CACHE": str(root / "cache")}):
                self.assertEqual(load_scene(root / "scene.tar").scene_id, scene_id)
                self.assertEqual(load_session(root / "session.tar").session_id, session.session_id)
            suite = BenchmarkSuite(scene, [session])
            from roboodyssey.run import main
            self.assertEqual(main(["--scene-source", str(root / "scene_bundle"),
                                   "--session-source", str(root / "session_bundle"),
                                   "--dry-run"]), 0)
            state = suite.dataset.canonical_state(scene_id)
            output = suite.interact(Agent(), root / "run", lambda *_: Env(state), max_steps=1)
            summary = suite.evaluate(output)
            self.assertEqual(summary["sessions_total"], 1)
            self.assertTrue((output / "manifest.json").is_file())
            session.path.write_text(session.path.read_text() + " ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_session(root / "session_bundle")

    def test_rejects_archive_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "scene.tar"
            with tarfile.open(archive_path, "w") as archive:
                info = tarfile.TarInfo("../escape")
                info.size = 0
                archive.addfile(info)
            with patch.dict(os.environ, {"BENCHMARK_DATA_CACHE": str(Path(temporary) / "cache")}):
                with self.assertRaisesRegex(ValueError, "escapes"):
                    load_scene(archive_path)
