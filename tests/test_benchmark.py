import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from roboodyssey.interaction.protocol import validate_response
from roboodyssey.scenes.dataset_loader import BenchmarkDataset



class MetricTests(unittest.TestCase):
    def test_protocol(self):
        validate_response("step", {"action": {"action": "pick", "target": "cup"}})
        validate_response("step", {"action": "no_op", "answer": "kitchen"})
        with self.assertRaises(ValueError):
            validate_response("step", {"action": {"action": "teleport"}})











class DatasetTests(unittest.TestCase):
    def setUp(self):
        from tests.frozen_data import (
            fixture_v3,
            task,
            goal,
            select_scenario,
            write_dataset,
        )

        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        c, inputs, plan = fixture_v3(
            [task("ask", goal("book_red", "answer", query="current_room"))]
        )
        self.root = Path(temp.name) / "data"
        write_dataset(self.root, select_scenario(c, plan), c, inputs)
        self.dataset = BenchmarkDataset(self.root)

    def test_dataset_shape_and_canonical_state(self):
        self.assertEqual(self.dataset.scene_ids(), ["synthetic_demo"])
        sessions = self.dataset.sessions("synthetic_demo")
        self.assertEqual(len(sessions), 1)
        self.assertEqual(
            [t["task_id"] for t in sessions[0].tasks],
            [t["task_id"] for t in sessions[0].ground_truth()],
        )
        state = self.dataset.canonical_state("synthetic_demo")
        self.assertEqual(state["scene_id"], "synthetic_demo")
        self.assertIn("book_red", state["objects"])
        self.assertIn("robot_base_pose", state)

    def test_cli_dry_run_uses_explicit_published_dataset(self):
        from roboodyssey.cli import main
        from contextlib import redirect_stdout
        from io import StringIO

        with redirect_stdout(StringIO()) as output:
            self.assertEqual(main(["run", "--dataset", str(self.root), "--dry-run"]), 0)
        self.assertIn("synthetic_demo: 1 sessions", output.getvalue())

    def test_cli_executes_current_sequence_and_closes_simulator(self):
        from importlib import import_module
        import json
        from contextlib import redirect_stdout
        from io import StringIO
        from types import ModuleType
        from unittest.mock import patch
        from roboodyssey.run import main
        from tests.test_contextual_requests import Agent, Env

        # Load before patch.dict snapshots sys.modules; otherwise its rollback
        # leaves a stale parent-package attribute that Python 3.10 mocks follow.
        backend_module = import_module("roboodyssey.scenes.isaac_backend")

        class App:
            def __init__(self, options):
                self.closed = False

            def update(self):
                pass

            def close(self, exit_code=0):
                self.closed = True
                self.exit_code = exit_code

        app = App({})
        isaac = ModuleType("isaacsim")
        isaac.SimulationApp = lambda options: app
        omni = ModuleType("omni")
        omni.kit = ModuleType("omni.kit")
        omni.kit.app = ModuleType("omni.kit.app")
        omni.kit.app.get_app = lambda: app
        modules = {
            "isaacsim": isaac,
            "omni": omni,
            "omni.kit": omni.kit,
            "omni.kit.app": omni.kit.app,
        }
        output = self.root.parent / "run"
        env = Env(self.dataset.canonical_state("synthetic_demo"))
        with (
            patch.dict("sys.modules", modules),
            patch("signal.signal"),
            patch("roboodyssey.run.AgentProcess") as process,
            patch.object(
                backend_module, "make_env_factory",
                return_value=lambda *args: env,
            ),
            redirect_stdout(StringIO()),
        ):
            process.return_value.__enter__.return_value = Agent()
            self.assertEqual(
                main(["--dataset", str(self.root), "--run-directory", str(output),
                      "--agent-cmd", "python3 -m example.agent"]), 0
            )
        self.assertTrue(env.closed)
        self.assertTrue(app.closed)
        self.assertEqual(app.exit_code, 0)
        summary = json.loads((output / "summary.json").read_text())
        self.assertTrue(summary["evaluation_valid"])
        self.assertEqual(summary["task_score"], 1)
        self.assertEqual(
            json.loads((output / "status.json").read_text())["execution"], "completed"
        )

    def test_official_dry_run_uses_shared_validator_and_stops_on_errors(self):
        from roboodyssey.cli import main
        from contextlib import redirect_stdout
        from io import StringIO
        from unittest.mock import patch

        with (
            patch(
                "roboodyssey.validation.dataset.validate",
                return_value=["invalid fixture"],
            ) as validate,
            redirect_stdout(StringIO()),
        ):
            self.assertEqual(
                main(["run", "--dataset", str(self.root), "--official", "--dry-run"]), 1
            )
        validate.assert_called_once_with(self.dataset.root)

    def test_cli_rejects_missing_data_and_primitive_task_filters(self):
        from roboodyssey.cli import main
        from contextlib import redirect_stderr
        from io import StringIO

        for args in (
            ["run", "--dry-run"],
            ["doctor"],
            ["run", "--dataset", str(self.root), "--task", "ask", "--dry-run"],
        ):
            with self.subTest(args=args), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(args)
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
