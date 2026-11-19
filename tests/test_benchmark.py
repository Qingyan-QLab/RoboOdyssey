import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from benchmark.agents.protocol import validate_response
from benchmark.scenes.dataset_loader import BenchmarkDataset
from examples.holoagent_benchmark_server import (
    CompositeHoloAgentPlanner,
    FsrVlnPlanner,
    LiveOvoPlanner,
    ManipulationStateMachine,
    assert_public_agent_request,
)
from examples.holoagent_agent import normalize_response


class MetricTests(unittest.TestCase):
    def test_protocol(self):
        validate_response("step", {"action": {"action": "pick", "target": "cup"}})
        validate_response("step", {"action": "no_op", "answer": "kitchen"})
        with self.assertRaises(ValueError):
            validate_response("step", {"action": {"action": "teleport"}})

    def test_holoagent_gateway_rejects_private_ground_truth(self):
        with self.assertRaisesRegex(ValueError, "private benchmark field"):
            assert_public_agent_request(
                {"type": "step", "ground_truth": {"answer": "x"}}
            )

    def test_holoagent_parses_public_point_navigation_target(self):
        self.assertEqual(
            FsrVlnPlanner._point_target("Go to the point at (19.03, -6.91) metres."),
            (19.03, -6.91),
        )

    def test_holoagent_normalizes_common_action_aliases(self):
        response = normalize_response(
            {"action": {"action": "pickup", "arguments": {"target": "cup"}}}
        )
        self.assertEqual(response["action"]["action"], "pick")
        self.assertEqual(response["action"]["target"], "cup")

    def test_manipulation_policy_uses_public_names_and_feedback(self):
        class Vlm:
            def request(self, payload):
                return {"action": {"action": "pick", "target": "reading book"}}

        planner = ManipulationStateMachine(Vlm())
        payload = {
            "type": "step",
            "task_type": "long_horizon",
            "capabilities": ["manipulation"],
            "task_id": "m",
            "instruction": "Carry the reading book.",
            "observation": {"holding_object": False},
        }
        self.assertEqual(
            planner.plan(payload)["action"],
            {"action": "pick", "target": "reading book"},
        )
        payload["observation"]["action_feedback"] = {"collision": True}
        self.assertEqual(planner.plan(payload)["action"]["action"], "turn_left")
        self.assertEqual(planner.plan(payload)["action"]["action"], "turn_right")
        planner.reset()
        self.assertEqual(planner.plan(payload)["action"]["action"], "turn_left")

    def test_holoagent_rejects_private_action_handles(self):
        with self.assertRaisesRegex(RuntimeError, "public names"):
            normalize_response({"action": {"action": "pick", "object_id": "private"}})

    def test_live_ovo_acknowledges_fresh_map_and_rejects_map_reuse(self):
        planner = LiveOvoPlanner(SimpleNamespace())
        payload = {
            "type": "session_start",
            "sequence_context": {"run_id": "first", "reset_memory": True},
        }
        self.assertEqual(planner.plan(payload)["sequence_reset_ack"], "first")
        payload["sequence_context"] = {"run_id": "first", "reset_memory": False}
        self.assertNotIn("sequence_reset_ack", planner.plan(payload))
        payload["sequence_context"] = {"run_id": "next", "reset_memory": True}
        with self.assertRaisesRegex(RuntimeError, "restart"):
            planner.plan(payload)

    def test_live_ovo_rejects_false_arrival_instead_of_idling(self):
        mapper = SimpleNamespace(
            obj_detect_track=None,
            slam_backbone=None,
            frame_id=2,
            rgb_message_count=1,
            depth_message_count=1,
            pose_message_count=1,
            processing_frame_id=2,
            initialization_error=None,
            processing_error=None,
        )
        planner = LiveOvoPlanner(mapper)
        planner._task_instruction = "Find bookshelf"
        planner._target = (1.0, 1.0)
        planner._candidate_object_id = 7
        payload = {
            "type": "step",
            "task_type": "long_horizon",
            "capabilities": ["navigation"],
            "instruction": "Find bookshelf",
            "observation": {
                "pose": {
                    "position": [1.0, 1.0, 1.5],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                    "yaw_degrees": 0.0,
                }
            },
        }
        first = planner.plan(payload)
        second = planner.plan(payload)
        self.assertEqual(
            first["gateway_status"], "candidate_reached_pending_evaluation"
        )
        self.assertEqual(second["gateway_status"], "false_candidate_rejected")
        self.assertEqual(second["action"]["action"], "turn_left")
        self.assertIn(7, planner._attempted_object_ids)

    def test_live_ovo_explores_when_target_is_unresolved(self):
        mapper = SimpleNamespace(
            obj_detect_track=None,
            slam_backbone=None,
            frame_id=2,
            rgb_message_count=1,
            depth_message_count=1,
            pose_message_count=1,
            processing_frame_id=2,
            initialization_error=None,
            processing_error=None,
        )
        response = LiveOvoPlanner(mapper).plan(
            {
                "type": "step",
                "task_type": "long_horizon",
                "capabilities": ["navigation"],
                "instruction": "Find bookshelf",
            }
        )
        self.assertEqual(response["gateway_status"], "live_ovo_target_unresolved")
        self.assertEqual(response["action"]["action"], "turn_left")

    def test_holoagent_composite_routes_capabilities(self):
        class Planner:
            def __init__(self, response):
                self.response = response
                self.requests = []

            def plan(self, payload):
                self.requests.append(payload)
                return dict(self.response)

        class Vlm:
            def __init__(self):
                self.requests = []

            def request(self, payload):
                self.requests.append(payload)
                if "qa" in payload.get("capabilities", []):
                    return {"action": "no_op", "answer": "kitchen"}
                return {"action": {"action": "pick", "target": "vase"}}

        navigation, vlm = Planner({"action": {"action": "move_forward"}}), Vlm()
        planner = CompositeHoloAgentPlanner(navigation, vlm)
        self.assertEqual(
            planner.plan(
                {
                    "type": "step",
                    "task_type": "long_horizon",
                    "capabilities": ["navigation"],
                }
            )["action"]["action"],
            "move_forward",
        )
        self.assertEqual(
            planner.plan(
                {
                    "type": "step",
                    "task_type": "long_horizon",
                    "capabilities": ["manipulation"],
                }
            )["action"]["action"],
            "pick",
        )
        self.assertEqual(
            planner.plan(
                {
                    "type": "step",
                    "task_type": "long_horizon",
                    "capabilities": ["qa"],
                    "instruction": "Where?",
                }
            )["answer"],
            "kitchen",
        )


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
        from benchmark.cli import main
        from contextlib import redirect_stdout
        from io import StringIO

        with redirect_stdout(StringIO()) as output:
            self.assertEqual(main(["run", "--dataset", str(self.root), "--dry-run"]), 0)
        self.assertIn("synthetic_demo: 1 sessions", output.getvalue())

    def test_cli_executes_current_sequence_and_closes_simulator(self):
        import json
        from contextlib import redirect_stdout
        from io import StringIO
        from types import ModuleType
        from unittest.mock import patch
        from benchmark.run import main
        from tests.test_contextual_requests import Agent, Env

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
            patch("benchmark.run.AgentProcess") as process,
            patch("benchmark.run._make_env_factory", return_value=lambda *args: env),
            redirect_stdout(StringIO()),
        ):
            process.return_value.__enter__.return_value = Agent()
            self.assertEqual(
                main(["--dataset", str(self.root), "--run-directory", str(output)]), 0
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
        from benchmark.cli import main
        from contextlib import redirect_stdout
        from io import StringIO
        from unittest.mock import patch

        with (
            patch(
                "benchmark.validation.dataset.validate",
                return_value=["invalid fixture"],
            ) as validate,
            redirect_stdout(StringIO()),
        ):
            self.assertEqual(
                main(["run", "--dataset", str(self.root), "--official", "--dry-run"]), 1
            )
        validate.assert_called_once_with(self.dataset.root)

    def test_cli_rejects_missing_data_and_primitive_task_filters(self):
        from benchmark.cli import main
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
