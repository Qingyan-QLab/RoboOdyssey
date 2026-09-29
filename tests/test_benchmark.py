import json
import numpy as np
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from benchmark.agents.protocol import validate_response
from benchmark.evaluation.success import evaluate_embodied_task, evaluate_qa
from benchmark.evaluation.report import write_summary
from benchmark.runtime.session_runner import SessionRunner
from benchmark.runtime.reference import resolve_runtime_reference
from benchmark.scenes.dataset_loader import BenchmarkDataset, SessionRecord
from benchmark.scenes.state_composer import compose_session_state
from examples.holoagent_benchmark_server import (
    CompositeHoloAgentPlanner,
    FsrVlnPlanner,
    LiveOvoPlanner,
    ManipulationStateMachine,
    assert_public_agent_request,
)
from examples.holoagent_agent import normalize_response

DATASET = Path(__file__).resolve().parents[1] / "dataset" / "scene000430_after_changes"


class DatasetTests(unittest.TestCase):
    def test_example_dataset_shape_and_chain(self):
        dataset = BenchmarkDataset(DATASET)
        self.assertEqual(dataset.scene_ids(), ["scene000430"])
        sessions = dataset.sessions("scene000430")
        self.assertEqual(len(sessions), 10)
        parent = None
        for session in sessions:
            self.assertEqual(session.metadata["dataset_schema"], "scene_package_v2")
            self.assertEqual(session.metadata["parent_session_id"], parent)
            self.assertEqual(
                {task["task_id"] for task in session.tasks},
                {task["task_id"] for task in session.ground_truth()["tasks"]},
            )
            parent = session.session_id

    def test_room_instruction_is_compact(self):
        session = BenchmarkDataset(DATASET).sessions("scene000430")[0]
        rooms = session.metadata["instruction"]["rooms"]
        self.assertTrue(rooms)
        self.assertTrue(all(room["render_region"] == {"type": "room"} for room in rooms))
        self.assertTrue(all("camera_trajectory" not in room for room in rooms))

    def test_canonical_state_comes_from_metadata(self):
        state = BenchmarkDataset(DATASET).canonical_state("scene000430")
        self.assertEqual(state["scene_id"], "scene000430")
        self.assertGreater(len(state["objects"]), 300)
        self.assertTrue(all("pose" in item for item in state["objects"].values()))

    def test_mutation_does_not_modify_input(self):
        state = {"objects": {"cup": {"room_id": "a", "pose": {"position": [0, 0, 0]}}}}
        mutation = {"changes": [{"object_id": "cup", "pose_after": {"position": [1, 2, 3]},
                                 "relation_after": {"room_id": "b", "relation_type": "in_room"}}]}
        result = compose_session_state(state, mutation)
        self.assertEqual(state["objects"]["cup"]["room_id"], "a")
        self.assertEqual(result["objects"]["cup"]["room_id"], "b")
        self.assertEqual(result["objects"]["cup"]["pose"]["position"], [1, 2, 3])


class MetricTests(unittest.TestCase):
    def test_navigation_and_collision_policy(self):
        task = {"task_type": "navigation"}
        gt = {"expected_outcome": {"goal_pose": {"position": [1, 2, 0]}},
              "success_criteria": {"position_tolerance_m": 0.3, "collision_allowed": False},
              "reference_trajectory": {"length_m": 1.0}}
        event = {"before": {"position": [1.0, 2, 1.5]},
                 "after": {"position": [1.1, 2, 1.5]}, "result": {"collision": False}}
        self.assertTrue(evaluate_embodied_task(task, gt, [event], {})["success"])
        self.assertAlmostEqual(evaluate_embodied_task(task, gt, [event], {})["path_length_m"], 0.1)
        event["result"]["collision"] = True
        self.assertFalse(evaluate_embodied_task(task, gt, [event], {})["success"])

    def test_navigation_uses_aabb_distance_for_large_targets(self):
        task = {"task_type": "navigation"}
        gt = {"expected_outcome": {
                  "goal_pose": {"position": [5, 5, 0]},
                  "target_geometry": {"type": "aabb_2d",
                                      "world_aabb_xy": {"min": [0, 0], "max": [4, 4]}}},
              "success_criteria": {"target_distance_type": "aabb_2d",
                                    "position_tolerance_m": 0.2,
                                    "collision_allowed": True},
              "reference_trajectory": {"length_m": 1.0}}
        event = {"before": {"position": [4.0, 2.0, 1.5]},
                 "after": {"position": [4.1, 2.0, 1.5]}, "result": {"collision": False}}
        result = evaluate_embodied_task(task, gt, [event], {})
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["final_distance_m"], 0.1)

    def test_manipulation_room_relation(self):
        task = {"task_type": "manipulation"}
        gt = {"expected_outcome": {"state_changes": [{
            "object_id": "cup", "relation_after": {"type": "in_room", "room_id": "kitchen"},
        }]}}
        state = {"objects": {"cup": {"room_id": "kitchen"}}}
        self.assertTrue(evaluate_embodied_task(task, gt, [], state)["success"])

    def test_facility_state_uses_runtime_door_state(self):
        task = {"task_type": "manipulation"}
        gt = {"expected_outcome": {"state_changes": [{
            "object_id": "door", "state_after": "open",
        }]}}
        state = {"objects": {"door": {"door_open": True}}}
        self.assertTrue(evaluate_embodied_task(task, gt, [], state)["success"])

    def test_qa_accepted_answer(self):
        gt = {"expected_outcome": {"answer": "living room", "accepted_answers": ["lounge"]}}
        self.assertTrue(evaluate_qa("Lounge.", gt)["success"])

    def test_qa_semantic_answer_contains_reference(self):
        gt = {"expected_outcome": {"answer": "bathroom", "accepted_answers": []}}
        result = evaluate_qa("The toilet is located in the bathroom.", gt)
        self.assertTrue(result["semantic_match"])
        self.assertTrue(result["success"])
        self.assertFalse(evaluate_qa("The toilet is in the bedroom.", gt)["success"])

    def test_protocol(self):
        validate_response("step", {"action": {"action": "pick", "object_id": "cup"}})
        validate_response("qa", {"answer": "kitchen"})
        with self.assertRaises(ValueError):
            validate_response("step", {"action": {"action": "teleport"}})

    def test_runtime_reference_starts_at_actual_pose(self):
        class ReferenceEnvironment:
            _navigation = {"resolution": 1.0, "origin_xy": [0.0, 0.0], "floor_height": 0.0}
            _static = np.zeros((3, 3), dtype=np.uint8)
            _dynamic = np.zeros((3, 3), dtype=np.uint8)

            def get_agent_pose(self):
                return {"position": [0.5, 0.5, 1.5]}

            def runtime_state(self):
                return {"objects": {}}

        task = {"task_type": "navigation", "task_family": "point_navigation",
                "target": {"object_ids": []}}
        gt = {"expected_outcome": {"goal_pose": {"position": [2.5, 0.5, 0.0]}},
              "success_criteria": {}}
        resolved = resolve_runtime_reference(task, gt, ReferenceEnvironment())
        self.assertEqual(resolved["reference_trajectory"]["points"][0]["position"][:2],
                         [0.5, 0.5])
        self.assertAlmostEqual(resolved["reference_trajectory"]["length_m"], 2.0)

    def test_holoagent_gateway_rejects_private_ground_truth(self):
        with self.assertRaisesRegex(ValueError, "private benchmark field"):
            assert_public_agent_request({"type": "step", "ground_truth": {"answer": "x"}})

    def test_holoagent_parses_public_point_navigation_target(self):
        self.assertEqual(
            FsrVlnPlanner._point_target("Go to the point at (19.03, -6.91) metres."),
            (19.03, -6.91),
        )

    def test_holoagent_normalizes_common_action_aliases(self):
        response = normalize_response(
            {"action": {"action": "pickup", "arguments": {"target": "cup"}}})
        self.assertEqual(response["action"]["action"], "pick")
        self.assertEqual(response["action"]["object_id"], "cup")

    def test_manipulation_state_machine_uses_local_affordances(self):
        class Vlm:
            def request(self, payload):
                return {"action": {"action": "move_forward"}}

        planner = ManipulationStateMachine(Vlm())
        payload = {"type": "step", "task_type": "manipulation", "task_id": "m",
                   "instruction": "Pick up the computer in the study, carry it to master bedroom 2.",
                   "session_instruction": {"rooms": [{"room_id": "studyRoom1"},
                                                       {"room_id": "masterBedroom2"}]},
                   "observation": {"room_id": "studyRoom1", "held_object": None,
                                   "nearby_manipulable_objects": [{
                                       "object_id": "studyRoom1_computer002",
                                       "category": "computer", "distance_m": 1.0,
                                       "reachable": True}]}}
        self.assertEqual(planner.plan(payload)["action"],
                         {"action": "pick", "object_id": "studyRoom1_computer002"})
        payload["observation"] = {"room_id": "masterBedroom2",
                                  "held_object": "studyRoom1_computer002"}
        self.assertEqual(planner.plan(payload)["action"],
                         {"action": "place", "room_id": "masterBedroom2"})

    def test_manipulation_state_machine_opens_closed_door_while_carrying(self):
        class Vlm:
            def request(self, payload):
                return {"action": {"action": "move_forward"}}

        planner = ManipulationStateMachine(Vlm())
        payload = {
            "type": "step", "task_type": "manipulation", "task_id": "m",
            "instruction": "Carry the computer to masterBedroom2.",
            "session_instruction": {"rooms": [{"room_id": "masterBedroom2"}]},
            "observation": {
                "room_id": "studyRoom1", "held_object": "computer",
                "nearby_doors": [{"object_id": "study_door", "open": False,
                                  "distance_m": 0.8}],
            },
        }
        self.assertEqual(planner.plan(payload)["action"],
                         {"action": "open", "object_id": "study_door"})

    def test_live_ovo_rejects_false_arrival_instead_of_idling(self):
        mapper = SimpleNamespace(obj_detect_track=None, slam_backbone=None, frame_id=2,
                                 rgb_message_count=1, depth_message_count=1,
                                 pose_message_count=1, processing_frame_id=2,
                                 initialization_error=None, processing_error=None)
        planner = LiveOvoPlanner(mapper)
        planner._task_instruction = "Find bookshelf"
        planner._target = (1.0, 1.0)
        planner._candidate_object_id = 7
        payload = {"type": "step", "task_type": "navigation",
                   "instruction": "Find bookshelf",
                   "observation": {"pose": {"position": [1.0, 1.0, 1.5],
                                               "orientation": [0.0, 0.0, 0.0, 1.0],
                                               "yaw_degrees": 0.0}}}
        first = planner.plan(payload)
        second = planner.plan(payload)
        self.assertEqual(first["gateway_status"], "candidate_reached_pending_evaluation")
        self.assertEqual(second["gateway_status"], "false_candidate_rejected")
        self.assertEqual(second["action"]["action"], "turn_left")
        self.assertIn(7, planner._attempted_object_ids)

    def test_live_ovo_explores_when_target_is_unresolved(self):
        mapper = SimpleNamespace(obj_detect_track=None, slam_backbone=None, frame_id=2,
                                 rgb_message_count=1, depth_message_count=1,
                                 pose_message_count=1, processing_frame_id=2,
                                 initialization_error=None, processing_error=None)
        response = LiveOvoPlanner(mapper).plan(
            {"type": "step", "task_type": "navigation", "instruction": "Find bookshelf"})
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
                if payload["type"] == "qa":
                    return {"answer": "kitchen"}
                return {"action": {"action": "pick", "target": "vase"}}

        navigation, vlm = Planner({"action": {"action": "move_forward"}}), Vlm()
        planner = CompositeHoloAgentPlanner(navigation, vlm)
        self.assertEqual(planner.plan({"type": "step", "task_type": "navigation"})["action"]["action"],
                         "move_forward")
        self.assertEqual(planner.plan({"type": "step", "task_type": "manipulation"})["action"]["action"],
                         "pick")
        self.assertEqual(planner.plan({"type": "qa", "question": "Where?"})["answer"],
                         "kitchen")


class FakeAgent:
    def __init__(self):
        self.requests = []

    def request(self, payload, on_wait=None):
        self.requests.append(payload)
        if payload["type"] == "session_start":
            return {"ok": True, "memory_update": {"seen": True}}
        if payload["type"] == "qa":
            return {"answer": "yes"}
        if payload["task_type"] == "navigation":
            return {"action": {"action": "no_op"}}
        return {"action": {"action": "place", "room_id": "room_b"}}


class FakeEnvironment:
    def __init__(self, state):
        self.state = json.loads(json.dumps(state))
        self.pose = {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]}
        self.task = {}

    def apply_state(self, state):
        self.state = json.loads(json.dumps(state))

    def set_task(self, task):
        self.task = task

    def observe(self):
        return {"images": {}, "pose": self.pose}

    def get_agent_pose(self):
        return dict(self.pose)

    def step(self, action):
        if action.get("action") == "place":
            self.state["objects"]["cup"]["room_id"] = action["room_id"]
        return self.observe(), 0.0, False, False, {"success": True}

    def runtime_state(self):
        return self.state

    def state_changes(self, initial):
        if initial["objects"]["cup"]["room_id"] == self.state["objects"]["cup"]["room_id"]:
            return []
        return [{"object_id": "cup", "after": {"room_id": self.state["objects"]["cup"]["room_id"]}}]

    def close(self):
        pass


class RunnerTests(unittest.TestCase):
    def test_tasks_execute_in_authored_order_and_outputs_are_minimal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session_dir = root / "session_000"
            session_dir.mkdir()
            tasks = [
                {"task_id": "n", "task_type": "navigation", "task_family": "point_navigation",
                 "capabilities": ["navigation"], "instruction": "Go.", "depends_on": []},
                {"task_id": "q", "task_type": "qa", "task_family": "spatial_relation_qa",
                 "capabilities": ["qa"], "question": "Present?", "depends_on": ["n"]},
                {"task_id": "m", "task_type": "manipulation", "task_family": "object_transfer",
                 "capabilities": ["navigation", "manipulation"], "instruction": "Move cup.",
                 "target": {"object_ids": ["cup"], "room_ids": ["room_a", "room_b"]}, "depends_on": ["q"]},
            ]
            metadata = {"schema_version": 2, "dataset_schema": "scene_package_v2",
                        "scene_id": "scene", "session_id": "session_000", "parent_session_id": None,
                        "instruction": None, "tasks": tasks, "ground_truth_file": "ground_truth.json"}
            gt = {"tasks": [
                {"task_id": "n", "expected_outcome": {"goal_pose": {"position": [0, 0, 0]}},
                 "success_criteria": {"position_tolerance_m": 0.1, "collision_allowed": True},
                 "reference_trajectory": {"length_m": 0}},
                {"task_id": "q", "expected_outcome": {"answer": "yes", "accepted_answers": []}},
                {"task_id": "m", "expected_outcome": {"state_changes": [
                    {"object_id": "cup", "relation_after": {"type": "in_room", "room_id": "room_b"}},
                ]}},
            ]}
            (session_dir / "ground_truth.json").write_text(json.dumps(gt))
            record = SessionRecord("scene", session_dir, metadata)
            agent = FakeAgent()
            output = root / "results"
            runner = SessionRunner(output, agent, max_steps=2)
            state = {"objects": {"cup": {"room_id": "room_a", "pose": {"position": [0, 0, 0]}}}}
            runner.run_scene("scene", [record], state,
                             lambda _scene, initial, _session: FakeEnvironment(initial))
            self.assertEqual([request["type"] for request in agent.requests],
                             ["session_start", "step", "qa", "step"])
            for request in agent.requests:
                assert_public_agent_request(request)
            session_output = output / "scenes" / "scene" / "sessions" / "session_000"
            self.assertEqual(
                {path.name for path in session_output.iterdir()},
                {"action_changelog.jsonl", "state_changelog.json", "metrics.json",
                 "session_meta.json", "agent_io.jsonl"},
            )
            metrics = json.loads((session_output / "metrics.json").read_text())
            self.assertTrue(metrics["success"])
            self.assertEqual([item["task_id"] for item in metrics["task_results"]], ["n", "q", "m"])

    def test_summary_counts_missing_tasks_as_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = write_summary(BenchmarkDataset(DATASET), directory)
            authored = sum(len(session.tasks)
                           for session in BenchmarkDataset(DATASET).sessions("scene000430"))
            self.assertEqual(summary["tasks_evaluated"], authored)
            self.assertEqual(summary["task_success_rate"], 0.0)
            self.assertEqual(summary["sessions_evaluated"], 10)


if __name__ == "__main__":
    unittest.main()
