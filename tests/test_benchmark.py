import json
import tempfile
import unittest
from pathlib import Path

from benchmark.agents.protocol import validate_response
from benchmark.evaluation.success import evaluate_embodied_task, evaluate_qa
from benchmark.runtime.session_runner import SessionRunner
from benchmark.scenes.dataset_loader import BenchmarkDataset, SessionRecord
from benchmark.scenes.state_composer import compose_session_state

DATASET = Path(__file__).resolve().parents[1] / "dataset" / "scene000430_new"


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
        event = {"after": {"position": [1.1, 2, 1.5]}, "result": {"collision": False}}
        self.assertTrue(evaluate_embodied_task(task, gt, [event], {})["success"])
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
        event = {"after": {"position": [4.1, 2.0, 1.5]}, "result": {"collision": False}}
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

    def test_qa_accepted_answer(self):
        gt = {"expected_outcome": {"answer": "living room", "accepted_answers": ["lounge"]}}
        self.assertTrue(evaluate_qa("Lounge.", gt)["success"])

    def test_protocol(self):
        validate_response("step", {"action": {"action": "pick", "object_id": "cup"}})
        validate_response("qa", {"answer": "kitchen"})
        with self.assertRaises(ValueError):
            validate_response("step", {"action": {"action": "teleport"}})


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
            session_output = output / "scenes" / "scene" / "sessions" / "session_000"
            self.assertEqual(
                {path.name for path in session_output.iterdir()},
                {"action_changelog.jsonl", "state_changelog.json", "metrics.json",
                 "session_meta.json", "agent_io.jsonl"},
            )
            metrics = json.loads((session_output / "metrics.json").read_text())
            self.assertTrue(metrics["success"])
            self.assertEqual([item["task_id"] for item in metrics["task_results"]], ["n", "q", "m"])


if __name__ == "__main__":
    unittest.main()
