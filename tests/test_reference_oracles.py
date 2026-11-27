"""Regression cases migrated with the optional DataSynth reference models."""

"""CPU integration tests using the real pinned restools search backend."""
import json
import math
import unittest

import numpy as np
try:
    from restools.navigation import GridMap, is_segment_collision_free
except ImportError as error:
    raise unittest.SkipTest("install RoboOdyssey[reference] for these optional model regressions") from error

from roboodyssey.reference_oracles.navigation import NavigationSnapshot, PointNavigator, SearchLimits
from roboodyssey.reference_oracles.reference import (CandidateGoal, GoalRegion, Interaction, ReferenceOracle,
                                       ReferenceStage, TargetPose)


def snapshot(raw=None, **kwargs):
    return NavigationSnapshot(np.zeros((9, 12), bool) if raw is None else raw,
                              resolution_m=1, robot_radius_m=kwargs.pop("robot_radius_m", 0.1),
                              state_source=kwargs.pop("state_source", "runtime"), **kwargs)


def candidates(*points):
    return CandidateGoal(tuple(TargetPose(p) for p in points))


class NavigationTests(unittest.TestCase):
    def test_diagonal_metric_and_path_cost_agree(self):
        navigator = PointNavigator(snapshot())
        result = navigator.path((1.5, 1.5), (4.5, 3.5))
        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.cost_m, 1 + 2 * math.sqrt(2))
        self.assertAlmostEqual(result.cost_m, sum(math.dist(a, b) for a, b in
                                                 zip(result.points, result.points[1:])))

    def test_noncentre_endpoints_use_safe_connectors(self):
        raw = np.zeros((9, 12), bool)
        raw[:6, 5] = True
        snap = snapshot(raw)
        result = PointNavigator(snap).path((2.23, 2.17), (8.11, 2.31))
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.points[0], (2.23, 2.17))
        self.assertEqual(result.points[-1], (8.11, 2.31))
        grid = GridMap(raw)
        for a, b in zip(result.points, result.points[1:]):
            self.assertTrue(is_segment_collision_free(grid, a, b, .1))
        self.assertGreater(result.cost_m, math.dist(result.points[0], result.points[-1]))

    def test_closed_wall_and_invalid_start_are_distinct(self):
        raw = np.zeros((9, 12), bool)
        raw[:, 5] = True
        navigator = PointNavigator(snapshot(raw))
        self.assertEqual(navigator.distance((4.8, 2.5), (6.2, 2.5)).status, "unreachable")
        self.assertEqual(navigator.distance((5.5, 2.5), (6.5, 2.5)).status, "invalid")

    def test_circular_sweep_profile_is_explicitly_not_legacy_corner_rule(self):
        raw = np.zeros((8, 8), bool)
        raw[2, 2] = True
        result = PointNavigator(snapshot(raw, robot_radius_m=.6)).distance((3.5, 3.5), (2.5, 4.5))
        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.cost_m, math.sqrt(2))

    def test_snapshot_is_frozen_and_cache_does_not_cross_maps(self):
        raw = np.zeros((9, 12), bool)
        before = snapshot(raw)
        nav = PointNavigator(before)
        raw[:, 5] = True
        after = snapshot(raw, revision="1")
        self.assertNotEqual(before.digest, after.digest)
        self.assertEqual(nav.distance((2.5, 2.5), (8.5, 2.5)).status, "ok")
        self.assertEqual(PointNavigator(after).distance((2.5, 2.5), (8.5, 2.5)).status, "unreachable")

    def test_radius_changes_reachability(self):
        raw = np.zeros((9, 12), bool)
        raw[:, 5] = True
        raw[4, 5] = False
        self.assertEqual(PointNavigator(snapshot(raw)).distance((2.5, 4.5), (8.5, 4.5)).status, "ok")
        self.assertEqual(PointNavigator(snapshot(raw, robot_radius_m=.6)).distance(
            (2.5, 4.5), (8.5, 4.5)).status, "unreachable")

    def test_search_limit_is_not_unreachable(self):
        nav = PointNavigator(snapshot(), SearchLimits(max_expansions=1))
        self.assertEqual(nav.distance((1.5, 1.5), (8.5, 7.5)).status, "indeterminate")

    def test_invalid_snapshot_and_parameters_are_rejected(self):
        for changes in ({"occupancy_kind": "inflated"}, {"robot_radius_m": -1},
                        {"state_source": "unknown"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                snapshot(**changes)
        with self.assertRaises(ValueError):
            snapshot([[float("nan")]])
        with self.assertRaises(ValueError):
            SearchLimits(max_candidates_per_goal=0)


class TaskReferenceTests(unittest.TestCase):
    def test_ordered_goals_jointly_choose_endpoint_instead_of_nearest(self):
        result = ReferenceOracle().navigation(snapshot(), (1.5, 1.5), [
            candidates((2.5, 3.5), (4.5, 1.5)), candidates((8.5, 1.5))])
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.legs[0].target.position, (4.5, 1.5))
        self.assertAlmostEqual(result.cost, 7)

    def test_region_around_occupied_object_does_not_target_its_centre(self):
        raw = np.zeros((9, 12), bool)
        raw[4, 7] = True
        goal = GoalRegion((7.0, 4.0), (8.0, 5.0), .6)
        result = ReferenceOracle().navigation(snapshot(raw), (1.5, 4.5), [goal])
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.legs[0].target.position, (6.5, 4.5))
        self.assertAlmostEqual(result.cost, 5)

    def test_actual_start_in_overlapping_regions_has_zero_cost(self):
        goal = GoalRegion((1, 1), (2, 2))
        result = ReferenceOracle().navigation(snapshot(), (1.23, 1.77), [goal, goal])
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.cost, 0)
        self.assertEqual(len(result.legs), 2)

    def test_region_without_legal_samples_is_explicit(self):
        result = ReferenceOracle().navigation(snapshot(), (1.5, 1.5), [
            GoalRegion((5.1, 5.1), (5.2, 5.2))])
        self.assertEqual(result.status, "unreachable")
        self.assertIn("no_legal_goal_samples", result.reason)

    def test_candidate_and_search_budgets_never_return_approximate_success(self):
        for limits in (SearchLimits(max_candidates_per_goal=1), SearchLimits(max_search_calls=1)):
            result = ReferenceOracle(limits).navigation(snapshot(), (1.5, 1.5), [
                candidates((3.5, 1.5), (4.5, 1.5))])
            self.assertEqual(result.status, "indeterminate")
            self.assertIsNone(result.cost)
            self.assertFalse(result.legs)
            json.dumps(result.to_dict(), allow_nan=False)

    def test_joint_pick_and_place_costs_include_operations(self):
        snap = snapshot()
        result = ReferenceOracle().plan_stages((1.5, 1.5), [
            ReferenceStage(snap, candidates((2.5, 3.5), (4.5, 1.5)), Interaction("pick", "cup")),
            ReferenceStage(snap, CandidateGoal((TargetPose((8.5, 1.5), 0),)), Interaction("place", "cup"))
        ], metres_per_cost_unit=.5, cost_unit="equivalent_actions")
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.legs[0].target.position, (4.5, 1.5))
        self.assertEqual([leg.after.action for leg in result.legs], ["pick", "place"])
        self.assertAlmostEqual(result.cost, 7 / .5 + 2)

    def test_transfer_removes_target_and_held_state_omits_pick(self):
        raw = np.zeros((9, 12), bool)
        raw[2, 2] = True
        approach, carrying = snapshot(raw), snapshot(revision="after_pick")
        oracle = ReferenceOracle()
        kwargs = dict(object_id="cup", object_origin_xy=(2.5, 2.5),
                      placements=CandidateGoal((TargetPose((6.5, 2.5), 0),)), pick_radius_m=1.1)
        result = oracle.transfer(approach, carrying, (1.5, 2.5), **kwargs)
        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.cost, 5 / .25 + 2)
        self.assertNotEqual(result.legs[0].snapshot_digest, result.legs[1].snapshot_digest)
        held = oracle.transfer(approach, carrying, (6.5, 2.5), already_held=True, **kwargs)
        self.assertEqual(held.cost, 1)
        self.assertEqual(len(held.legs), 1)

    def test_transition_must_be_walkable_on_next_map(self):
        raw = np.zeros((9, 12), bool)
        raw[1, 4] = True
        result = ReferenceOracle().plan_stages((1.5, 1.5), [
            ReferenceStage(snapshot(), candidates((4.5, 1.5))),
            ReferenceStage(snapshot(raw), candidates((8.5, 1.5)))])
        self.assertEqual(result.status, "unreachable")

    def test_public_reference_contains_versions_and_no_nonfinite_numbers(self):
        result = ReferenceOracle().navigation(snapshot(), (1.5, 1.5), [candidates((2.5, 1.5))])
        data = json.loads(json.dumps(result.to_dict(), allow_nan=False))
        self.assertEqual(data["backend_version"], "0.0.1.3")
        self.assertEqual(data["reference_version"], "restools_holonomic_regions_v1")
        self.assertEqual(data["state_source"], "runtime")
        self.assertEqual(len(data["snapshot_digests"][0]), 64)

    def test_state_sources_and_cost_units_cannot_be_mixed(self):
        with self.assertRaises(ValueError):
            ReferenceOracle().plan_stages((1.5, 1.5), [
                ReferenceStage(snapshot(), candidates((2.5, 1.5))),
                ReferenceStage(snapshot(state_source="planned"), candidates((3.5, 1.5)))])
        with self.assertRaises(ValueError):
            ReferenceOracle().plan_stages((1.5, 1.5), [
                ReferenceStage(snapshot(), candidates((2.5, 1.5)), Interaction("pick", "cup"))])



"""Room answers must preserve actual task-boundary history and public names."""
import unittest

from roboodyssey.reference_oracles.answers import RoomSnapshot, TaskHistory, room_answer


class AnswerTests(unittest.TestCase):
    def setUp(self):
        self.current = RoomSnapshot({"cup": "kitchen"}, "runtime")
        self.history = TaskHistory("s0", "t0", RoomSnapshot({"cup": "study"}, "runtime"), self.current)
        self.names = {"study": "Study", "kitchen": "Kitchen"}

    def answer(self, query, **kwargs):
        return room_answer(query, "cup", self.current, area_names=self.names,
                           history=kwargs.pop("history", self.history),
                           source_task=kwargs.pop("source_task", ("s0", "t0")), **kwargs)

    def test_current_and_boundary_rooms_use_public_names(self):
        for query, expected in (("current_room", "Kitchen"), ("room_before", "Study"),
                                ("room_after", "Kitchen")):
            result = self.answer(query)
            self.assertEqual(result.status, "ok")
            self.assertEqual(result.value, expected)
            self.assertEqual(result.accepted_answers, (expected,))

    def test_changed_compares_current_with_source_after_not_before(self):
        self.assertFalse(self.answer("has_changed").value)
        self.assertTrue(self.answer("has_changed", invert=True).value)
        self.current = RoomSnapshot({"cup": "study"}, "runtime")
        self.assertTrue(self.answer("has_changed").value)

    def test_missing_mismatched_or_unissued_history_never_falls_back(self):
        for kwargs in ({"history": None}, {"source_task": ("other", "t0")},
                       {"history": TaskHistory("s0", "t0", self.current, self.current, issued=False)}):
            self.assertEqual(self.answer("room_before", **kwargs).status, "invalid")

    def test_missing_rooms_names_and_wrong_source_are_explicit(self):
        self.names.clear()
        self.assertEqual(self.answer("current_room").reason, "missing_public_area_name")
        self.current = RoomSnapshot({}, "runtime")
        self.assertEqual(self.answer("has_changed").reason, "missing_current_object_room")
        self.current = RoomSnapshot({"cup": "study"}, "planned")
        self.assertEqual(self.answer("room_before").reason, "state_source_mismatch")

    def test_room_snapshot_does_not_alias_mutable_input(self):
        rooms = {"cup": "study"}
        snap = RoomSnapshot(rooms, "runtime")
        rooms["cup"] = "kitchen"
        self.assertEqual(snap.object_rooms["cup"], "study")
        with self.assertRaises(TypeError):
            snap.object_rooms["cup"] = "kitchen"
