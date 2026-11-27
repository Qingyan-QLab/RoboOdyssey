"""Arrival is tied to obstacle-verified regions, including nearby walls."""
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from roboodyssey.evaluation.navigation import GoalRegion, GridDistance, OrderedRoute


def goal(box=None, *, kind='visit', tolerance=.8):
    box = box or {'min': [2.35, 1.95, 0.0], 'max': [2.55, 2.15, 1.0]}
    result = {'goal_id': 'g', 'goal_kind': kind, 'target': {'object_ids': ['target']},
              'success_criteria': {'position_tolerance_m': tolerance},
              'expected_outcome': {'target_geometry': {'world_aabb_xy': box}}}
    if kind == 'revisit_place':
        result['fixed_target_geometry'] = {'world_aabb': box}
    return result


def wall_grid(*, opening=None, radius=.29, target=True):
    wall = np.zeros((45, 70), dtype=bool)
    wall[:, 20:22] = True
    if opening:
        wall[opening[0]:opening[1], 20:22] = False
    raw = wall.copy()
    if target:
        raw[19:22, 23:26] = True
    return GridDistance(raw, [0, 0], .1, radius,
                        target_blocked={'target': wall})


class ObstacleAwareArrivalTests(unittest.TestCase):
    def test_complete_wall_rejects_nearby_robot_and_wrong_side_goal_seeds(self):
        grid = wall_grid()
        route = OrderedRoute(grid, [goal()])
        wrong_side = [1.65, 2.05]
        self.assertTrue(grid.walkable(wrong_side))
        self.assertFalse(route.reached(wrong_side, 0))
        self.assertFalse(route.masks[0][:, :20].any())
        self.assertTrue(math.isinf(route.distance(wrong_side, 0)))
        self.assertTrue(route.reached([2.95, 2.05], 0))

    def test_detour_through_door_is_reachable_but_not_already_arrived(self):
        grid = wall_grid(opening=(30, 40))
        route = OrderedRoute(grid, [goal()])
        wrong_side = [1.65, 2.05]
        self.assertFalse(route.reached(wrong_side, 0))
        distance = route.distance(wrong_side, 0)
        self.assertTrue(math.isfinite(distance))
        self.assertGreater(distance, .8)
        self.assertTrue(route.reached([2.95, 2.05], 0))

    def test_narrow_opening_does_not_admit_robot_sized_approach(self):
        route = OrderedRoute(wall_grid(opening=(19, 22)), [goal()])
        self.assertFalse(route.reached([1.65, 2.05], 0))
        self.assertTrue(math.isinf(route.distance([1.65, 2.05], 0)))

    def test_slit_immediately_before_target_is_not_hidden_by_stop_clearance(self):
        wall = np.zeros((40, 60), dtype=bool)
        wall[:, 20] = True
        wall[19:22, 20] = False  # only .3m wide, narrower than the avatar
        raw = wall.copy()
        raw[20:21, 21:24] = True
        grid = GridDistance(raw, [0, 0], .1, .29, target_blocked={'target': wall})
        target = goal({'min': [2.11, 2.01], 'max': [2.3, 2.09]})
        point = [1.65, 2.05]
        self.assertTrue(grid.walkable(point))
        self.assertTrue(grid.unobstructed_segment(point, [2.11, 2.05], wall))
        self.assertFalse(GoalRegion(grid, target).reached(point))
        self.assertTrue(math.isinf(grid.between(point, [2.95, 2.05])))

    def test_small_wall_backed_target_can_be_approached_from_its_front(self):
        wall = np.zeros((40, 60), dtype=bool)
        wall[:, 20:22] = True
        raw = wall.copy()
        raw[20:21, 22:24] = True
        grid = GridDistance(raw, [0, 0], .1, .29, target_blocked={'target': wall})
        target = goal({'min': [2.21, 2.01], 'max': [2.3, 2.09]})
        point = [2.75, 2.05]
        # A circle centred on the surface would overlap the backing wall, but
        # no obstacle lies between the standing robot and the front surface.
        self.assertFalse(grid._clear(wall, grid.radius, [2.3, 2.05], boundary=True))
        self.assertTrue(GoalRegion(grid, target).reached(point))

    def test_no_wall_accepts_same_distance_and_does_not_double_tolerance(self):
        blocked = np.zeros((45, 70), dtype=bool)
        route = OrderedRoute(GridDistance(blocked, [0, 0], .1, .29), [goal()])
        self.assertTrue(route.reached([1.65, 2.05], 0))
        self.assertFalse(route.reached([.95, 2.05], 0))
        self.assertGreater(route.distance([.95, 2.05], 0), 0)

    def test_occupied_target_does_not_require_avatar_to_enter_object(self):
        grid = wall_grid()
        route = OrderedRoute(grid, [goal()])
        self.assertFalse(grid.walkable([2.45, 2.05]))
        self.assertFalse(route.reached([2.45, 2.05], 0))
        self.assertTrue(route.reached([2.95, 2.05], 0))
        # Lacking a target exclusion layer is conservative, never a fallback
        # to the old Euclidean success rule.
        raw_only = GridDistance(grid.raw, grid.origin, grid.resolution, grid.radius)
        self.assertFalse(GoalRegion(raw_only, goal()).reached([2.95, 2.05]))

    def test_target_exclusion_keeps_wall_in_overlapping_target_box(self):
        grid = wall_grid(target=False)
        # A large object bounding box overlaps a structural wall. The
        # separate target layer must retain that wall even inside the AABB.
        box = {'min': [2.05, 1.95], 'max': [2.55, 2.15]}
        route = OrderedRoute(grid, [goal(box)])
        self.assertFalse(route.reached([1.65, 2.05], 0))
        np.testing.assert_array_equal(grid.target_blocked['target'][:, 20:22], True)

    def test_target_height_layer_allows_tabletop_but_keeps_wall(self):
        raw = np.zeros((45, 70), dtype=bool)
        raw[17:25, 23:30] = True  # supporting table and the target
        wall = np.zeros_like(raw)
        grid = GridDistance(raw, [0, 0], .1, .29, target_blocked={'target': wall})
        self.assertTrue(GoalRegion(grid, goal()).reached([1.95, 2.05]))
        wall[:, 20:22] = True
        raw |= wall
        grid = GridDistance(raw, [0, 0], .1, .29, target_blocked={'target': wall})
        self.assertFalse(GoalRegion(grid, goal()).reached([1.65, 2.05]))

    def test_corner_touching_ray_cannot_slip_between_obstacles(self):
        raw = np.zeros((40, 40), dtype=bool)
        raw[19, 20] = raw[20, 19] = True
        box = {'min': [2.15, 2.15], 'max': [2.15, 2.15]}
        grid = GridDistance(raw, [0, 0], .1)
        self.assertFalse(GoalRegion(grid, goal(box)).reached([1.95, 1.95]))

    def test_revisit_uses_original_place_geodesic_without_cross_wall_seeds(self):
        box = {'min': [2.9, 2.0], 'max': [3.0, 2.1]}
        target = goal(box, kind='revisit_place', tolerance=1.4)
        grid = wall_grid(target=False)
        route = OrderedRoute(grid, [target])
        self.assertFalse(route.reached([1.65, 2.05], 0))
        self.assertFalse(route.masks[0][:, :20].any())
        self.assertTrue(route.reached([3.55, 2.05], 0))
        self.assertTrue(route.reached([2.95, 2.05], 0))

    def test_empty_arrival_region_is_explicitly_unreachable(self):
        raw = np.ones((10, 10), dtype=bool)
        grid = GridDistance(raw, [0, 0], .1)
        route = OrderedRoute(grid, [goal({'min': [.4, .4], 'max': [.5, .5]})])
        self.assertFalse(route.reached([.45, .45], 0))
        self.assertFalse(route.masks[0].any())
        self.assertTrue(math.isinf(route.distance([.45, .45], 0)))

    def test_reference_round_trip_preserves_target_layers_and_arrival(self):
        grid = wall_grid(opening=(30, 40))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'reference.npz'
            grid.save(path)
            restored = GridDistance.load(path)
        np.testing.assert_array_equal(grid.raw, restored.raw)
        np.testing.assert_array_equal(grid.target_blocked['target'], restored.target_blocked['target'])
        original = OrderedRoute(grid, [goal()])
        replay = OrderedRoute(restored, [goal()])
        self.assertEqual(original.reached([1.65, 2.05], 0), replay.reached([1.65, 2.05], 0))
        self.assertEqual(original.distance([1.65, 2.05], 0), replay.distance([1.65, 2.05], 0))

    def test_cached_region_does_not_change_when_callers_mutate_goal(self):
        target = goal()
        grid = wall_grid()
        route = OrderedRoute(grid, [target])
        region = GoalRegion(grid, target)
        target['expected_outcome']['target_geometry']['world_aabb_xy']['min'][0] = 1.
        target['success_criteria']['position_tolerance_m'] = 100.
        target['target']['object_ids'][0] = 'different_object'
        self.assertEqual(route.goals[0]['target']['object_ids'], ['target'])
        self.assertEqual(route.goals[0]['success_criteria']['position_tolerance_m'], .8)
        self.assertEqual(route.regions[0].box['min'][0], 2.35)
        self.assertEqual(region.box['min'][0], 2.35)
        self.assertFalse(route.reached([1.65, 2.05], 0))


if __name__ == '__main__':
    unittest.main()
