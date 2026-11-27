"""Arrival evidence preserves walls while separating target and support geometry."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

from roboodyssey.contracts.doors import POLICY
from roboodyssey.contracts.raster import derive_arrival_navigation
from roboodyssey.evaluation.navigation import GridDistance, GoalRegion, OrderedRoute
from roboodyssey.evaluation.requests import evaluate_request, request_totals, SUCCESS_RULE_VERSION
from roboodyssey.evaluation.transfer import collision_digest
from roboodyssey.scenes.isaac_env import IsaacVLNEnv


def entity(oid, lower, upper, category='cabinet'):
    return {'object_id': oid, 'category': category, 'room_id': 'right',
            'geometry': {'world_aabb': {'min': lower, 'max': upper}}}


def goal(obj):
    return {'goal_id': 'visit', 'goal_kind': 'visit',
            'target': {'object_ids': [obj['object_id']]},
            'success_criteria': {'position_tolerance_m': .8},
            'expected_outcome': {'target_geometry': {'world_aabb_xy': obj['geometry']['world_aabb']}}}


class ArrivalSnapshotTests(unittest.TestCase):
    rooms = [
        {'room_id': 'left', 'boundary_polygon': [[0,0],[2,0],[2,4],[0,4]]},
        {'room_id': 'right', 'boundary_polygon': [[2.2,0],[5,0],[5,4],[2.2,4]]},
    ]

    def snapshot(self, objects, rooms=None):
        return derive_arrival_navigation(rooms or self.rooms, objects, ['target'],
                                         resolution=.1, robot_radius=.29)

    def test_wall_overlap_is_not_erased_with_target_occupancy(self):
        target = entity('target', [2.05,1.8,0], [2.5,2.2,1])
        snapshot = self.snapshot([target])
        grid = GridDistance(**snapshot)
        region = GoalRegion(grid, goal(target))
        self.assertTrue(grid.walkable([1.6,2]))
        self.assertFalse(region.reached([1.6,2]))
        self.assertTrue(region.reached([3.05,2]))
        x,y = grid.cell([2.1,2])
        self.assertTrue(snapshot['target_blocked']['target'][y,x])

    def test_table_support_does_not_hide_target_but_standing_cannot_enter_table(self):
        rooms = [{'room_id': 'right', 'boundary_polygon': [[0,0],[5,0],[5,4],[0,4]]}]
        target = entity('target', [2.4,1.8,.8], [2.6,2,1], 'book')
        table = entity('table', [2.2,1.6,0], [3,2.4,.8], 'table')
        grid = GridDistance(**self.snapshot([target, table], rooms))
        region = GoalRegion(grid, goal(target))
        self.assertTrue(region.reached([1.8,1.9]))
        self.assertFalse(region.reached([2.3,1.9]))

    def test_other_tall_obstacle_remains_in_target_approach_layer(self):
        rooms = [{'room_id': 'right', 'boundary_polygon': [[0,0],[5,0],[5,4],[0,4]]}]
        target = entity('target', [2.4,1.8,.8], [2.6,2,1], 'book')
        screen = entity('screen', [2.1,0,0], [2.2,4,2], 'cabinet')
        grid = GridDistance(**self.snapshot([target, screen], rooms))
        self.assertTrue(grid.walkable([1.7,1.9]))
        self.assertFalse(GoalRegion(grid, goal(target)).reached([1.7,1.9]))

    def test_environment_snapshot_matches_actual_occupancy_and_rejects_mismatch(self):
        target = entity('target', [2.3,1.8,0], [2.5,2.2,1])
        snapshot = self.snapshot([target])
        env = IsaacVLNEnv.__new__(IsaacVLNEnv)
        env._np = np
        env._metadata = {'rooms': deepcopy(self.rooms)}
        env._environment_setup = {'policy': POLICY, 'doors': []}
        env._held_object = None
        env._static_base = snapshot['blocked'].copy()
        env._static = env._static_base
        env._dynamic = np.zeros_like(env._static_base)
        env._door_open, env._door_cells = set(), {}
        env._origin, env._resolution = np.array(snapshot['origin']), .1
        env.config = SimpleNamespace(robot_radius=.29)
        env.runtime_state_v5 = lambda: {'objects': {'target': deepcopy(target)}}
        actual = env.evaluation_arrival([goal(target)])
        np.testing.assert_array_equal(actual['blocked'], snapshot['blocked'])
        np.testing.assert_array_equal(actual['target_blocked']['target'], snapshot['target_blocked']['target'])
        env._static_base[10,10] = ~env._static_base[10,10]
        with self.assertRaisesRegex(ValueError, 'differs from actual'):
            env.evaluation_arrival([goal(target)])

    def test_cached_route_cannot_prove_arrival_at_a_different_target(self):
        original = entity('target', [1,1,0], [1.2,1.2,1])
        moved = entity('target', [4,1,0], [4.2,1.2,1])
        grid = GridDistance(np.zeros((40,60), bool), [0,0], .1, .29)
        route = OrderedRoute(grid, [goal(original)])
        state = {'objects': {'target': moved}}
        outcome = evaluate_request({'goals': [goal(moved)]}, state,
                                   [{'position': [1.5,1.1,0]}], [], [], None,
                                   navigation=route)
        self.assertFalse(outcome['success'])
        self.assertEqual(outcome['goals'][0]['navigation_evidence_status'],
                         'navigation_reference_does_not_match_goal')

    def test_collision_digest_normalizes_integer_snapshot_origin(self):
        snapshot = {'blocked': np.zeros((4,4), bool), 'origin': [0,0],
                    'resolution': 1, 'radius': 0}
        self.assertEqual(collision_digest(snapshot), collision_digest(GridDistance(**snapshot)))

    def test_mixed_success_rule_versions_invalidate_aggregate(self):
        rows = [{'task_id': str(i), 'scored': True, 'success': True, 'success_rule_version': version}
                for i, version in enumerate((SUCCESS_RULE_VERSION, 'contextual_v3_correctness_only_v1'))]
        totals = request_totals(rows)
        self.assertFalse(totals['evaluation_valid'])
        self.assertIsNone(totals['task_success_rate'])


if __name__ == '__main__':
    unittest.main()
