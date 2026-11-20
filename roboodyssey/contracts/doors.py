"""Frozen-data runtime helpers. Adapted from MIT-licensed sources; see LICENSE."""
from __future__ import annotations

import hashlib

import json

POLICY = 'environment_open_doors_v1'

def profile_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def door_records(navigation: dict) -> dict:
    return {d['object_id']: d for d in navigation.get('environment_setup', {}).get('doors', [])}

def collision_bounds(entity: dict, navigation: dict) -> list[dict]:
    door = door_records(navigation).get(entity['object_id'])
    if door:
        return [c['world_aabb'] for c in door['colliders']]
    bounds = entity.get('geometry', {}).get('world_aabb') or entity.get('world_aabb')
    return [bounds] if bounds else []

def apply_open_doors(stage, profile: dict | None) -> dict:
    """Author the exact saved poses into the session layer, retaining collisions.

    Safe to repeat. The source USDZ stays unchanged. Bodies are held kinematic
    and joints disabled so physics cannot close an environment-opened door.
    """
    if not profile:
        return {'policy': 'authored', 'door_count': 0}
    if profile.get('policy') != POLICY:
        raise ValueError('unsupported environment door policy')
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for door in profile['doors']:
            root = stage.GetPrimAtPath(door['prim_path'])
            if not root:
                raise ValueError('missing door prim: ' + door['prim_path'])
            for prim in Usd.PrimRange(root):
                if prim.IsA(UsdPhysics.Joint):
                    UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
            for body in door['bodies']:
                prim = stage.GetPrimAtPath(body['prim_path'])
                if not prim:
                    raise ValueError('missing door body: ' + body['prim_path'])
                xform = UsdGeom.Xformable(prim)
                op = xform.MakeMatrixXform()
                xform.SetResetXformStack(True)
                op.Set(Gf.Matrix4d(*[v for row in body['world_matrix4x4'] for v in row]))
    return {'policy': POLICY, 'door_count': len(profile['doors']), 'profile_sha256': profile_digest(profile)}

def verify_open_doors(stage, profile: dict, *, tolerance: float = 1e-4) -> dict:
    """Independently compare composed USD collider bounds with planning geometry."""
    from pxr import Gf, Usd, UsdGeom
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    rows = []
    for door in profile['doors']:
        error = 0.
        for collider in door['colliders']:
            prim = stage.GetPrimAtPath(collider['prim_path'])
            if not prim:
                raise ValueError('missing collider: ' + collider['prim_path'])
            if 'face_index' in collider:
                mesh = UsdGeom.Mesh(prim)
                counts = list(mesh.GetFaceVertexCountsAttr().Get())
                indices = mesh.GetFaceVertexIndicesAttr().Get()
                points = mesh.GetPointsAttr().Get()
                index = collider['face_index']
                offset = sum(counts[:index])
                matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                vertices = [matrix.Transform(Gf.Vec3d(points[i])) for i in indices[offset:offset+counts[index]]]
                actual = {'min': [min(p[i] for p in vertices) for i in range(3)],
                          'max': [max(p[i] for p in vertices) for i in range(3)]}
            else:
                box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                actual = {'min': list(box.GetMin()), 'max': list(box.GetMax())}
            expected = collider['world_aabb']
            error = max(error, max(abs(actual[k][i] - expected[k][i]) for k in ('min', 'max') for i in range(3)))
        if error > tolerance:
            raise ValueError(f'open-door USD/planning geometry mismatch: {door["object_id"]}: {error}')
        rows.append({'object_id': door['object_id'], 'max_bounds_error_m': error})
    return {'status': 'passed', 'scope': 'usd_collider_bounds_not_physics_rollout', 'doors': rows}
