"""Private USD state and rendered instance evidence for contextual requests."""
from copy import deepcopy
import time


def runtime_state(env):
    from pxr import Usd, UsdGeom
    bbox=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render,UsdGeom.Tokens.proxy])
    result=deepcopy(env.runtime_state())
    for oid,obj in result['objects'].items():
        prim=env._prim(oid)
        world=bbox.ComputeWorldBound(prim).ComputeAlignedRange()
        minimum,maximum=world.GetMin(),world.GetMax()
        obj['geometry']={'world_aabb':{'min':[float(v) for v in minimum],'max':[float(v) for v in maximum]}}
        obj.pop('pose',None)
        obj['object_id']=oid
        obj['held']=env._held_object==oid
        obj['interaction_position']=env._get_object_world_position(oid).tolist()
        center=[(a+b)/2 for a,b in zip(minimum,maximum)]
        # Membership is based on actual geometry; fixture metadata may straddle
        # a boundary, in which case preserve the scene's authored membership.
        measured=env._room_at(center)
        if measured is not None:obj['room_id']=measured
    result['robot_pose']=env.get_agent_pose()
    result['floor_heights']={r['room_id']:r.get('floor_height',env._floor_height) for r in env._metadata['rooms']}
    return result


def observe(env, requirements):
    if not getattr(env,'_request_instance_evidence_enabled',False):
        # Instance IDs resolve to prim paths, unlike semantic class labels.
        env.camera.add_instance_id_segmentation_to_frame()
        env._request_instance_evidence_enabled=True
    previous=getattr(env,'_request_render_sequence',None)
    frame=None
    for _ in range(8):
        sequence=env.render_once()
        frame=env.camera.get_current_frame()
        stamp=frame.get('rendering_frame') if isinstance(frame,dict) else None
        segment=frame.get('instance_id_segmentation') if isinstance(frame,dict) else None
        # Camera.rendering_frame is a simulation-time ReferenceTime in Kit 6.
        # It remains constant during zero-time rendering. Freshness comes from
        # a successfully completed synchronous render, not advancing physics.
        if sequence is not None and sequence!=previous and isinstance(segment,dict) and segment.get('data') is not None:
            break
    else:
        raise RuntimeError('fresh rendered instance segmentation unavailable; inspection cannot be evaluated')
    env._request_render_sequence=sequence
    env.observation_frame_id=getattr(env,'observation_frame_id',0)+1
    labels=segment.get('info',{}).get('idToLabels')
    if not isinstance(labels,dict):raise RuntimeError('instance segmentation prim-path mapping missing')
    state=runtime_state(env)
    pose=env.get_agent_pose()
    evidence=[]
    pixels=env._np.asarray(segment['data'])
    for req in requirements:
        oid=req['object_id'];path=env._prim_paths[oid]
        ids=[]
        for value,label in labels.items():
            prim=label if isinstance(label,str) else label.get('primPath',label.get('prim_path',''))
            if prim==path or prim.startswith(path+'/'):ids.append(int(value))
        visible=int(env._np.isin(pixels,ids).sum()) if ids else 0
        box=state['objects'][oid]['geometry']['world_aabb']
        distance=sum(max(box['min'][i]-pose['position'][i],0,pose['position'][i]-box['max'][i])**2 for i in (0,1))**.5
        evidence.append({'object_id':oid,'frame_id':env.observation_frame_id,'rendering_frame':deepcopy(stamp),
            'completed_render_sequence':sequence,
            'observed_at_ns':time.monotonic_ns(),'current_frame':True,'visible_pixels':visible,
            'distance_m':distance,'pose':deepcopy(pose),'method':'rendered_instance_pixels'})
    # Keep ID maps, object bindings and evidence out of the public observation.
    observation=env.observe()
    if frame.get('rgba') is not None:
        observation['images']['front']=frame['rgba'][...,:3]
    elif frame.get('rgb') is not None:
        observation['images']['front']=frame['rgb']
    else:
        raise RuntimeError('inspection segmentation has no corresponding RGB frame')
    observation['depth']=frame.get('distance_to_image_plane')
    return observation,evidence


def apply_mutation(env, prepared):
    oid=prepared['object_id']
    if env._held_object==oid:raise RuntimeError('cannot mutate a held object')
    before=runtime_state(env)['objects'][oid]['geometry']['world_aabb']
    target=prepared['geometry']['world_aabb']
    delta=[(target['min'][i]+target['max'][i]-before['min'][i]-before['max'][i])/2 for i in range(3)]
    position=env._get_object_world_position(oid)
    env._set_object_world_position(oid,[position[i]+delta[i] for i in range(3)])
    env._objects[oid]['room_id']=prepared['destination_room_id']
    env._objects[oid]['held']=False
    env._objects[oid]['placement_surface']='floor'
    env._refresh_vlm_navigation()
    env.render_once()
