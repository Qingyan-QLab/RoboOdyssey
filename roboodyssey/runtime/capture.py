"""Optional original sensor storage, linked to the exact sampled actual state."""
from pathlib import Path
import json

import numpy as np

from ..artifacts.layout import json_digest, sha256_file, write_json, safe_id
from ..contracts.snapshots import state_version


class ObservationArchive:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, observation, *, scene_id, session_id, task_id, state, camera_pose=None):
        timing = observation['timing']
        frame_id = timing['frame_id']
        directory = self.root/safe_id(session_id)/f'{frame_id:09d}'
        directory.mkdir(parents=True, exist_ok=False)
        images = {}
        for name, value in observation.get('images', {}).items():
            if value is None:
                continue
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise ValueError('sensor arrays cannot contain Python objects')
            file = directory/(safe_id(str(name))+'.npy')
            np.save(file, array, allow_pickle=False)
            images[str(name)] = {'file': file.relative_to(self.root.parent).as_posix(),
                                'sha256': sha256_file(file), 'shape': list(array.shape), 'dtype': str(array.dtype)}
        depth = None
        if observation.get('depth') is not None:
            array = np.asarray(observation['depth'])
            if array.dtype.hasobject:
                raise ValueError('depth cannot contain Python objects')
            file = directory/'depth.npy'
            np.save(file, array, allow_pickle=False)
            depth = {'file': file.relative_to(self.root.parent).as_posix(), 'sha256': sha256_file(file),
                     'shape': list(array.shape), 'dtype': str(array.dtype)}
        version = state_version(state, source='actual')
        state_file = self.root/'states'/(version+'.json')
        if not state_file.exists():
            write_json(state_file, state)
        record = {'schema_version': 1, 'scene_id': scene_id, 'session_id': session_id,
                  'task_id': task_id, 'frame_id': frame_id, 'timing': dict(timing),
                  'state_source': 'actual', 'state_version': version,
                  'state_content_sha256': json_digest(state, allow_nan=False),
                  'state_file': state_file.relative_to(self.root.parent).as_posix(),
                  'state_sha256': sha256_file(state_file), 'camera_pose': camera_pose,
                  'pose': observation.get('pose'), 'images': images, 'depth': depth,
                  'media_status': 'ready' if images else 'missing_rgb'}
        with (self.root/'frames.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
        return record
