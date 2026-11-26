"""CPU-only preflight and command construction. No policy or simulator imports."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from urllib.parse import urlparse

from ..artifacts.layout import sha256_file

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = {'dataset', 'simplenav_root', 'checkpoint', 'agent_python', 'isaac_python', 'output'}
OPTIONAL = {'url','unnorm_key','max_steps','agent_timeout','history_frames','collision_reset_threshold',
            'video_fps','video_event_seconds','video_layout','ffmpeg','viewer','record_video','gpu',
            'sequences','all_sequences','model_service','environment','auto_analyze',
            'robot_profile','robot','robot_assets'}


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    if not isinstance(config, dict) or REQUIRED-config.keys() or config.keys()-(REQUIRED|OPTIONAL):
        raise ValueError('missing or unsupported launch configuration fields')
    for key in REQUIRED:
        value = Path(config[key]).expanduser()
        config[key] = str((path.parent/value).resolve() if not value.is_absolute() else value.resolve())
    from ..scenes.robot_body import DEFAULT_ROBOT, validate_robot_selection
    validate_robot_selection(config.get('robot'), config.get('robot_profile'), config.get('robot_assets'))
    for key in ('robot_profile','robot_assets'):
        if key in config:
            if not isinstance(config[key],str) or not config[key].strip():
                raise ValueError(key+' must be a nonempty path')
            value = Path(config[key]).expanduser()
            config[key] = str((path.parent/value).resolve())
    if 'robot_profile' not in config:
        config.setdefault('robot', DEFAULT_ROBOT)
    for key, default in [('max_steps',2000),('history_frames',8),('collision_reset_threshold',5),('video_fps',10)]:
        config.setdefault(key,default)
        if type(config[key]) is not int or config[key] < (0 if key=='collision_reset_threshold' else 1):
            raise ValueError('invalid '+key)
    config.setdefault('agent_timeout',120)
    if not isinstance(config['agent_timeout'],(float,int)) or not 0 < config['agent_timeout'] < float('inf'):
        raise ValueError('agent_timeout must be finite and positive')
    for key in ('viewer','record_video'):
        config.setdefault(key,True)
        if type(config[key]) is not bool: raise ValueError(key+' must be boolean')
    config.setdefault('url','ws://127.0.0.1:10093')
    url = urlparse(config['url'])
    if (url.scheme != 'ws' or not url.hostname or url.path not in ('','/') or url.query or url.fragment
            or url.username or url.password or (url.port is not None and not 0 < url.port < 65536)):
        raise ValueError('current SimpleNav client supports ws://host:port without credentials/path')
    config.setdefault('video_layout','front_overview')
    config.setdefault('video_event_seconds',.5)
    config.setdefault('ffmpeg','ffmpeg')
    if 'gpu' in config and (type(config['gpu']) is not int or config['gpu'] < 0):
        raise ValueError('gpu must be a non-negative Isaac/Vulkan device index')
    if config.get('sequences') and config.get('all_sequences'):
        raise ValueError('select sequences or all_sequences, not both')
    if 'sequences' in config and (not isinstance(config['sequences'], list) or not config['sequences']
                                or any(not isinstance(s, str) or not s for s in config['sequences'])
                                or len(set(config['sequences'])) != len(config['sequences'])):
        raise ValueError('sequences must be distinct nonempty sequence IDs')
    for key in ('all_sequences','auto_analyze'):
        config.setdefault(key, key == 'auto_analyze')
        if type(config[key]) is not bool: raise ValueError(key+' must be boolean')
    config.setdefault('environment', {})
    if not isinstance(config['environment'], dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in config['environment'].items()):
        raise ValueError('environment must map names to strings')
    service = config.setdefault('model_service', {'mode':'external'})
    if not isinstance(service,dict) or set(service)-{'mode','gpu_uuid','startup_timeout','extra_args','environment'}:
        raise ValueError('invalid model_service configuration')
    if service.get('mode') not in {'external','managed'}: raise ValueError('model_service.mode must be external or managed')
    service.setdefault('startup_timeout',600)
    if type(service['startup_timeout']) not in (int,float) or not 0 < service['startup_timeout'] < float('inf'):
        raise ValueError('invalid model startup timeout')
    if not isinstance(service.get('extra_args',[]),list) or any(not isinstance(v,str) for v in service.get('extra_args',[])):
        raise ValueError('model extra_args must be an argv list')
    if any(a.split('=')[0] in {'--port','--ckpt_path'} for a in service.get('extra_args',[])):
        raise ValueError('model port and checkpoint must come from url/checkpoint')
    if not isinstance(service.get('environment',{}),dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in service.get('environment',{}).items()):
        raise ValueError('model environment must map names to strings')
    if service['mode'] == 'managed':
        if url.hostname not in {'127.0.0.1','localhost','::1'}: raise ValueError('managed model service must be local')
        if not isinstance(service.get('gpu_uuid'),str) or not service['gpu_uuid']:
            raise ValueError('managed model service requires an explicit gpu_uuid')
    return config


def commands(config):
    agent = [config['agent_python'],'-m','roboodyssey.agents.simplenav_cli',
             '--simplenav-root',config['simplenav_root'],'--checkpoint',config['checkpoint'],
             '--url',config['url'],'--history-frames',str(config['history_frames']),
             '--collision-reset-threshold',str(config['collision_reset_threshold'])]
    if config.get('unnorm_key'): agent += ['--unnorm-key',config['unnorm_key']]
    run = [config['isaac_python'],'-m','roboodyssey.run','--dataset',config['dataset'],
           '--agent-cmd',shlex.join(agent),'--output',config['output'],'--max-steps',str(config['max_steps']),
           '--agent-timeout',str(config['agent_timeout'])]
    if config['viewer']: run += ['--viewer']
    if config.get('robot_profile'): run += ['--robot-profile',config['robot_profile']]
    if config.get('robot'): run += ['--robot',config['robot']]
    if config.get('robot_assets'): run += ['--robot-assets',config['robot_assets']]
    if 'gpu' in config: run += ['--gpu',str(config['gpu'])]
    if config['record_video']:
        run += ['--record-video','--video-layout',config['video_layout'],'--video-fps',str(config['video_fps']),
                '--video-event-seconds',str(config['video_event_seconds']),'--ffmpeg',config['ffmpeg']]
    if config.get('all_sequences'): run += ['--all-sequences']
    for sequence in config.get('sequences',[]): run += ['--sequence',sequence]
    return {'cwd':str(ROOT),'agent_argv':agent,'benchmark_argv':run,'shell_preview':shlex.join(run)}


def preflight(config):
    checks = []
    def check(name, action):
        try:
            info = action()
            checks.append({'check':name,'ok':True,'detail':info})
        except Exception as exc:
            checks.append({'check':name,'ok':False,'detail':str(exc)})
    def executable(key):
        path = Path(config[key])
        if not path.is_file() or not os.access(path,os.X_OK): raise ValueError('missing executable: '+str(path))
        return str(path)
    for key in ('agent_python','isaac_python'):
        check(key,lambda key=key:executable(key))
    from ..scenes.robot_body import resolve_robot
    check('robot_assets',lambda:resolve_robot(config.get('robot'), profile=config.get('robot_profile'),
                                             assets=config.get('robot_assets')).identity())
    def source():
        root = Path(config['simplenav_root'])
        for name in ('deployment/model_server/tools/websocket_policy_client.py','NavVLAeval/common/data/runtime_dataset.py',
                     'NavVLAeval/common/model/action_codec.py'):
            if not (root/name).is_file(): raise ValueError('missing SimpleNav source: '+name)
        if config.get('model_service', {}).get('mode') == 'managed' and not (root/'deployment/model_server/server_policy.py').is_file():
            raise ValueError('managed SimpleNav model server source is missing')
        return str(root)
    check('simplenav_source',source)
    def statistics():
        path = Path(config['checkpoint'])
        if not path.exists(): raise ValueError('checkpoint not found: '+str(path))
        base = path.parent if path.is_file() else path
        candidates = [base/'dataset_statistics.json',base.parent/'dataset_statistics.json']
        stats_path = next((p for p in candidates if p.is_file()),None)
        if stats_path is None: raise ValueError('dataset_statistics.json missing beside checkpoint')
        stats = json.loads(stats_path.read_text())
        key = config.get('unnorm_key')
        if key is None:
            if len(stats)!=1: raise ValueError('unnorm_key required; keys: '+str(list(stats)))
            key = next(iter(stats))
        if not isinstance(stats.get(key,{}).get('action'),dict): raise ValueError('action statistics missing for '+str(key))
        return {'path':str(stats_path),'unnorm_key':key}
    check('action_statistics',statistics)
    def packages():
        # Launch only a host interpreter, hide CUDA, and inspect package specs;
        # never import torch, SimpleNav, Isaac or contact the model server.
        env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=''; env.pop('PYTHONPATH',None); env.pop('PYTHONHOME',None)
        code = "import importlib.util,json; print(json.dumps({n:importlib.util.find_spec(n) is not None for n in ['numpy','PIL','websockets','msgpack','typing_extensions','torch']}))"
        result = subprocess.run([config['agent_python'],'-c',code],env=env,capture_output=True,text=True,timeout=15,check=True)
        found = json.loads(result.stdout)
        if not all(found.values()): raise ValueError('missing packages: '+str([n for n,v in found.items() if not v]))
        return found
    check('agent_package_specs_no_import',packages)
    def dataset():
        datasets = selected_datasets(config)
        return {'sequences': [inspect_dataset(data) for data in datasets]}
    def inspect_dataset(data):
        for scene in data.scene_ids():
            asset = data.scene_asset(scene)
            asset_info = data._requests.asset()
            if sha256_file(asset)!=asset_info['sha256']: raise ValueError('scene asset hash mismatch')
        return {**data.sequence_identity(),'dataset':str(data.root),'scenes':data.scene_ids(),'sessions':sum(len(data.sessions(s)) for s in data.scene_ids()),
                'tasks':sum(len(session.tasks) for s in data.scene_ids() for session in data.sessions(s))}
    check('archive_integrity_and_asset',dataset)
    if config['record_video']:
        def encoder():
            from ..runtime.recording import validate_video_options
            opts = validate_video_options({'ffmpeg':config['ffmpeg'],'layout':config['video_layout'],
                 'fps':config['video_fps'],'event_seconds':config['video_event_seconds']})
            result = subprocess.run([opts['ffmpeg'],'-hide_banner','-encoders'],capture_output=True,text=True,timeout=10,check=True)
            if 'libx264' not in result.stdout: raise ValueError('CPU libx264 encoder missing')
            return {'encoder':'libx264','timeline':'event_sampled'}
        check('cpu_encoder',encoder)
    def output():
        path = Path(config['output'])
        if path.exists() and not path.is_dir(): raise ValueError('output base is not a directory')
        existing = next(p for p in [path,*path.parents] if p.exists())
        if not os.access(existing,os.W_OK): raise ValueError('output parent not writable')
        return {'base':str(path),'free_bytes':shutil.disk_usage(existing).free}
    check('output_space',output)
    return {'schema_version':1,'ok':all(c['ok'] for c in checks),'mode':'cpu_preflight',
            'not_executed':['model_connection','model_inference','cuda_initialization','isaac_startup','display_connection'],
            'checks':checks,'commands':commands(config)}


def selected_datasets(config):
    from ..scenes.dataset_loader import BenchmarkDataset
    if config.get('all_sequences') or len(config.get('sequences', [])) > 1:
        from .sequences import selected_paths
        return [BenchmarkDataset(p) for p in selected_paths(config['dataset'], config.get('sequences'))]
    return [BenchmarkDataset(config['dataset'], sequence_id=config.get('sequences', [None])[0])]


def prepare(config_path, report_path=None):
    """Shared CPU preparation; never overwrite an existing preflight report."""
    config = load_config(config_path)
    report = preflight(config)
    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if report_path:
        with Path(report_path).open('x', encoding='utf-8') as stream:
            stream.write(text)
    print(text, flush=True)
    return config, report


def main(argv=None, *, mode='preflight'):
    """CLI for CPU preparation and managed/external experiments."""
    if mode not in {'preflight', 'experiment'}:
        raise ValueError('unknown launch mode: ' + mode)
    parser = argparse.ArgumentParser(description='CPU preflight; execution requires --execute.')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--report', type=Path, help='new preflight JSON file')
    if mode != 'preflight':
        parser.add_argument('--execute', action='store_true', help='start model/Isaac experiment; uses GPUs')
    args = parser.parse_args(argv)
    config, report = prepare(args.config, args.report)
    if not report['ok']:
        return 1
    if getattr(args, 'execute', False):
        from .experiment import execute
        return execute(config, checked=report)[0]
    return 0
