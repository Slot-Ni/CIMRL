"""C07: inspect/import existing Bars-Rev evidence; run only missing cells on request."""
import argparse
from contextlib import contextmanager
from collections import defaultdict
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

from run_suite import HERE, identity, sha, write, load_plan, preflight

ALGORITHM_FILES=('shared_experiment_core.py','s2v_adapter.py','s2v_network.py')
BASE=dict(orders=10,alpha=.5,budget=10,beta=.03,horizon=30,episodes=600,
          train_simulations=50,eval_simulations=1000,repeats=5,learning_rate=.03,
          gamma=.9,batch_size=64,replay_capacity=30000,target_update_steps=100,
          seed=20261027,distance_source='existing_cache',datasets=['Bars-Rev'],ratios=[.2],modes=['product'])


def tasks():
    output=[]
    for method in ('dqn','greedy','s2v'):
        for repeat in range(5):
            config=dict(BASE,methods=[method],repeat_ids=[repeat])
            if method=='s2v':config.update(s2v_embedding_dim=32,s2v_iterations=32)
            output.append(dict(task_id=identity(config)[:20],config=config,experiments=['C07_same_pool_fair_budget']))
    return sorted(output,key=lambda j:j['task_id'])


def safe_child(root,relative):
    relative=Path(relative)
    if relative.is_absolute() or '..' in relative.parts or ':' in str(relative):
        raise ValueError('Unsafe artifact/archive path')
    target=(root/relative).resolve()
    if os.path.commonpath([str(root.resolve()),str(target)])!=str(root.resolve()):
        raise ValueError('Path escapes evidence folder')
    return target


@contextmanager
def source_folder(source):
    source=source.resolve()
    if source.is_file() and source.suffix.lower()=='.zip':
        with tempfile.TemporaryDirectory(prefix='c07_import_') as temp:
            root=Path(temp)
            with ZipFile(source) as archive:
                for entry in archive.infolist():
                    safe_child(root,entry.filename)
                    if stat.S_ISLNK(entry.external_attr>>16):raise ValueError('Archive symlinks are not accepted')
                archive.extractall(root)
            yield locate_source(root),'zip_snapshot'
    elif source.is_dir():yield locate_source(source),'directory'
    else:raise FileNotFoundError(source)


def locate_source(root):
    expected={j['task_id'] for j in tasks()}
    if any((root/task).is_dir() for task in expected):return root
    candidates={p.parent for p in root.rglob('*') if p.is_dir() and p.name in expected}
    if len(candidates)!=1:raise ValueError('Cannot uniquely locate Bars-Rev task folder')
    return next(iter(candidates))


def verify_artifacts(directory,receipt):
    for relative,digest in receipt['artifacts'].items():
        if sha(safe_child(directory,relative))!=digest:raise ValueError('Artifact hash mismatch: '+relative)


def verify_algorithm(attempt):
    manifest=json.loads((attempt/'manifest.json').read_text(encoding='utf-8'))
    for name in ALGORITHM_FILES:
        if manifest['code_sha256'].get(name)!=sha(HERE/name):
            raise ValueError('Algorithm differs from accepted reference: '+name)
    if manifest.get('semantics')!='equal-beta,target-priority,synchronous':raise ValueError('Propagation mechanism mismatch')
    return manifest


def inspect_source(root):
    expected={j['task_id']:j for j in tasks()};status=[]
    for task_id,job in expected.items():
        directory=root/task_id;done=directory/'complete.json'
        item=dict(task_id=task_id,method=job['config']['methods'][0],repeat=job['config']['repeat_ids'][0],status='not_complete',checkpoint_available=False)
        if done.exists():
            receipt=json.loads(done.read_text(encoding='utf-8'))
            if receipt['task']['config']!=job['config']:raise ValueError('Budget/task mismatch')
            verify_artifacts(directory,receipt);verify_algorithm(directory/receipt['attempt'])
            item['status']='complete'
        else:
            checkpoints=sorted(directory.glob('attempt_*/*/*_training_latest.pt'),key=lambda f:f.stat().st_mtime)
            if checkpoints:item['checkpoint_available']=True
            logs=sorted(directory.glob('attempt_*/*/*_training.jsonl'),key=lambda f:f.stat().st_mtime)
            if logs:
                lines=logs[-1].read_text(encoding='utf-8').splitlines()
                if lines:
                    try:item['last_logged_episode']=json.loads(lines[-1])['episode']
                    except (ValueError,KeyError):item['last_log_incomplete']=True
            item['lock_present']=(directory/'RUNNING.lock').exists()
        status.append(item)
    return status


def active_scheduler(root):
    # Verify the PID command line against the exact source output folder;
    # a lock file alone may be a stale copied snapshot.
    matches=[]
    if os.name!='posix' or not Path('/proc').is_dir():return matches
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit():continue
        try:
            if folder.stat().st_uid!=os.getuid():continue
            args=[x.decode(errors='replace') for x in (folder/'cmdline').read_bytes().split(b'\0') if x]
            if not any(Path(a).name=='run_suite.py' for a in args) or 'run' not in args:continue
            if '--output' not in args:continue
            output=Path(args[args.index('--output')+1])
            if not output.is_absolute():output=(folder/'cwd').resolve()/output
            if output.resolve()==root.resolve():matches.append(int(folder.name))
        except (OSError,ValueError,IndexError):pass
    return matches


def print_status(status):
    for method in ('dqn','greedy','s2v'):
        cells=[s for s in status if s['method']==method]
        print(method,':',sum(c['status']=='complete' for c in cells),'/5 complete',flush=True)
    for item in status:
        if item['status']!='complete':print('PENDING',json.dumps(item),flush=True)


def prepare(args):
    workspace=args.workspace.resolve();workspace.mkdir(parents=True,exist_ok=True)
    if list(workspace.glob('*/RUNNING.lock')):raise RuntimeError('New workspace has task locks; do not modify it while running')
    plan=dict(version=1,tasks=tasks(),code_sha256={f.name:sha(f) for f in sorted(HERE.glob('*.py'))},
              design='C07 corrected S2V adaptation and HCDP-DQN/Greedy share product pool p=.2; equal training/reward/evaluation budget, no tuning search')
    plan['plan_id']=identity(plan);plan_path=workspace/'fair_plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding='utf-8'))!=plan:raise ValueError('Existing workspace plan differs; choose a new workspace')
    else:write(plan_path,plan)
    sources=preflight(args.data_root.resolve(),args.cache_root.resolve(),plan)
    adopted=0;resumed=[]
    with source_folder(args.source) as (root,kind):
        if root==workspace or root in workspace.parents or workspace in root.parents:raise ValueError('Source/workspace must be separate nonnested folders')
        status=inspect_source(root)
        active=active_scheduler(root) if kind=='directory' else []
        for job in tasks():
            task_id=job['task_id'];old=root/task_id;new=workspace/task_id
            if (new/'complete.json').exists():continue
            done=old/'complete.json'
            if done.exists():
                receipt=json.loads(done.read_text(encoding='utf-8'))
                if receipt['sources']!=sources:raise ValueError('Source data/cache hashes differ')
                # inspect_source already checked the full artifacts, code and settings.
                shutil.copytree(old,new,dirs_exist_ok=True,ignore=shutil.ignore_patterns('RUNNING.lock'))
                original_task=receipt['task']
                receipt.update(origin_task=original_task,task=job,execution_plan_id=receipt['plan_id'],
                               plan_id=plan['plan_id'],adopted_result=True)
                write(new/'complete.json',receipt);adopted+=1
            elif args.import_checkpoints:
                if active:raise RuntimeError('Original scheduler is active; importing live checkpoints for a second run is blocked')
                files=sorted(old.glob('attempt_*/*/*_training_latest.pt'),key=lambda f:f.stat().st_mtime)
                if not files:continue
                path=files[-1];attempt=path.parent.parent;verify_algorithm(attempt)
                manifest=json.loads((attempt/'manifest.json').read_text(encoding='utf-8'))
                evidence=manifest['datasets']['Bars-Rev']
                partial_sources={str(evidence['matrix']).replace('\\','/').rsplit('/',1)[-1]:evidence['sha256']}
                partial_sources.update({str(item['path']).replace('\\','/').rsplit('/',1)[-1]:item['sha256'] for item in evidence['cache_sources']})
                if partial_sources!=sources['Bars-Rev']:raise ValueError('Checkpoint data/cache provenance differs')
                for k,v in job['config'].items():
                    if manifest['config'].get(k)!=v:raise ValueError('Checkpoint configuration differs: '+k)
                import torch
                # Only use checkpoints generated by this user's verified experiment code.
                saved=torch.load(path,map_location='cpu',weights_only=False)
                if not 0<=saved['next_episode']<=600 or saved['episodes']!=600:raise ValueError('Invalid episode checkpoint')
                destination=new/'attempt_imported'/path.parent.name/path.name
                destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
                write(new/'attempt_imported/resume_origin.json',dict(backend='fast',source=str(path),sha256=sha(destination),episode=saved['next_episode']))
                write(new/'run_identity.json',dict(plan_id=plan['plan_id'],sources=sources))
                resumed.append(dict(task_id=task_id,next_episode=saved['next_episode']))
    missing=[j['task_id'] for j in tasks() if not (workspace/j['task_id']/'complete.json').exists()]
    write(workspace/'pending_tasks.json',missing)
    write(workspace/'import_receipt.json',dict(source=str(args.source.resolve()),source_kind=kind,adopted_now=adopted,
        resumed=resumed,pending=len(missing),plan_id=plan['plan_id'],original_scheduler_active_pids=active,
        snapshot_warning='ZIP/locks are snapshots, not proof of current remote process state'))
    print_status(status);print('Prepared workspace; no training started. Pending tasks:',len(missing),flush=True)


def run_missing(args):
    args.workspace=args.workspace.resolve()
    args.data_root=args.data_root.resolve()
    args.cache_root=args.cache_root.resolve()
    if args.live_source.suffix.lower()=='.zip':raise ValueError('Use the latest live source folder, not an old ZIP snapshot')
    with source_folder(args.live_source) as (root,kind):
        active=active_scheduler(root)
        if active:raise RuntimeError('Original scheduler is still running; do not start duplicate S2V tasks. PIDs='+str(active))
        completed=[s for s in inspect_source(root) if s['status']=='complete']
        missing_import=[s['task_id'] for s in completed if not (args.workspace/s['task_id']/'complete.json').exists()]
        if missing_import:raise RuntimeError('New completed results exist at source. Run prepare again before run.')
    plan=args.workspace/'fair_plan.json';load_plan(plan)
    command=[sys.executable,'-u',str(HERE/'run_suite.py'),'run','--plan',str(plan),'--task-file',str(args.workspace/'pending_tasks.json'),
             '--output',str(args.workspace),'--data-root',str(args.data_root),'--cache-root',str(args.cache_root),
             '--workers',str(args.workers),'--s2v-workers',str(args.s2v_workers)]
    env=dict(os.environ)
    if os.name=='nt':env['PYTHONPATH']=str(HERE/'.ops')+os.pathsep+env.get('PYTHONPATH','')
    subprocess.run(command,cwd=HERE,check=True,env=env)


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('inspect');p.add_argument('--source',type=Path,required=True)
    for name in ('prepare','run'):
        p=sub.add_parser(name)
        p.add_argument('--workspace',type=Path,default=HERE/'results/fair_BarsRev')
        p.add_argument('--data-root',type=Path,default=HERE.parent.parent/'HyperGraphMatrix')
        p.add_argument('--cache-root',type=Path,default=HERE.parent.parent/'Hyper_Distance')
        if name=='prepare':
            p.add_argument('--source',type=Path,required=True);p.add_argument('--import-checkpoints',action='store_true')
        else:
            p.add_argument('--live-source',type=Path,required=True);p.add_argument('--workers',type=int,default=1);p.add_argument('--s2v-workers',type=int,default=1)
    p=sub.add_parser('analyze');p.add_argument('--workspace',type=Path,default=HERE/'results/fair_BarsRev')
    args=parser.parse_args()
    if args.command=='inspect':
        with source_folder(args.source) as (root,kind):
            print('Source kind:',kind,'(snapshot locks do not prove a process is running)');print_status(inspect_source(root))
    elif args.command=='prepare':prepare(args)
    elif args.command=='run':run_missing(args)
    else:
        from fair_report import analyze
        analyze(args.workspace.resolve())


if __name__=='__main__':main()
