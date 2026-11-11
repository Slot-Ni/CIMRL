"""Adopt verified old results and checkpoint snapshots without rewriting original evidence."""
import argparse
import json
import shutil
from pathlib import Path
from run_suite import HERE, load_plan, preflight, identity, sha, verify_artifacts, write


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--previous-code',type=Path,required=True)
    parser.add_argument('--previous-output',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',type=Path,default=HERE/'plan.json')
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--cache-root',type=Path,required=True)
    args=parser.parse_args()
    plan=load_plan(args.plan)
    old_plan=json.loads((args.previous_code/'plan.json').read_text(encoding='utf-8'))
    if identity({k:v for k,v in old_plan.items() if k!='plan_id'})!=old_plan['plan_id']:
        raise ValueError('Old plan checksum mismatch')
    for name,expected in old_plan['code_sha256'].items():
        if sha(args.previous_code/name)!=expected:raise ValueError('Old source has changed')
    sources=preflight(args.data_root.resolve(),args.cache_root.resolve(),plan)
    new_jobs={j['task_id']:j for j in plan['tasks']}
    old_jobs={j['task_id']:j for j in old_plan['tasks']}
    args.output.mkdir(parents=True,exist_ok=True)
    adopted=[];resumed=[]
    for previous in args.previous_output.iterdir():
        if not previous.is_dir() or previous.name not in new_jobs:continue
        task_id=previous.name
        if old_jobs.get(task_id)!=new_jobs[task_id]:raise ValueError('Task definition differs')
        target=args.output/task_id
        if (target/'complete.json').exists():continue
        done=previous/'complete.json'
        if done.exists():
            record=json.loads(done.read_text(encoding='utf-8'))
            if record['plan_id']!=old_plan['plan_id'] or record['sources']!=sources:raise ValueError('Old provenance mismatch')
            verify_artifacts(previous,record)
            shutil.copytree(previous,target,dirs_exist_ok=True)
            # plan_id is the current scheduling plan. Actual execution provenance
            # remains explicit, and the original manifest/artifacts are untouched.
            record.update(execution_plan_id=old_plan['plan_id'],plan_id=plan['plan_id'],
                          adopted_result=True,adopted_from=str(previous.resolve()))
            write(target/'complete.json',record)
            adopted.append(task_id)
        else:
            candidates=sorted(previous.glob('attempt_*/*/*_training_latest.pt'),key=lambda f:f.stat().st_mtime)
            if not candidates:continue
            saved=candidates[-1]
            import torch
            checkpoint=torch.load(saved,map_location='cpu',weights_only=False)
            cfg=new_jobs[task_id]['config']
            if checkpoint['episodes']!=cfg['episodes']:raise ValueError('Episode budget mismatch')
            destination=target/'attempt_imported'/saved.parent.name/saved.name
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(saved,destination)
            write(target/'attempt_imported/resume_origin.json',dict(backend='reference',execution_plan_id=old_plan['plan_id'],
                source=str(saved.resolve()),sha256=sha(destination),next_episode=checkpoint['next_episode']))
            write(target/'run_identity.json',dict(plan_id=plan['plan_id'],sources=sources))
            resumed.append(dict(task_id=task_id,next_episode=checkpoint['next_episode']))
    write(args.output/'adoption_receipt.json',dict(plan_id=plan['plan_id'],adopted=adopted,resumed=resumed))
    completed={p.parent.name for p in args.output.glob('*/complete.json')}
    active={r['task_id'] for r in resumed}
    local=[];server=[]
    for job in plan['tasks']:
        task_id=job['task_id']
        if task_id in completed:continue
        if job['config']['methods'][0] in ('greedy','s2v') and task_id not in active:server.append(task_id)
        else:local.append(task_id)
    write(HERE/'tasks_local.json',local);write(HERE/'tasks_server.json',server)
    write(HERE/'assignment.json',dict(plan_id=plan['plan_id'],completed=sorted(completed),local=local,server=server,
        policy='Local: DQN/ranking and existing S2V checkpoint; server: remaining greedy/S2V. No overlap.'))
    print(json.dumps(dict(adopted=len(adopted),resume_checkpoints=resumed,local_remaining=len(local),server_remaining=len(server)),indent=2))


if __name__=='__main__':main()
