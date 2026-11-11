"""Prepare ten new first-order screening runs, reusing verified ten-order baselines."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from run_suite import HERE, sha, identity, load_plan, verify_artifacts, write


def array_sha(array):
    a=np.ascontiguousarray(array)
    return identity(dict(dtype=str(a.dtype),shape=list(a.shape),bytes_sha=__import__('hashlib').sha256(a.tobytes()).hexdigest()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline-code',type=Path,default=HERE.parent/'revision_experiments_fast_20261029')
    parser.add_argument('--baseline-output',type=Path,default=HERE.parent/'revision_experiments_fast_20261029/results/local_fast')
    args=parser.parse_args()
    old=json.loads((args.baseline_code/'plan.json').read_text(encoding='utf-8'))
    if identity({k:v for k,v in old.items() if k!='plan_id'})!=old['plan_id']:raise ValueError('Baseline plan checksum mismatch')
    jobs={j['task_id']:j for j in old['tasks']};tasks=[];baseline=[]
    folder=HERE/'baseline_reference';folder.mkdir(exist_ok=True)
    for receipt_path in sorted(args.baseline_output.glob('*/complete.json')):
        receipt=json.loads(receipt_path.read_text(encoding='utf-8'));task_id=receipt_path.parent.name
        job=jobs.get(task_id)
        if not job:continue
        cfg=job['config']
        if not (cfg['methods']==['dqn'] and cfg['modes']==['product'] and cfg['ratios']==[.2]):continue
        if cfg['datasets'][0] not in ('Algebra','synthetic_HCL_v1000_e1000_exp2.5_maxdeg10'):continue
        verify_artifacts(receipt_path.parent,receipt)
        attempt=receipt_path.parent/receipt['attempt']
        record=json.loads((attempt/'results.jsonl').read_text(encoding='utf-8').splitlines()[0])
        dataset=record['dataset'];repeat=record['repeat']
        tag=f"product_p0.2_dqn_r{repeat}"
        sources=receipt['sources'][dataset]
        target=folder/dataset;target.mkdir(exist_ok=True)
        for source in [attempt/dataset/(tag+'_samples.npy'),attempt/dataset/f'seeds_{repeat}.npz']:
            shutil.copy2(source,target/source.name)
        features=np.load(attempt/dataset/'features.npz')
        ref=dict(record=record,config=cfg,execution_plan_id=receipt.get('execution_plan_id',receipt['plan_id']),
                 source_attempt=str(attempt.resolve()),sources=sources,feature_sha256=dict(
                     xi=array_sha(features['xi']),phi=array_sha(features['phi']),rivals=array_sha(features['rivals'])),
                 samples_sha256=sha(target/(tag+'_samples.npy')),seeds_sha256=sha(target/f'seeds_{repeat}.npz'))
        write(target/f'repeat_{repeat}.json',ref);baseline.append(dict(dataset=dataset,repeat=repeat))
        new=dict(cfg,screening_orders=1,feature_orders=10)
        assert new['orders']==10 and new['episodes']==600 and new['train_simulations']==50 and new['eval_simulations']==1000
        tasks.append(dict(task_id=identity(new)[:20],config=new,experiments=['C05_higher_order_screening_ablation']))
    assert len(tasks)==10 and len({(r['dataset'],r['repeat']) for r in baseline})==10,'Need all ten verified baselines'
    plan=dict(version=1,tasks=sorted(tasks,key=lambda j:j['task_id']),
              code_sha256={f.name:sha(f) for f in sorted(HERE.glob('*.py'))},
              design='Screening orders=1 vs reused10; DQN inputs fixed phi,xi10; only first-order tasks are newly trained')
    plan['plan_id']=identity(plan)
    path=HERE/'plan_order_ablation.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=plan:raise ValueError('Plan changed: use a new folder rather than replacing an active plan')
    else:write(path,plan)
    write(folder/'manifest.json',dict(baselines=baseline,original_plan_id=old['plan_id'],new_plan_id=plan['plan_id']))
    print('Prepared 10 first-order tasks; verified/reused 10 multi-order baselines.')


if __name__=='__main__':main()
