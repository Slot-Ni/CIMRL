"""Verify paired evidence and summarize first-order vs ten-order candidate screening."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
from run_suite import HERE, sha, write, load_plan, verify_artifacts
from prepare_order_ablation import array_sha


def describe(a):
    a=np.array(a,dtype=float);sd=float(a.std(ddof=1)) if len(a)>1 else None
    half=float(t.ppf(.975,len(a)-1)*sd/np.sqrt(len(a))) if sd is not None else None
    return dict(n=len(a),mean=float(a.mean()),sd=sd,ci95_halfwidth=half)


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=HERE/'results/first_order')
    p.add_argument('--output',type=Path,default=HERE/'comparison');args=p.parse_args()
    plan=load_plan(HERE/'plan_order_ablation.json');expected={j['task_id'] for j in plan['tasks']}
    complete=list(args.results.glob('*/complete.json'))
    if {f.parent.name for f in complete}!=expected:raise RuntimeError('All 10 new runs must finish before final comparison')
    groups={};pairs=[]
    for path in complete:
        receipt=json.loads(path.read_text(encoding='utf-8'));verify_artifacts(path.parent,receipt)
        if receipt['plan_id']!=plan['plan_id']:raise ValueError('Unexpected plan')
        attempt=path.parent/receipt['attempt'];r=json.loads((attempt/'results.jsonl').read_text(encoding='utf-8').splitlines()[0])
        ds=r['dataset'];repeat=r['repeat'];target=HERE/'baseline_reference'/ds
        base=json.loads((target/f'repeat_{repeat}.json').read_text(encoding='utf-8'));b=base['record']
        if receipt['sources'][ds]!=base['sources']:raise ValueError('Data/cache provenance differs')
        f=np.load(attempt/ds/'features.npz')
        for name,expected_hash in base['feature_sha256'].items():
            if array_sha(f[name])!=expected_hash:raise ValueError(f'DQN input feature changed: {name}')
        tag=f'product_p0.2_dqn_r{repeat}'
        with np.load(attempt/ds/f'seeds_{repeat}.npz') as new_seeds,np.load(target/f'seeds_{repeat}.npz') as old_seeds:
            assert all(np.array_equal(new_seeds[k],old_seeds[k]) for k in ['train','evaluation'])
        samples=np.load(attempt/ds/(tag+'_samples.npy'));baseline_samples=np.load(target/(tag+'_samples.npy'))
        assert len(samples)==len(baseline_samples)==1000
        assert sha(target/(tag+'_samples.npy'))==base['samples_sha256']
        assert sha(target/f'seeds_{repeat}.npz')==base['seeds_sha256']
        assert abs(samples.mean()-r['mean'])<1e-10 and abs(baseline_samples.mean()-b['mean'])<1e-10
        assert r['screening_orders']==1 and r['feature_orders']==10
        difference=b['mean']-r['mean'];pool_a=set(r['pool']);pool_b=set(b['pool'])
        paired=dict(dataset=ds,repeat=repeat,first_order=r['mean'],multi_order=b['mean'],
                    multi_minus_first=difference,pool_overlap=len(pool_a&pool_b),
                    pool_jaccard=len(pool_a&pool_b)/len(pool_a|pool_b),
                    first_training_seconds=r['training_seconds'],baseline_training_seconds=b['training_seconds'])
        pairs.append(paired);groups.setdefault(ds,[]).append(paired)
    rows=[]
    report='# 一阶与多阶候选筛选对照结果\n\n仅改变筛选得分；两组DQN输入均为[phi,xi10]，原600/50/1000/5预算不变。\n\n'
    for ds,records in sorted(groups.items()):
        a=describe([r['first_order'] for r in records]);b=describe([r['multi_order'] for r in records]);d=describe([r['multi_minus_first'] for r in records])
        row=dict(dataset=ds,first_order=a,multi_order=b,paired_multi_minus_first=d,
                 mean_pool_jaccard=float(np.mean([r['pool_jaccard'] for r in records])))
        rows.append(row)
        report+=f"## {ds}\n\n一阶：{a['mean']:.4f} ± {a['sd']:.4f}（跨运行SD）；多阶：{b['mean']:.4f} ± {b['sd']:.4f}。\n\n多阶减一阶的配对均值差：{d['mean']:.4f}，描述性95% t区间[{d['mean']-d['ci95_halfwidth']:.4f}, {d['mean']+d['ci95_halfwidth']:.4f}]。\n\n"
    report+='区间仅基于5次独立重复，尚未校正两网络的多重比较。旧缓存缺陷仍需披露，尤其HCL不能据此证明理想高阶距离的普遍贡献。复用组与新组后端仅增加筛选切换，速度差也不构成从零距离预处理成本比较。\n'
    args.output.mkdir(parents=True,exist_ok=True)
    write(args.output/'summary.json',dict(complete=True,groups=rows,paired_records=pairs,feature_identity_verified=True,seed_banks_matched=True))
    (args.output/'结果分析.md').write_text(report,encoding='utf-8');print(report)


if __name__=='__main__':main()
