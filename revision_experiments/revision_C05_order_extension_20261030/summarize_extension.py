"""Report all fixed-cohort outcomes, including null/negative results and incomplete groups."""
import argparse
import hashlib
import json
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import t
from run_suite import HERE, load_plan, verify_artifacts, write


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def stats(values):
    a=np.asarray(values,dtype=float);n=len(a)
    sd=float(a.std(ddof=1)) if n>1 else None
    return dict(n=n,mean=float(a.mean()),sd=sd)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,default=HERE/'results/extension')
    parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
    plan=load_plan(HERE/'plan_order_extension.json');jobs={j['task_id']:j for j in plan['tasks']}
    declarations=json.loads((HERE/'cohort_declaration.json').read_text(encoding='utf-8'))
    records=defaultdict(dict);details=[];expected=set(jobs)
    for done in args.results.glob('*/complete.json'):
        receipt=json.loads(done.read_text(encoding='utf-8'))
        if receipt['plan_id']!=plan['plan_id'] or done.parent.name not in expected:raise ValueError('Unexpected task/plan')
        verify_artifacts(done.parent,receipt)
        attempt=done.parent/receipt['attempt'];r=json.loads((attempt/'results.jsonl').read_text(encoding='utf-8').splitlines()[0])
        ds=r['dataset'];order=r['screening_orders'];repeat=r['repeat'];tag=f'product_p0.2_dqn_r{repeat}'
        samples=np.load(attempt/ds/(tag+'_samples.npy'))
        assert len(samples)==1000 and abs(samples.mean()-r['mean'])<1e-10
        with np.load(attempt/ds/'features.npz') as f:
            hashes={k:array_hash(f[k]) for k in ['xi','phi','rivals']}
        with np.load(attempt/ds/f'seeds_{repeat}.npz') as f:
            seed_hashes={k:array_hash(f[k]) for k in ['train','evaluation']}
        r.update(feature_hashes=hashes,seed_hashes=seed_hashes,sources=receipt['sources'][ds])
        key=(ds,order)
        if repeat in records[key]:raise ValueError('Duplicate independent repeat')
        records[key][repeat]=r;details.append(r)
    finished={p.parent.name for p in args.results.glob('*/complete.json')}
    missing=sorted(expected-finished)
    if missing and not args.allow_partial:raise RuntimeError(f'{len(missing)} tasks remain; use --allow-partial for interim output')
    report='# 扩展网络：一阶与多阶筛选\n\n固定新增网络：senate-committees、Restaurants-Rev、Geometry。全部结果均报告，不能只保留正结果。\n\n'
    report+='| 网络 | 状态 | 一阶均值±SD | 多阶均值±SD | 多阶−一阶 | 配对95%区间 | 新3网络Bonferroni区间 |\n|---|---|---:|---:|---:|---|---|\n'
    groups=[]
    for ds in declarations['datasets']:
        a=records[(ds,1)];b=records[(ds,10)]
        if len(a)!=5 or len(b)!=5:
            report+=f'| {ds} | 进行中：一阶{len(a)}/5，多阶{len(b)}/5 | — | — | — | — | — |\n';continue
        assert a.keys()==b.keys()
        for repeat in a:
            for k in ['feature_hashes','seed_hashes','sources']:
                if a[repeat][k]!=b[repeat][k]:raise ValueError(f'Paired design mismatch {ds}: {k}')
        first=stats([a[i]['mean'] for i in sorted(a)]);multi=stats([b[i]['mean'] for i in sorted(b)])
        delta=stats([b[i]['mean']-a[i]['mean'] for i in sorted(a)])
        se=delta['sd']/np.sqrt(5);half=float(t.ppf(.975,4)*se);adjusted=float(t.ppf(1-.05/(2*3),4)*se)
        entry=dict(dataset=ds,first_order=first,multi_order=multi,delta=delta,ci95=[delta['mean']-half,delta['mean']+half],
                   bonferroni_three_network_ci=[delta['mean']-adjusted,delta['mean']+adjusted],paired_design_verified=True)
        groups.append(entry)
        report+=(f"| {ds} | 完成 | {first['mean']:.3f}±{first['sd']:.3f} | {multi['mean']:.3f}±{multi['sd']:.3f} | {delta['mean']:.3f} | "
                 f"[{entry['ci95'][0]:.3f},{entry['ci95'][1]:.3f}] | [{entry['bonferroni_three_network_ci'][0]:.3f},{entry['bonferroni_three_network_ci'][1]:.3f}] |\n")
    original=json.loads((HERE/'previous_evidence/original_two_networks_summary.json').read_text(encoding='utf-8'))
    report+='\n## 原先两个网络（始终保留，不因新增结果而删除）\n\n'
    for g in original['groups']:
        d=g['paired_multi_minus_first'];report+=f"- {g['dataset']}：多阶减一阶={d['mean']:.4f}，95%区间[{d['mean']-d['ci95_halfwidth']:.4f},{d['mean']+d['ci95_halfwidth']:.4f}]。\n"
    report+='\n这是看到原两网络结果后开展的探索性扩展，不作为独立预注册的确认性验证。所有网络输入均含xi10，仅检验候选筛选阶段，不能归因为整个框架高阶信息的完全去除。区间基于5重复和t分布近似，校正也不消除模型/缓存缺陷。旧缓存不重算；新增3网络仅通过一阶共享超边检查，不代表全部最短路/全部阶正确。\n'
    output=HERE/'comparison';output.mkdir(exist_ok=True)
    write(output/'summary.json',dict(complete=not missing,completed_tasks=len(finished),missing_task_ids=missing,new_groups=groups,original_two_groups=original['groups'],all_finished_records=details))
    temporary=output/'扩展结果.tmp';temporary.write_text(report,encoding='utf-8');temporary.replace(output/'扩展结果.md')
    print(report)


if __name__=='__main__':main()
