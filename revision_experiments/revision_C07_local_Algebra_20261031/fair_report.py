"""Report verified complete runs only; missing S2V is never treated as zero spread."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
from fair_comparison import verify_artifacts,tasks
from run_suite import write,load_plan


def describe(values):
    a=np.asarray(values,dtype=float);n=len(a)
    if n==0:return dict(n=0,mean=None,between_run_sd=None,ci95=None)
    sd=float(a.std(ddof=1)) if n>1 else None
    half=float(t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else None
    return dict(n=n,mean=float(a.mean()),between_run_sd=sd,ci95=[float(a.mean()-half),float(a.mean()+half)] if half is not None else None)


def analyze(workspace):
    plan=load_plan(workspace/'fair_plan.json');expected={j['task_id']:j for j in tasks()}
    groups=defaultdict(dict);pools={};sources=None;timing=defaultdict(list);records=[]
    for done in workspace.glob('*/complete.json'):
        receipt=json.loads(done.read_text(encoding='utf-8'))
        if receipt['plan_id']!=plan['plan_id'] or done.parent.name not in expected:raise ValueError('Unexpected task plan')
        if receipt['task']['config']!=expected[done.parent.name]['config']:raise ValueError('Run settings differ')
        verify_artifacts(done.parent,receipt);attempt=done.parent/receipt['attempt']
        manifest=json.loads((attempt/'manifest.json').read_text(encoding='utf-8'))
        row=json.loads((attempt/'results.jsonl').read_text(encoding='utf-8').splitlines()[0]);method=row['method'];repeat=row['repeat'];ds=row['dataset']
        samples=np.load(attempt/ds/f"product_p0.2_{method}_r{repeat}_samples.npy",allow_pickle=False)
        if len(samples)!=1000 or abs(samples.mean()-row['mean'])>1e-10:raise ValueError('Invalid sample bank/mean')
        if len(set(row['selected']))!=10 or not set(row['selected'])<=set(row['pool']):raise ValueError('Invalid seed set')
        with np.load(attempt/ds/'features.npz') as f:
            if set(row['selected'])&set(f['rivals'].tolist()):raise ValueError('Rival selected')
        with np.load(attempt/ds/f'seeds_{repeat}.npz') as f:
            bank=(f['train'].tobytes(),f['evaluation'].tobytes())
        if repeat in pools:
            if pools[repeat]!=(row['pool'],bank):raise ValueError('Candidate pool or RNG banks differ across methods')
        else:pools[repeat]=(row['pool'],bank)
        if sources is not None and sources!=receipt['sources']:raise ValueError('Data/cache sources differ')
        sources=receipt['sources']
        if repeat in groups[method]:raise ValueError('Duplicate repeat')
        groups[method][repeat]=row
        backend=row.get('fast_backend','reference');host=manifest.get('host','unknown')
        timing[(method,host,backend)].append(row)
        row=dict(row,host=host,execution_backend=backend,execution_plan_id=receipt.get('execution_plan_id',receipt['plan_id']))
        records.append(row)
    stats={m:describe([r['mean'] for r in groups[m].values()]) for m in ('dqn','greedy','s2v')}
    comparisons=[]
    for right in ('greedy','s2v'):
        common=sorted(groups['dqn'].keys()&groups[right].keys())
        values=[groups['dqn'][i]['mean']-groups[right][i]['mean'] for i in common]
        item=dict(comparison='dqn_minus_'+right,paired_repeats=common,**describe(values),formal_five_run_comparison=(len(common)==5))
        if len(values)>1:
            a=np.array(values);half=float(t.ppf(1-.05/(2*2),len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)))
            item['bonferroni_two_comparison_ci']=[float(a.mean()-half),float(a.mean()+half)]
        else:item['bonferroni_two_comparison_ci']=None
        comparisons.append(item)
    times=[]
    for (method,host,backend),runs in sorted(timing.items()):
        item=dict(method=method,host=host,backend=backend,n=len(runs))
        for field in ('preprocessing_seconds','screening_seconds','training_seconds','inference_seconds','selection_seconds','evaluation_seconds'):
            item[field]=float(np.mean([r[field] for r in runs]))
        item['seed_selection_including_training_seconds']=item['preprocessing_seconds']+item['screening_seconds']+item['training_seconds']+item['inference_seconds']+item['selection_seconds']
        times.append(item)
    complete=all(stats[m]['n']==5 for m in stats)
    report='# C07同候选池公平预算对照\n\n'
    report+='Algebra，p=.2，种子预算10，同beta=.03、目标优先、T=30。S2V为修正后的本地Structure2Vec适配版，完整图嵌入、同HCDP池选动作，不使用xi/phi输入；不是只删除特征的同架构消融，也不是原论文全空间S2V。\n\n'
    report+='| 方法 | 完成重复 | 平均影响力 | 跨运行SD | 95% t区间 |\n|---|---:|---:|---:|---|\n'
    for method in ('dqn','greedy','s2v'):
        s=stats[method]
        if s['n']:
            report+=f"| {method} | {s['n']}/5 | {s['mean']:.4f} | {s['between_run_sd'] if s['between_run_sd'] is not None else '不足2次'} | {s['ci95']} |\n"
        else:report+=f'| {method} | 0/5 | 尚无完整结果 | — | — |\n'
    report+='\n配对差值方向为DQN减对照，依据相同独立重复编号；不能把单次结果冒充五重复比较。\n\n'
    for c in comparisons:report+=f"- {c['comparison']}：n={c['n']}，均值差={c['mean']}，95%区间={c['ci95']}，两比较Bonferroni区间={c['bonferroni_two_comparison_ci']}。\n"
    report+='\n## 计时（按主机与后端分层）\n\n'
    for r in times:
        report+=f"- {r['method']}，{r['host']}，{r['backend']}：n={r['n']}，训练{r['training_seconds']/60:.2f}分钟、选种{r['selection_seconds']/60:.2f}分钟、推理{r['inference_seconds']:.6f}秒、最终评估{r['evaluation_seconds']:.2f}秒。\n"
    report+='\n同回合/奖励仿真预算不等于FLOPs、墙钟或调参投入相同。Greedy无训练回合；其公平性是同池、同传播后端和选择仿真预算。没有执行新增超参数搜索，不能声称各模型已调至最优。跨主机或混合接续后端不计算同硬件加速比。缓存生成成本不在本轮预处理中。\n'
    report+=('\n状态：15项均完整，可以形成五重复对照。\n' if complete else '\n状态：仍不完整；仅展示已完成证据，不输出完整S2V结论。\n')
    output=workspace/'analysis';output.mkdir(exist_ok=True)
    write(output/'summary.json',dict(complete=complete,groups=stats,comparisons=comparisons,timing_by_host_backend=times,records=records))
    (output/'C07分析.md').write_text(report,encoding='utf-8');print(report)
