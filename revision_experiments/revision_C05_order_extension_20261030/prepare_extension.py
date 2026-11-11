"""Fixed exploratory cohort, selected for size/cache availability before outcomes."""
import json
from pathlib import Path
import numpy as np
from run_suite import HERE, sha, identity, write, preflight
from shared_experiment_core import cached_separation, screening_separation, pool_for

DATASETS=['senate-committees','Restaurants-Rev','Geometry']


def main():
    original=HERE.parent/'revision_C05_order_ablation_20261030/plan_order_ablation.json'
    template=json.loads(original.read_text(encoding='utf-8'))['tasks'][0]['config']
    assert template['episodes']==600 and template['train_simulations']==50 and template['eval_simulations']==1000
    tasks=[]
    for dataset in DATASETS:
        for order in (1,10):
            for repeat in range(5):
                cfg=dict(template,datasets=[dataset],screening_orders=order,feature_orders=10,
                         repeat_ids=[repeat],repeats=5,orders=10)
                tasks.append(dict(task_id=identity(cfg)[:20],config=cfg,experiments=['C05_order_screening_extension']))
    plan=dict(version=1,tasks=sorted(tasks,key=lambda j:j['task_id']),code_sha256={f.name:sha(f) for f in sorted(HERE.glob('*.py'))},
        design='Exploratory fixed 3-network cohort; both screening orders trained from scratch; DQN features always phi,xi10',
        dataset_selection='Fixed by size and availability of all ten old caches, before training outcomes; retain original two network results')
    plan['plan_id']=identity(plan)
    path=HERE/'plan_order_extension.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=plan:raise ValueError('Existing plan differs; do not overwrite running experiments')
    else:write(path,plan)
    data=HERE.parent.parent/'HyperGraphMatrix';cache=HERE.parent.parent/'Hyper_Distance'
    preflight(data,cache,plan)
    audit=[]
    for name in DATASETS:
        path=data/(name+'.npy')
        if not path.exists():path=data/'HyperGraphMatrix'/(name+'.npy')
        matrix=np.load(path,allow_pickle=False);n=len(matrix)
        rivals=np.lexsort((np.arange(n),-matrix.sum(axis=1)))[:max(1,int(.01*n))]
        xi,phi,distance,sources=cached_separation(matrix,rivals,10,.5,name,cache)
        one=pool_for(screening_separation(xi,distance,rivals,.5,1),phi,rivals,.2,10,'product')
        many=pool_for(xi,phi,rivals,.2,10,'product')
        first=next(p for p in (cache/(name+'_distance_1.npy'),cache/'Hyper_Distance'/(name+'_distance_1.npy')) if p.exists())
        cached=np.load(first,mmap_mode='r');violations=0
        for edge in range(matrix.shape[1]):
            members=np.flatnonzero(matrix[:,edge]);violations+=int(np.count_nonzero(cached[np.ix_(members,members)]!=1))
        audit.append(dict(dataset=name,shape=list(matrix.shape),pool_size=len(one),overlap=len(set(one)&set(many)),
            ordered_pools_identical=bool(np.array_equal(one,many)),first_order_shared_edge_violations=violations,
            all_orders_shortest_paths_verified=False,matrix_sha256=sha(path),cache_sources=sources))
        write(HERE/f'tasks_{name}.json',[j['task_id'] for j in plan['tasks'] if j['config']['datasets']==[name]])
    write(HERE/'cohort_declaration.json',dict(datasets=DATASETS,total_new_tasks=30,repeats=5,screening_orders=[1,10],
        feature_orders=10,selection_basis='size and existing-cache availability; no outcome-based selection',
        original_networks_retained=['Algebra','synthetic_HCL_v1000_e1000_exp2.5_maxdeg10'],audit=audit,plan_id=plan['plan_id']))
    print('Prepared 30 tasks in fixed order: '+', '.join(DATASETS))


if __name__=='__main__':main()
