import json
from run_suite import HERE,identity,sha,write,preflight
BASE=dict(orders=10,alpha=.5,budget=10,beta=.03,horizon=30,episodes=600,train_simulations=50,eval_simulations=1000,repeats=5,learning_rate=.03,gamma=.9,batch_size=64,replay_capacity=30000,target_update_steps=100,seed=20261027,distance_source='existing_cache',modes=['product'],methods=['dqn'])
def main():
    jobs=[]
    for ds,ratios in [('Bars-Rev',[.1,.4]),('NDC-classes',[.05,.1,.2,.4,1.])]:
        for ratio in ratios:
            for repeat in range(5):
                cfg=dict(BASE,datasets=[ds],ratios=[ratio],repeat_ids=[repeat])
                jobs.append(dict(task_id=identity(cfg)[:20],config=cfg,experiments=['C04_real_network_extension']))
    plan=dict(version=1,tasks=sorted(jobs,key=lambda j:j['task_id']),code_sha256={p.name:sha(p) for p in sorted(HERE.glob('*.py'))},design='Fixed cohort Bars-Rev missing ratios .1/.4 and NDC all five ratios; previous target-priority equal-beta supplementary backend; no new cache generation')
    plan['plan_id']=identity(plan)
    preflight(HERE.parent.parent/'HyperGraphMatrix',HERE.parent.parent/'Hyper_Distance',plan)
    path=HERE/'plan.json'
    if path.exists():
        assert json.loads(path.read_text())==plan,'Existing plan differs'
    else:write(path,plan)
    print('Validated 35 new DQN tasks; 10 Bars-Rev + 25 NDC-classes')
if __name__=='__main__':main()
