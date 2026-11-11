"""Portable dispatcher: check by default; training requires --run."""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CASES={
 'c04_original':('revision_experiments_fast_20261029','plan.json'),
 'c04_real_extension':('revision_C04_real_extension_20261109','plan.json'),
 'c05_order_pair':('revision_C05_order_ablation_20261030','plan_order_ablation.json'),
 'c05_order_extension':('revision_C05_order_extension_20261030','plan_order_extension.json'),
 'c05_order_round2':('revision_C05_order_extension_round2_20261031','plan_order_extension.json'),
 'c07_algebra':('revision_C07_local_Algebra_20261031',None),
}
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('experiment',choices=CASES);p.add_argument('--run',action='store_true');p.add_argument('--merge',action='store_true')
p.add_argument('--data-root',type=Path,default=ROOT.parent/'HyperGraphMatrix');p.add_argument('--cache-root',type=Path,default=ROOT.parent/'Hyper_Distance')
p.add_argument('--output',type=Path);p.add_argument('--workers',type=int,default=1)
a=p.parse_args();folder,plan=CASES[a.experiment];folder=ROOT/'revision_experiments'/folder
plan=folder/plan if plan else ROOT/'reproducibility/plans/algebra_fair_plan.json'
output=(a.output or ROOT/'reproduced_results'/a.experiment).resolve()
cmd=[sys.executable,'-u',str(folder/'run_suite.py')]
if a.merge:
 cmd+=['merge','--plan',str(plan),'--inputs',str(output),'--output',str(output.parent/(output.name+'_merged'))]
else:
 cmd+=['run' if a.run else 'check','--plan',str(plan),'--data-root',str(a.data_root.resolve()),'--cache-root',str(a.cache_root.resolve())]
 if a.run:cmd+=['--output',str(output),'--workers',str(a.workers),'--s2v-workers','1']
import os
env=dict(os.environ);env['PYTHONPATH']=str(folder/'.ops')+os.pathsep+env.get('PYTHONPATH','')
subprocess.run(cmd,cwd=ROOT,env=env,check=True)
