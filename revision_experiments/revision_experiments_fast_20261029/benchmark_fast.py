"""Engineering timing only: 8 episodes with the original 600-episode exploration schedule."""
from pathlib import Path
import json
import argparse
import platform
import tempfile
import time
import numpy as np
from shared_experiment_core import Diffusion, cached_separation, pool_for, learn


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path,default=Path(__file__).parent/'../../HyperGraphMatrix')
    parser.add_argument('--cache-root',type=Path,default=Path(__file__).parent/'../../Hyper_Distance')
    parser.add_argument('--methods',nargs='+',default=['dqn','s2v'],choices=['dqn','s2v'])
    args=parser.parse_args()
    rows=[]
    for name in ['synthetic_HCL_v1000_e1000_exp2.5_maxdeg10','Algebra']:
        root=args.data_root
        path=root/(name+'.npy')
        if not path.exists():path=root/'HyperGraphMatrix'/(name+'.npy')
        matrix=np.load(path)
        rivals=np.lexsort((np.arange(len(matrix)),-matrix.sum(axis=1)))[:max(1,int(.01*len(matrix)))]
        xi,phi,_,_=cached_separation(matrix,rivals,10,.5,name,args.cache_root)
        pool=pool_for(xi,phi,rivals,.2,10,'product')
        streams=np.random.SeedSequence([20261027,0]).spawn(3)
        seeds=streams[0].generate_state(50,dtype=np.uint64)
        for method in args.methods:
            simulator=Diffusion(matrix,rivals,.03,30)
            cfg=dict(learner=method,learning_rate=.03,replay_capacity=30000,episodes=8,
                     epsilon_schedule_episodes=600,batch_size=64,gamma=.9,target_update_steps=100,
                     s2v_embedding_dim=32,s2v_iterations=32)
            with tempfile.TemporaryDirectory(prefix='fast_profile_') as d:
                cfg.update(checkpoint_path=str(Path(d)/'latest.pt'),training_log=str(Path(d)/'training.jsonl'))
                _,training,inference,_=learn(pool,np.column_stack((phi,xi)),10,simulator,seeds,cfg,int(streams[2].generate_state(1)[0]))
            row=dict(dataset=name,method=method,engineering_episodes=8,training_seconds=training,
                     seconds_per_episode=training/8,inference_seconds=inference,
                     cache_hits=simulator.cache_hits,cache_misses=simulator.cache_misses,host=platform.node())
            print(json.dumps(row),flush=True);rows.append(row)
            Path('benchmark_fast_results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')


if __name__=='__main__':main()
