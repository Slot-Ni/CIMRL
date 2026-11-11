"""Temporary tiny cached fixture: both shards, resume, merge and duplicate rejection."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
from run_suite import HERE, identity, sha, write


def main():
    with tempfile.TemporaryDirectory(prefix='ins_smoke_') as temp:
        root = Path(temp)
        data, cache = root/'data', root/'cache'
        data.mkdir(); cache.mkdir()
        np.save(data/'fixture.npy', np.ones((4,1)))
        np.save(cache/'fixture_distance_1.npy', np.ones((4,4)))
        tasks = []
        for repeat in range(2):
            for method in ('dqn','greedy','s2v'):
                cfg = dict(datasets=['fixture'], orders=1, alpha=.5, budget=2,
                    ratios=[1.0], modes=['product'], methods=[method], beta=.2, horizon=2,
                    episodes=2, train_simulations=2, eval_simulations=3, repeats=2,
                    repeat_ids=[repeat], learning_rate=.001, gamma=.9, batch_size=1,
                    replay_capacity=20, target_update_steps=1, seed=42,
                    distance_source='existing_cache',s2v_embedding_dim=4,s2v_iterations=2)
                tasks.append(dict(task_id=identity(cfg)[:20],config=cfg,experiments=['fixture']))
        plan = dict(version=1,tasks=sorted(tasks,key=lambda j:j['task_id']),
                    code_sha256={p.name:sha(p) for p in sorted(HERE.glob('*.py'))})
        plan['plan_id'] = identity(plan)
        write(root/'plan.json',plan)
        common = [sys.executable,str(HERE/'run_suite.py')]
        for shard in range(2):
            command = common+['run','--plan',str(root/'plan.json'),'--data-root',str(data),
                '--cache-root',str(cache),'--output',str(root/f'worker{shard}'),
                '--workers','2','--shards','2','--shard-index',str(shard)]
            subprocess.run(command,check=True)
            # Repeating must skip all completed tasks, with no new attempts.
            before = len(list(root.rglob('attempt_*')))
            subprocess.run(command,check=True)
            assert before == len(list(root.rglob('attempt_*')))
        subprocess.run(common+['merge','--plan',str(root/'plan.json'),'--inputs',
            str(root/'worker0'),str(root/'worker1'),'--output',str(root/'merged')],check=True)
        summary = json.loads((root/'merged/merge_manifest.json').read_text())
        assert summary['complete'] and summary['completed']==6
        assert len((root/'merged/results.jsonl').read_text().splitlines())==6
        duplicate = subprocess.run(common+['merge','--plan',str(root/'plan.json'),'--inputs',
            str(root/'worker0'),str(root/'worker0'),'--allow-partial','--output',str(root/'bad')],capture_output=True,text=True)
        assert duplicate.returncode != 0 and 'Duplicate task' in duplicate.stderr
        print('PASS: 6 tiny tasks, both shards, completed-task resume, audited merge, duplicate rejection; NOT paper results.')


if __name__ == '__main__': main()
