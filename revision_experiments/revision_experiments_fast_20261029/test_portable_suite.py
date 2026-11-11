import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from shared_experiment_core import Diffusion, learn, cached_separation
from s2v_adapter import make_network
from run_suite import make_plan, load_plan


class PortableTests(unittest.TestCase):
    def test_plan_deduplicates_and_shards(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'plan.json'
            plan = make_plan(path)
            self.assertEqual(len(plan['tasks']), 140)
            self.assertEqual(plan, load_plan(path))
            cells = [j for j in plan['tasks'] if len(j['experiments']) == 3]
            self.assertEqual(len(cells), 10)
            shards = [{j['task_id'] for i,j in enumerate(plan['tasks']) if i%2==k} for k in range(2)]
            self.assertFalse(shards[0] & shards[1])
            self.assertEqual(len(shards[0] | shards[1]),140)

    def test_cache_missing_stops(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                cached_separation(np.ones((3,1)), [0], 1, .5, 'fixture', d)
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_embedding_is_trainable_and_terminal_safe(self):
        matrix = np.ones((3,1))
        cfg = dict(learner='s2v',s2v_embedding_dim=4,s2v_iterations=2,
                   learning_rate=.001,replay_capacity=20,episodes=2,batch_size=1,
                   gamma=.9,target_update_steps=1)
        net = make_network(matrix, [0], np.array([1,2]), cfg)
        before = net.embedding.theta1.weight.detach().clone()
        optimizer = torch.optim.Adam(net.parameters(), lr=.01)
        optimizer.zero_grad()
        net(torch.tensor([0.,1.,0.])).square().sum().backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, net.embedding.theta1.weight))
        chosen, _, _, checkpoint = learn(np.array([1,2]), np.ones((3,2)), 2,
             Diffusion(matrix,[0],.2,2), [17,18], cfg, 42)
        self.assertEqual(set(chosen),{1,2})
        self.assertTrue(all(torch.isfinite(v).all() for v in checkpoint.values()))

    def test_episode_checkpoint_resume_matches_uninterrupted(self):
        from unittest.mock import patch
        import shared_experiment_core as core
        matrix = np.ones((4,1))
        pool = np.array([1,2,3])
        cfg = dict(learning_rate=.001,replay_capacity=20,episodes=3,batch_size=1,
                   gamma=.9,target_update_steps=2)
        sim = Diffusion(matrix,[0],.2,2)
        expected, _, _, weights = learn(pool,np.ones((4,2)),2,sim,[17,18],cfg,42)
        with tempfile.TemporaryDirectory() as d:
            partial = dict(cfg,checkpoint_path=str(Path(d)/'latest.pt'))
            original = core.save_checkpoint
            def stop(path,state):
                original(path,state)
                raise RuntimeError('simulated interruption')
            with patch.object(core,'save_checkpoint',side_effect=stop):
                with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                    learn(pool,np.ones((4,2)),2,sim,[17,18],partial,42)
            actual, _, _, recovered = learn(pool,np.ones((4,2)),2,sim,[17,18],partial,42)
            self.assertEqual(actual,expected)
            for key in weights:
                self.assertTrue(torch.equal(weights[key],recovered[key]),key)
            saved = torch.load(partial['checkpoint_path'],weights_only=False)
            self.assertEqual(saved['next_episode'],3)


if __name__ == '__main__': unittest.main()
