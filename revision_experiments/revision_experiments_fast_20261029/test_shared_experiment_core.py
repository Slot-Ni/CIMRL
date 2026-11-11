import unittest
import numpy as np
from shared_experiment_core import separation, pool_for, Diffusion, learn


class CoreTests(unittest.TestCase):
    def test_high_order_unreachable_and_shared_edge(self):
        matrix = np.array([[1,0],[1,1],[0,1],[0,0]])
        xi, phi, distances = separation(matrix, np.array([0]), 2, .5)
        self.assertEqual(distances[0,0,2], 2)
        self.assertTrue(np.isinf(distances[1,0,2]))
        self.assertEqual(distances[1,0,1], 1)
        self.assertTrue(np.isinf(distances[0,0,3]))
        self.assertTrue(np.isfinite(xi).all())
        self.assertEqual(phi[3], 0)

    def test_skipped_rows_regression(self):
        matrix = np.ones((202,1), dtype=int)
        _, _, distances = separation(matrix, np.array([1,3]), 3, .5)
        self.assertTrue((distances == 1).all())

    def test_exclusion_and_zero_ties(self):
        pool = pool_for(np.zeros(4), np.zeros(4), [0], 1, 3, 'product')
        self.assertEqual(pool.tolist(), [1,2,3])

    def test_target_priority_and_next_step(self):
        sim = Diffusion(np.ones((3,1)), [0], 1, 1)
        self.assertEqual(sim.trial([1], 42), 2)
        path = np.array([[1,0],[1,1],[0,1]])
        sim = Diffusion(path, [], 1, 1)
        self.assertEqual(sim.trial([0], 42), 2)

    def test_rng_isolation(self):
        sim = Diffusion(np.ones((4,1)), [0], .2, 3)
        np.random.seed(19)
        expected = np.random.random()
        np.random.seed(19)
        a = sim.trial([1], 23)
        self.assertEqual(np.random.random(), expected)
        self.assertEqual(a, sim.trial([1], 23))

    def test_terminal_full_pool_dqn(self):
        sim = Diffusion(np.ones((3,1)), [0], .2, 2)
        cfg = dict(learning_rate=.001, replay_capacity=20, episodes=2,
                   batch_size=1, gamma=.9, target_update_steps=1)
        selected, *_ = learn(np.array([1,2]), np.ones((3,2)), 2, sim, [17,18], cfg, 42)
        self.assertEqual(set(selected), {1,2})


if __name__ == '__main__':
    unittest.main()
