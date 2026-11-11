import importlib.util
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
from shared_experiment_core import Diffusion
from s2v_network import Structure2Vec


def load_reference(name, file):
    spec=importlib.util.spec_from_file_location(name,file)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FastTests(unittest.TestCase):
    def test_diffusion_and_memo_match_reference_exactly(self):
        reference=load_reference('reference_core',Path(__file__).parent/'reference/core_original.py')
        generator=np.random.default_rng(42)
        for horizon in (1,4):
            matrix=(generator.random((17,9))<.25).astype(int)
            old=reference.Diffusion(matrix,[0,1],.3,horizon)
            new=Diffusion(matrix,[0,1],.3,horizon)
            seeds=np.arange(10,dtype=np.uint64)
            for selected in ([2],[3,5],[5,3],[2,6,9]):
                expected=old.samples(selected,seeds)
                self.assertTrue(np.array_equal(new.samples(selected,seeds),expected))
                self.assertTrue(np.array_equal(new.samples(selected,seeds),expected))
            self.assertGreater(new.cache_hits,0)
            self.assertTrue(np.array_equal(new.samples([3,5],seeds+50),old.samples([3,5],seeds+50)))

    def test_sparse_forward_and_gradients_match_dense_reference(self):
        reference=load_reference('reference_network',Path(__file__).parent/'reference/s2v_original.py')
        torch.manual_seed(42)
        old=reference.Structure2Vec(4,23,4)
        new=Structure2Vec(4,23,4);new.load_state_dict(old.state_dict())
        adjacency=(torch.rand(23,23)<.15).float()
        weights=adjacency.clone();x=torch.rand(3,23,2)
        expected=old(x,adjacency,weights)
        actual=new(x,adjacency.to_sparse().coalesce(),weights)
        torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-5)
        expected.square().mean().backward();actual.square().mean().backward()
        for (name,a),(_,b) in zip(old.named_parameters(),new.named_parameters()):
            torch.testing.assert_close(a.grad,b.grad,rtol=5e-4,atol=5e-5,msg=name)

    def test_microbatch_accumulation_preserves_full_batch_gradient(self):
        from s2v_adapter import make_network
        torch.manual_seed(15)
        matrix=np.ones((7,2))
        a=make_network(matrix,[0],np.array([1,2,3]),dict(s2v_embedding_dim=4,s2v_iterations=3))
        b=make_network(matrix,[0],np.array([1,2,3]),dict(s2v_embedding_dim=4,s2v_iterations=3))
        b.load_state_dict(a.state_dict())
        states=torch.rand(5,7);expected=torch.rand(5,3)
        (a(states)-expected).square().mean().backward()
        for begin in range(0,5,2):
            ((b(states[begin:begin+2])-expected[begin:begin+2]).square().sum()/expected.numel()).backward()
        for (name,x),(_,y) in zip(a.named_parameters(),b.named_parameters()):
            torch.testing.assert_close(x.grad,y.grad,rtol=5e-4,atol=5e-5,msg=name)


if __name__=='__main__':unittest.main()
