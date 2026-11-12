import unittest
import numpy as np
import torch
from Diffusion_In_HyperGraph import HG_Diffusion_SICP,HG_Diffusion_SICP_pos
from CompeteAlgorithm import Compete_DQN
class Tests(unittest.TestCase):
 def test_competitor_wins_and_synchronous(self):
  m=np.array([[1,0],[1,0],[1,1],[0,1]])
  self.assertEqual(HG_Diffusion_SICP(m,[0,1,2,3],[0],[1],1,1,seed=1),(4,1,2,1))
  self.assertEqual(HG_Diffusion_SICP_pos(m,[0,1,2,3],[0],[1],1,1,seed=1),(4,1,2,1))
 def test_terminal_mask_has_finite_loss(self):
  a=Compete_DQN(0,.03,.9,2,2,[1,1],[2,2],[0,1])
  loss=a.update(([np.zeros(2)],[0],[1.],[np.zeros(2)],[True],[np.array([False,False])]))
  self.assertTrue(np.isfinite(loss))
 def test_nonterminal_empty_mask_rejected(self):
  a=Compete_DQN(0,.03,.9,2,2,[1,1],[2,2],[0,1])
  with self.assertRaises(ValueError):a.update(([np.zeros(2)],[0],[1.],[np.zeros(2)],[False],[np.array([False,False])]))
if __name__=='__main__':unittest.main()
