"""Corrected local S2V adaptation, not an unmodified published baseline."""
import numpy as np
import torch
from torch import nn
from s2v_network import Structure2Vec, QNetwork


def make_network(matrix, rivals, pool, cfg):
    # Match the inherited dense adjacency / weighted projected-graph architecture.
    # Full graph is embedded; only the action set is restricted to the common pool.
    from scipy.sparse import csr_matrix
    incidence = csr_matrix(matrix.astype(np.float32))
    weights = (incidence @ incidence.T).toarray()
    np.fill_diagonal(weights, 0)
    adjacency = (weights > 0).astype(np.float32)
    n = len(matrix)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            dim = cfg.get('s2v_embedding_dim', 32)
            self.embedding = Structure2Vec(dim, n, cfg.get('s2v_iterations', 32))
            self.q = QNetwork(dim)
            self.register_buffer('adjacency', torch.tensor(adjacency))
            self.register_buffer('sparse_adjacency', torch.tensor(adjacency).to_sparse().coalesce(), persistent=False)
            self.register_buffer('weights', torch.tensor(weights))
            labels = torch.zeros(n)
            labels[list(rivals)] = 1
            self.register_buffer('rivals', labels)
            self.register_buffer('pool', torch.tensor(pool, dtype=torch.long))

        def forward(self, state):
            single = state.ndim == 1
            if single:
                state = state.unsqueeze(0)
            labels = torch.stack((state, self.rivals.expand_as(state)), dim=-1)
            values = self.q(self.embedding(labels, self.sparse_adjacency, self.weights))[:, self.pool]
            return values[0] if single else values

    return Model()
