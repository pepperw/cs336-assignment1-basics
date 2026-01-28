import torch
import torch.nn as nn
import math

class Linear(nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.d_in = in_features
        self.d_out = out_features
        self.W = nn.Parameter(torch.empty((self.d_out, self.d_in), device = device, dtype = dtype))
        self.reset_parameters()

    def reset_parameters(self):
        variance = 2.0 / (self.d_in + self.d_out)
        std = math.sqrt(variance)
        nn.init.trunc_normal_(self.W, mean =  0.0, std = std, a = -3.0 * std, b = 3.0 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.t()

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device = device, dtype = dtype))
        self.reset_parameters()
    def reset_parameters(self):
        variance = 2.0 / (self.num_embeddings + self.embedding_dim)
        std = math.sqrt(variance)
        nn.init.trunc_normal_(self.embedding, mean =  0.0, std = std, a = -3.0 * std, b = 3.0 * std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        '''
        Input shape: (batch_size, seq_len)
        Output shape: (batch_size, seq_len, embedding_dim)
        '''
        return self.embedding[token_ids]




