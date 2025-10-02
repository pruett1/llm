import torch
import torch.nn as nn
import math

# based on RoFormer, Su et al. 2023
class RoPositionalEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        assert d_model % 2 == 0 # d_model must be even
        half_d = d_model // 2

        thetas = (10000 ** (torch.arange(0, half_d).float() / d_model))
        self.register_buffer("thetas", thetas)

    def rot(self, x, position: int) -> torch.Tensor:
        freqs = position * self.thetas
        cos = freqs.cos().repeat_interleave(2)
        sin = freqs.sin().repeat_interleave(2)

        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x_rot = torch.stack([-x2, x1], dim=-1).reshape_as(x)

        return x * cos + x_rot * sin