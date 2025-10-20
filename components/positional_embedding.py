import torch
import torch.nn as nn
import math

# based on RoFormer, Su et al. 2023
class RoPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int):
        # TODO: max_seq_len is currently unused, but could be used to precompute 1/thetas and cos and sin for efficiency
        super().__init__()
        assert d_model % 2 == 0 # d_model must be even
        half_d = d_model // 2

        thetas = (10000 ** (torch.arange(0, half_d).float() / d_model))
        self.register_buffer("thetas", thetas)

    def rot(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        "x: B x L x D, positions: L, "
        freqs = torch.einsum("l,d->ld", positions.float(), 1.0 / self.thetas) # L x D/2
        cos = freqs.cos().repeat_interleave(2, dim=-1).unsqueeze(0) # 1 x L x D
        sin = freqs.sin().repeat_interleave(2, dim=-1).unsqueeze(0) # 1 x L x D

        x1 = x[..., ::2] # B x L x D/2
        x2 = x[..., 1::2] # B x L x D/2
        x_rot = torch.stack([-x2, x1], dim=-1).reshape_as(x)

        return x * cos + x_rot * sin