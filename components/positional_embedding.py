import torch
import torch.nn as nn

# based on RoFormer, Su et al. 2023
class RoPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        assert d_model % 2 == 0 # d_model must be even
        half_d = d_model // 2
        self.max_seq_len = max_seq_len

        thetas = (10000 ** (torch.arange(0, half_d).float() / d_model)) # D/2
        seq_pos = torch.arange(max_seq_len).float() # L
        freqs = torch.einsum("l,d->ld", seq_pos, 1.0 / thetas) # L x D/2
        
        cos = freqs.cos().repeat_interleave(2, dim=-1) # L x D
        sin = freqs.sin().repeat_interleave(2, dim=-1) # L x D
        
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)


    def rot(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        "x: B x L x H X D, positions: L, "
        cos = self.cos[positions].unsqueeze(0).unsqueeze(2).to(x.dtype) # 1 x L x 1 x D
        sin = self.sin[positions].unsqueeze(0).unsqueeze(2).to(x.dtype) # 1 x L x 1 x D

        x1 = x[..., ::2] # B x L x D/2
        x2 = x[..., 1::2] # B x L x D/2
        x_rot = torch.stack([-x2, x1], dim=-1).reshape_as(x)

        return x * cos + x_rot * sin