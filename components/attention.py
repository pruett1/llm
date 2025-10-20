import torch
import torch.nn as nn
import math
from components.positional_embedding import RoPositionalEmbedding

# modified self attention with rotary position embedding
class StreamingAttention(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, n_heads: int = 4, attn_dropout: float = 0.1, proj_dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.ModuleList([nn.Linear(d_model, self.head_dim) for _ in range(n_heads)])
        self.k_proj = nn.ModuleList([nn.Linear(d_model, self.head_dim) for _ in range(n_heads)])
        self.v_proj = nn.ModuleList([nn.Linear(d_model, self.head_dim) for _ in range(n_heads)])
        self.o_proj = nn.Linear(d_model, d_model)

        self.rope = RoPositionalEmbedding(max_seq_len, self.head_dim)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

        self.register_buffer("cache_k", None)
        self.register_buffer("cache_v", None)
        self.register_buffer("cache_index", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor, use_cache: bool) -> torch.Tensor:
        B, L, _ = x.size()
        all_heads = []

        for i in range(self.n_heads):
            q = self.q_proj[i](x)
            k = self.k_proj[i](x)
            v = self.v_proj[i](x)

            pos = torch.arange(self.cache_index, self.cache_index + L, device=x.device)
            q = torch.stack([self.rope.rot(q[:, j, :], pos[j]) for j in range(L)], dim = 1)
            k = torch.stack([self.rope.rot(k[:, j, :], pos[j]) for j in range(L)], dim = 1)

            if use_cache and self.cache_k is not None:
                k = torch.cat([self.cache_k[i], k], dim = 1)
                v = torch.cat([self.cache_v[i], v], dim = 1)
            
            qkT = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            smax = torch.softmax(qkT, dim = -1)
            smax = self.attn_dropout(smax)

            attn = torch.matmul(smax, v)
            all_heads.append(attn)

            if use_cache:
                self.cache_k = self.cache_k if self.cache_k is not None else [None] * self.n_heads
                self.cache_v = self.cache_v if self.cache_v is not None else [None] * self.n_heads
                self.cache_k[i] = k.detach()
                self.cache_v[i] = v.detach()

        if use_cache:
            self.cache_index += L

        out = torch.cat(all_heads, dim = -1)
        out = self.o_proj(out)
        out = self.proj_dropout(out)

        return out