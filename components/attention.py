import torch
import torch.nn as nn
from components.positional_embedding import RoPositionalEmbedding

# modified self attention with rotary position embedding
class StreamingAttention(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int,  n_heads: int = 4, attn_dropout: float = 0.1, proj_dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.rope = RoPositionalEmbedding(max_seq_len, self.head_dim)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

        self.register_buffer("cache_k", None)
        self.register_buffer("cache_v", None)
        self.register_buffer("cache_index", torch.tensor(0, dtype=torch.long))

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None
        self.cache_index = torch.tensor(0, dtype=torch.long)

    def forward(self, x: torch.Tensor, use_cache: bool) -> torch.Tensor:
        B, L, _ = x.size()

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=-1) # B, L, D

        # reshape to be heads -> B, L, N_h, D_h
        q = q.view(B, L, self.n_heads, self.head_dim)
        k = k.view(B, L, self.n_heads, self.head_dim)
        v = v.view(B, L, self.n_heads, self.head_dim)

        pos = torch.arange(self.cache_index.item(), self.cache_index.item() + L, device=x.device)
        q = self.rope.rot(q, pos)
        k = self.rope.rot(k, pos)

        # if cache cat over layers
        if use_cache and self.cache_k is not None:
            k = torch.cat([self.cache_k, k], dim=1)
            v = torch.cat([self.cache_v, v], dim=1)

        if use_cache:
            self.cache_k = k.detach()
            self.cache_v = v.detach()
            self.cache_index = self.cache_index + L

        # reshape so N_h is a batch dim for mat mul
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        qkT = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=q.dtype, device=q.device))
        smax = torch.softmax(qkT, dim = -1)
        smax = self.attn_dropout(smax)

        out = torch.matmul(smax, v) #B, H, L, D
        out = out.transpose(1, 2) #B, L, H, D

        out = out.reshape(B, L, self.d_model) #reshape to B, L, d_model
        out = self.o_proj(out)
        out = self.proj_dropout(out)

        return out