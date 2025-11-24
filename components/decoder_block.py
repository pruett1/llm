import torch
import torch.nn as nn

from components.attention import StreamingAttention
# from components.add_norm import AddNorm

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int,  max_seq_len: int, n_heads: int = 4, ff_mult: int = 4, dropout: float  = 0.1):
        super().__init__()
        self.attn = StreamingAttention(d_model, max_seq_len, n_heads, attn_dropout=dropout, proj_dropout=dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.ff_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
    
    def reset_cache(self):
        self.attn.reset_cache()

    # using pre-norm structure
    def forward(self, x: torch.Tensor, use_cache: bool) -> torch.Tensor:
        x = self.attn_norm(x)
        dtype = x.dtype
        attn_out = self.attn(x, use_cache)
        x = x + self.dropout(attn_out.to(dtype))

        x = self.ff_norm(x)
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out.to(dtype))

        return x