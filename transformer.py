import torch
import torch.nn as nn

from components.token_embedding import TokenEmbedding
from components.decoder_block import DecoderBlock

class Transformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, n_heads: int, n_layers: int, ff_mult: int, dropout: float = 0.1):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(DecoderBlock(d_model, max_seq_len, n_heads, ff_mult, dropout))

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.token_embedding.weight
    
    def reset_cache(self):
        for layer in self.layers:
            layer.reset_cache()

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        x = self.token_embedding(x)
        for layer in self.layers:
            x = layer(x, use_cache)
        x = self.norm(x)
        return self.lm_head(x)
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_length: int) -> torch.Tensor:
        self.reset_cache()

        for _ in range(max_length):
            logits = self.forward(input_ids, use_cache = True)
            next_token = torch.argmax(logits[:, -1, :], dim = -1, keepdim = True)
            input_ids = torch.cat((input_ids, next_token), dim = 1)
        
        return input_ids