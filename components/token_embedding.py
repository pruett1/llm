import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        #weight matrix in R^(vocab_size, d_model) initialized with small random values
        self.weight = nn.Parameter(torch.randn(vocab_size, d_model) * 0.01)

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        #tokens: (B, T) -> (B, T, d_model)
        return self.weight[tokens]