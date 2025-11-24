import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, sparse: bool = True):
        super().__init__()
        #weight matrix in R^(vocab_size, d_model) initialized with small random values
        self.weight = nn.Parameter(torch.randn(vocab_size, d_model) * 0.01, requires_grad=True)

        self.sparse = sparse

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        #tokens: (B, T) -> (B, T, d_model)
        x = self.weight[tokens]
        if torch.is_autocast_enabled():
            return x.to(torch.get_autocast_dtype())
        return x