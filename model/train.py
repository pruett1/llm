import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from components.tokenizer_cpp import BlBPETokenizer

from model.transformer import Transformer

def masked_lm_loss(logits: torch.Tensor, labels: torch.Tensor, output_token_id: int) -> torch.Tensor:
    mask = (labels == output_token_id).cumsumprod(dim=1) > 0
    
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    mask = mask[:, 1:]

    logits = logits.reshape(-1, logits.size(-1))
    labels = labels.reshape(-1)
    mask = mask.reshape(-1).float()

    token_loss = nn.functional.cross_entropy(logits, labels, reduction='none')
    masked_loss = token_loss * mask

    return masked_loss.sum() / mask.sum()

def collate_fn_factory(tokenizer: BlBPETokenizer):
    def collate_fn(batch):
        return pad_sequence(batch, batch_first=True, padding_value = tokenizer.get_special_token_id("<|PAD|>"))
    return collate_fn

class WarmupLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, d_model: int, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        lr = (self.d_model ** -0.5) * min((self._step_count ** -0.5), (self._step_count * (self.warmup_steps ** -1.5)))
        return [lr for _ in self.optimizer.param_groups]

def train_model(model: Transformer, token_data: list[int], tokenizer: BlBPETokenizer, epochs: int = 10000, lr: float = 1e-4, batch_size: int = 32, seq_len: int = 128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    dataloader = DataLoader(token_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_factory(tokenizer))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = WarmupLR(optimizer, warmup_steps=1, d_model=model.token_embedding.weight.size(1))

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            batch = batch[:, :seq_len].to(device)

            optimizer.zero_grad()
            logits = model(batch, use_cache=False)

            loss = masked_lm_loss(logits, batch, tokenizer.get_special_token_id("<|OUTPUT|>"))
            loss.backward()

            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        if (epoch) % 1 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
    