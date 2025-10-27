import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from components.tokenizer_cpp import BlBPETokenizer

from model.transformer import Transformer
from helpers.text_handler import jsonl_to_texts

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
    def collate_fn(batch, tokenizer: BlBPETokenizer):
        return pad_sequence(batch, batch_first=True, padding_value = tokenizer.get_special_token_id("<|PAD|>"))
    return collate_fn

def train_model(model: nn.Module, token_data: list[int], tokenizer: BlBPETokenizer, epochs: int = 10000, lr: float = 1e-4, batch_size: int = 32, seq_len: int = 128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    dataloader = DataLoader(token_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_factory(tokenizer))

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        