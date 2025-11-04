import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from components.tokenizer_cpp import BlBPETokenizer
from model.transformer import Transformer

import time
import random
import os

from tqdm import tqdm

def masked_lm_loss(logits: torch.Tensor, labels: torch.Tensor, output_token_id: int, pad_id: int) -> torch.Tensor:
    mask = (labels == output_token_id).cumsum(dim=1) > 0
    mask = mask & (labels != pad_id)

    logits = logits.reshape(-1, logits.size(-1))
    labels = labels.reshape(-1)
    mask = mask.reshape(-1).float()

    token_loss = nn.functional.cross_entropy(logits, labels, reduction='none')
    masked_loss = token_loss * mask

    return masked_loss.sum() / (mask.sum() + 1e-8) # Avoid division by zero, shouldnt happen but just in case

def collate_fn_factory(tokenizer: BlBPETokenizer, seq_len: int, total_epochs: int, get_current_epoch):
    pad_id = tokenizer.get_special_token_id("<|PAD|>")
    output_id = tokenizer.get_special_token_id("<|OUTPUT|>")
    desc_id = tokenizer.get_special_token_id("<|DESC|>")
    ex_id = tokenizer.get_special_token_id("<|EXAMPLES|>")
    # cons_id = tokenizer.get_special_token_id("<|CONSTRAINTS|>")

    # Smoothly increase context length over training epochs, always include desc gradually add examples+constraints
    # if context is too long, randomly select a subset that fits
    def collate_fn(batch):
        epoch = get_current_epoch()
        progress = min(1.0, epoch / total_epochs)

        inputs = []
        labels = []

        for tokens in batch:
            t = torch.tensor(tokens, dtype=torch.long)

            # Get special token idxs
            d_start = (t == desc_id).nonzero(as_tuple=True)[0].item()
            e_start = (t == ex_id).nonzero(as_tuple=True)[0].item()
            # c_start = (t == cons_id).nonzero(as_tuple=True)[0]
            o_start = (t == output_id).nonzero(as_tuple=True)[0].item()

            # Split into desc, examples+constraints, output
            desc = t[d_start:e_start]
            examples_constraints = t[e_start:o_start]
            output = t[o_start:]

            #if output is too long, randomly select a subset that fits
            #if too long take 75% of seq_len from output (allows some context)
            if output.size(0) >= seq_len:
                sub_size = int(seq_len * 0.75)
                start_idx = random.randint(0, output.size(0) - sub_size)
                output = output[start_idx:start_idx + sub_size]

            context = torch.cat([desc, examples_constraints], dim=0)

            allowed_context_len = int((desc.size(0)) + (context.size(0) - desc.size(0)) * progress) # Smooth ramping of context
            max_context_len = min(int(seq_len - output.size(0)), allowed_context_len) # Max size of context we can fit, capped to allowed_context_len

            # If context is too long, randomly select a subset that fits
            if allowed_context_len > max_context_len:
                start_idx = random.randint(0, allowed_context_len - max_context_len)
                context = context[start_idx:start_idx + max_context_len]
            else:
                context = context[:allowed_context_len]

            t = torch.cat([context, output], dim=0)
            # Pad if needed
            if t.size(0) < seq_len:
                t = torch.cat([t, torch.full((seq_len - t.size(0),), pad_id, dtype=torch.long)], dim=0)
            
            inputs.append(t[:-1])
            labels.append(t[1:])
        
        return torch.stack(inputs), torch.stack(labels)

    return collate_fn

class WarmupLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, d_model: int, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        lr = (self.d_model ** -0.5) * min((self._step_count ** -0.5), (self._step_count * (self.warmup_steps ** -1.5)))
        return [lr for _ in self.optimizer.param_groups]

def train_model(model: Transformer, token_data: list[int], tokenizer: BlBPETokenizer, epochs: int = 10000, lr: float = 1e-4, batch_size: int = 32, seq_len: int = 128, resume: bool = False):
    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    print(f"Training on device: {device}")
    model.to(device)

    current_epoch = 0
    def get_current_epoch():
        return current_epoch

    dataloader = DataLoader(token_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_factory(tokenizer, seq_len, epochs, get_current_epoch))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = WarmupLR(optimizer, warmup_steps=4000, d_model=model.token_embedding.weight.size(1))

    start_epoch = 0
    if resume and os.path.exists('checkpoints/train_interrupt.pt'):
        model, start_epoch = Transformer.load('checkpoints/train_interrupt.pt', device, optimizer=optimizer, scheduler=scheduler)
        tokenizer = BlBPETokenizer.load('checkpoints/tokenizer.bin')
        print(f"Resuming training from epoch {start_epoch}")
    current_epoch = start_epoch

    model.train()
    try:
        for epoch in tqdm(range(start_epoch, epochs), desc="Training Epochs: ", leave=False, total=epochs, initial=start_epoch):
            start = time.time()
            total_loss = 0.0
            current_epoch = epoch

            for batch in dataloader:
                inputs, labels = batch

                optimizer.zero_grad()
                logits = model.forward(inputs, use_cache=False)

                logits.to(device)
                labels = labels.to(device)
                loss = masked_lm_loss(logits, labels, tokenizer.get_special_token_id("<|OUTPUT|>"), tokenizer.get_special_token_id("<|PAD|>"))
                loss.backward()

                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            end = time.time()
            if (epoch + 1) % 100 == 0 or epoch == 0:
                tqdm.write(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Time: {end - start:.2f}s")
            
            if epoch / epochs in [0.25, 0.5, 0.75, 1.0]:
                model.save(f'checkpoints/model_epoch_{epoch + 1}.pt')
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        model.save('checkpoints/train_interrupt.pt', optimizer=optimizer, scheduler=scheduler, epoch=epoch, rng_state=True)
        print("Saving tokenizer...")
        tokenizer.save('checkpoints/tokenizer.bin')
        print(f"Model and tokenizer saved at epoch: {epoch}\n Exiting training loop")