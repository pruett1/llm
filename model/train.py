import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.amp import GradScaler, autocast

from components.tokenizer_cpp import BlBPETokenizer
from model.transformer import Transformer

import time
import os

from tqdm import tqdm

def masked_lm_loss(logits: torch.Tensor, labels: torch.Tensor, output_token_id: int, pad_id: int) -> torch.Tensor:
    B, L, V = logits.size()

    output_idx = (labels == output_token_id).int().argmax(dim=1)
    # has_output = (labels == output_token_id).any(dim=1)

    mask = torch.arange(L, device=labels.device).unsqueeze(0) >= output_idx.unsqueeze(1)
    mask &= (labels != pad_id)

    logits = logits.reshape(-1, V)
    labels = labels.reshape(-1)
    mask = mask.reshape(-1)

    mask = mask.bool()
    logits = logits[mask]
    labels = labels[mask]

    loss = nn.functional.cross_entropy(logits, labels, reduction='mean')
    return loss # Avoid division by zero, shouldnt happen but just in case

class Collator:
    def __init__(self, tokenizer: BlBPETokenizer, seq_len: int, total_epochs: int, get_current_epoch: int):
        self.pad_id = tokenizer.get_special_token_id("<|PAD|>")
        self.output_id = tokenizer.get_special_token_id("<|OUTPUT|>")
        self.desc_id = tokenizer.get_special_token_id("<|DESC|>")
        self.ex_id = tokenizer.get_special_token_id("<|EXAMPLES|>")
        self.seq_len = seq_len
        self.total_epochs = total_epochs
        self.get_current_epoch = get_current_epoch
    
    def find_first(self, tokens, token_id):
        mask = (tokens == token_id).int()
        if not mask.any():
            return None
        return torch.argmax(mask).item()
    
    def __call__(self, batch):
        epoch = self.get_current_epoch
        progress = min(1.0, epoch / self.total_epochs)

        batch_size = len(batch)
        input_batch = torch.full((batch_size, self.seq_len), self.pad_id, dtype=torch.long)
        label_batch = torch.full((batch_size, self.seq_len), self.pad_id, dtype=torch.long)
        pad_row = torch.full((self.seq_len + 1, ), self.pad_id, dtype=torch.long)

        for b_idx, t in enumerate(batch):
            t_size = t.size(0)

            # Get special token idxs
            d_start = self.find_first(t, self.desc_id)
            e_start = self.find_first(t, self.ex_id)
            o_start = self.find_first(t, self.output_id)

            if d_start == None and e_start == None and o_start == None: #no special tokens -> pretraining or finetuning 1
                if t_size > self.seq_len: #tokens are too long randomly select a subset
                    start_idx = torch.randint(0, t_size - self.seq_len, ()).item()
                    t = t[start_idx:start_idx + self.seq_len]


            elif o_start == None and (d_start != None or e_start != None): #malformed data
                continue

            else:
                # Split into desc, examples+constraints, output
                desc = t[d_start:e_start]
                examples_constraints = t[e_start:o_start]
                output = t[o_start:]
                output_size = output.size(0)

                #if output is too long, randomly select a subset that fits
                #if too long take 75% of seq_len from output (allows some context)
                if output_size > self.seq_len:
                    sub_size = int(self.seq_len * 0.75)
                    start_idx = torch.randint(0, output_size - sub_size, ()).item()
                    output = output[start_idx:start_idx + sub_size]

                context = torch.cat([desc, examples_constraints], dim=0)

                allowed_context_len = int((desc.size(0)) + (context.size(0) - desc.size(0)) * progress) # Smooth ramping of context
                max_context_len = min(int(self.seq_len - output_size), allowed_context_len) # Max size of context we can fit, capped to allowed_context_len

                # If context is too long, randomly select a subset that fits
                if allowed_context_len > max_context_len:
                    start_idx = torch.randint(0, allowed_context_len - max_context_len, ()).item()
                    context = context[start_idx:start_idx + max_context_len]
                else:
                    context = context[:allowed_context_len]

                t = torch.cat([context, output], dim=0)
                t_size = t.size(0)
            # Pad if needed
            t_size = t.size(0)
            if t_size < self.seq_len + 1:
                temp = pad_row.clone()
                temp[:t_size] = t
                t = temp

            input_batch[b_idx] = t[:-1]
            label_batch[b_idx] = t[1:]

        return input_batch, label_batch

class WarmupLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, d_model: int, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        lr = (self.d_model ** -0.5) * min((self._step_count ** -0.5), (self._step_count * (self.warmup_steps ** -1.5)))
        return [lr for _ in self.optimizer.param_groups]

def train_model(model: Transformer, token_data: Dataset, tokenizer: BlBPETokenizer, epochs: int = 10000, lr: float = 1e-4, warmup_steps: int = 4000, batch_size: int = 32, seq_len: int = 128, resume: bool = False):
    # Set device for model
    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    print(f"Training on device: {device}")
    # model = torch.compile(model)
    model.to(device)
    model.train()

    # # initalize grad scaler
    # scaler = GradScaler(device=device.type, init_scale = 2.**16, growth_factor = 2.0, backoff_factor = 0.5)

    current_epoch = 0
    # Set up dataloader with collator
    collator = Collator(tokenizer, seq_len, epochs, current_epoch)
    dataloader = DataLoader(token_data, batch_size=batch_size, shuffle=True, collate_fn=collator, num_workers=4, persistent_workers=True)

    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)
    scheduler = WarmupLR(optimizer, warmup_steps=warmup_steps, d_model=model.token_embedding.weight.size(1))

    # Enable mid point resume
    start_epoch = 0
    if resume and os.path.exists('checkpoints/train_interrupt.pt'):
        model, start_epoch = Transformer.load('checkpoints/tokenizer.pt', device, optimizer=optimizer, scheduler=scheduler)
        tokenizer = BlBPETokenizer.load('checkpoints/tokenizer.bin')
        print(f"Resuming training from epoch {start_epoch}")
    current_epoch = start_epoch

    try:
        for epoch in tqdm(range(start_epoch, epochs), desc="Training Epochs: ", leave=False, total=epochs, initial=start_epoch):
            start = time.time()
            total_loss = 0.0
            current_epoch = epoch
            collator.get_current_epoch = current_epoch
            t_processed = 0

            for batch in dataloader:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                t_processed += inputs.numel()

                optimizer.zero_grad()

                # mixed-precision forward
                with autocast(device_type=device.type):
                    logits = model(inputs, use_cache=False)
                    loss = masked_lm_loss(logits, labels, tokenizer.get_special_token_id("<|OUTPUT|>"), tokenizer.get_special_token_id("<|PAD|>"))

                loss_val = loss.item() # before scaling for logging

                #scaling
                # scaler.scale(loss).backward()

                # scaler.step(optimizer)
                # scaler.update()
                loss.backward()
                optimizer.step()

                scheduler.step()
                total_loss += loss_val
            
            avg_loss = total_loss / len(dataloader)
            end = time.time()
            if (epoch + 1) % 100 == 0 or epoch == 0:
                current_lr = scheduler.get_last_lr()[0]
                tqdm.write(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | {t_processed * len(dataloader)/(end-start):.2f} tok/s")
            
            if (epoch + 1) % (epochs // 4) == 0:
                model.save(f'checkpoints/model_epoch_{epoch+1}.pt', optimizer=optimizer, scheduler=scheduler, epoch=epoch, rng_state=True)
            
            if epoch % (epochs * 0.05) == 0 and epoch != 0:
                model.save('checkpoints/train_interrupt.pt', optimizer=optimizer, scheduler=scheduler, epoch=epoch, rng_state=True)
                tokenizer.save('checkpoints/tokenizer.bin')
                # tqdm.write("Model and tokenizer saved in case of interruptions")

    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        model.save('checkpoints/train_interrupt.pt', optimizer=optimizer, scheduler=scheduler, epoch=epoch, rng_state=True)
        print("Saving tokenizer...")
        tokenizer.save('checkpoints/tokenizer.bin')
        print(f"Model and tokenizer saved at epoch: {epoch}\n Exiting training loop")