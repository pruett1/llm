import torch
import torch.nn as nn

import os
import random

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

        with torch.autocast(device_type=x.device.type, enabled = torch.is_autocast_enabled()):
            x = self.lm_head(x)
        return x
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_length: int) -> torch.Tensor:
        self.reset_cache()

        for _ in range(max_length):
            logits = self.forward(input_ids, use_cache = True)
            next_token = torch.argmax(logits[:, -1, :], dim = -1, keepdim = True) #TODO: add top-p sampling w/ temperature instead of greedy
            input_ids = torch.cat((input_ids, next_token), dim = 1)
        
        return input_ids
    
    def save(self, path: str, optimizer= None, scheduler = None, epoch: int = None, rng_state: bool = False):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': {
                'vocab_size': self.token_embedding.weight.size(0),
                'd_model': self.token_embedding.weight.size(1),
                'max_seq_len': self.layers[0].attn.rope.max_seq_len,
                'n_heads': self.layers[0].attn.n_heads,
                'n_layers': len(self.layers),
                'ff_mult': self.layers[0].ff[0].out_features // self.layers[0].ff[0].in_features,
                'dropout': self.layers[0].dropout.p
            }
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if rng_state:
            checkpoint['rng_state'] = {
                'torch': torch.get_rng_state(),
                'mps': torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
                'python': random.getstate()
            }

        torch.save(checkpoint, path)
        # print(f"saved model checkpoint to {path}")
    
    @classmethod
    def load(cls, path: str, device: torch.device, optimizer = None, scheduler = None):
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print(f"loaded model checkpoint from {path}")

        #try to load optimizer and scheduler states if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        starting_epoch = 0
        if 'epoch' in checkpoint:
            starting_epoch = checkpoint['epoch']

        if 'rng_state' in checkpoint:
            rng_state = checkpoint['rng_state']
            torch.set_rng_state(rng_state['torch'].cpu())
            if torch.backends.mps.is_available() and rng_state['mps'] is not None:
                torch.mps.set_rng_state(rng_state['mps'].cpu())
            random.setstate(rng_state['python'])

        return model, starting_epoch