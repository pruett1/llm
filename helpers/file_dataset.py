import torch
from torch.utils.data import Dataset
import random

class StreamingFileDataset(Dataset):
    def __init__(self, file_path: str, sample_frac: float = 1.0):
        self.path = file_path
        self.sample_frac = sample_frac

        self.offsets = []

        with open(self.path, 'r', encoding='utf-8') as f:
            offset = 0
            for line in f:
                if self.sample_frac < 1.0 and self.sample_frac > random.random():
                    offset += len(line)
                    continue
                self.offsets.append(offset)
                offset += len(line)
            
    
    def __len__(self):
        return len(self.offsets)
    
    def __get_item__(self, idx: int):
        offset = self.offsets[idx]
        with open(self.path, 'r', encoding='utf-8') as f:
            f.seek(offset)
            line = f.readline().strip()
            tokens = [int(tok) for tok in line.split()]
        return torch.tensor(tokens, dtype=torch.long)