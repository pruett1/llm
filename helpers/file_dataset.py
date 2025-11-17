import torch
from torch.utils.data import Dataset
import random

class StreamingFileDataset(Dataset):
    def __init__(self, file_path: str, sample_frac: float = 1.0):
        self.path = file_path
        self.sample_frac = sample_frac

        self.offsets = []

        with open(self.path, 'rb') as f:
            offset = 0
            for line in f:
                if self.sample_frac < 1.0 and self.sample_frac > random.random():
                    offset += len(line)
                    continue
                self.offsets.append(offset)
                offset += len(line)
            
        self._file = None
    
    def __len__(self):
        return len(self.offsets)
    
    def __getitem__(self, idx: int):
        if self._file is None:
            self._file = open(self.path, 'rb')

        offset = self.offsets[idx]
        self._file.seek(offset)
        line = self._file.readline().decode('utf-8').strip()
        tokens = [int(tok) for tok in line.split()]
        return torch.tensor(tokens, dtype=torch.long)
    
    def __del__(self):
        if self._file is not None:
            self._file.close()