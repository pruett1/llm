import torch
from torch.utils.data import Dataset
import random

class TokenDataset(Dataset):
    def __init__(self, source):
        if isinstance(source, str):
            self.ds = StreamingFileDataset(source)
        elif isinstance(source, list):
            self.ds = InMemDataset(source)
        else:
            raise ValueError("Invalid source type.")
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx: int):
        return self.ds[idx]

    def __del__(self):
        try:
            if hasattr(self.ds, "__del__"):
                self.ds.__del__()
        except Exception:
            pass

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

class InMemDataset(Dataset):
    def __init__(self, in_mem: list[list[int]]):
        self.data = [torch.tensor(seq, dtype = torch.long) for seq in in_mem]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        return self.data[idx]