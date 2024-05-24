from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import random
from prepare_data import load_tensor
import torch


class GPTDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int = 1024, batch_size: int = 8):
        self.data = data
        self.block_size = block_size
        self.batch_size = batch_size
        self.n_samples = len(self.data) - self.block_size
        self.current_index = 0

    def __getitem__(self, index):
        if self.current_index >= self.n_samples:
            self.current_index = random.randint(0, self.block_size)
            
        
        start_index = self.current_index
        end_index = min(start_index + self.block_size, len(self.data))
        
        x = self.data[start_index:end_index]
        y = self.data[start_index + 1:end_index + 1]
        
        self.current_index += self.block_size
        return x, y

    def __len__(self):
        return math.ceil(self.n_samples / self.block_size)