import os
import time
import math

import torch
from torch.utils.data import DataLoader
import numpy as np

from model import ModelArgs, Transformer
from prepare_data import load_tensor, prepare_data
from data_loader import GPTDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 32
block_size = 32

train_data = GPTDataset("train", batch_size=batch_size, block_size=block_size)
dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False, num_workers=0)


