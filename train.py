import os
import time
import math

import torch
import numpy as np

from model import ModelArgs, Transformer
from prepare_data import load_tensor, prepare_data

device = "cpu"
device_type = "cpu"

batch_size = 32
block_size = 128


train_data = load_tensor("train")
val_data = load_tensor("val")

print(train_data.shape)
print(val_data.shape)
