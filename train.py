import os
import time
import math
import pickle

from contextlib import nullcontext

import torch
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import ModelArgs, Transformer

