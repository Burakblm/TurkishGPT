import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F



device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Generate:
    def __init__(self, model_path: str = "turkishgpt.pt" ) -> None:
        pass


    def generate(self):
        pass

        