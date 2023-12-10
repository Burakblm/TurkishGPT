import os
import time
import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np

from model import ModelArgs, Transformer
from prepare_data import load_tensor, prepare_data
from utils import get_tokenizer
from data_loader import GPTDataset

from utils import get_tokenizer

tokenizer = get_tokenizer()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

eval_iters = 10
max_iters = 100
block_size = 1024
batch_size = 16
vocab_size= 32000
n_layer = 4
n_head = 4
n_embd= 256
dropout = 0.0
bias = False
learning_rate = 1e-3


@dataclass
class TrainArgs:
    num_epochs: int = 1
    batch_size: int = 32
    block_size: int = 32
    eval_iters: int = 200
    learning_rate: float = 1e-3
    dataset: str = "nutuk"
    vocab_size: int = 32000
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False
    out_dir: str = "out"
    


class Trainer:
    def __init__(self, args: TrainArgs, model: Transformer, train_dataset):
        self.args = args
        self.model = model
        self.model = self.model.to(device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.learning_rate)
        self.train_dataset = train_dataset


    def get_parameters_num(self):
        return sum(p.numel() for p in self.model.parameters())/1e6
    
    def generate(self, model, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = model(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
    def oku(self, max_new_tokens):
        inpt = torch.tensor([tokenizer.encode("mustafa kemal atatürk")], dtype=torch.long, device=device)
        print(tokenizer.decode(self.generate(self.model, inpt, max_new_tokens=max_new_tokens)[0].tolist()))

    
    def run(self):
        model, args = self.model, self.args

        data_loader = DataLoader(
            dataset = self.train_dataset,
            batch_size = args.batch_size,
            shuffle=False,
            num_workers=0
        )

        for epoch in range(args.num_epochs):
            print(f"Epoch: {epoch}")

            for i, (inputs, targets) in enumerate(data_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                logits, loss = model(inputs, targets)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                print(loss.item())
                self.optimizer.step()


if __name__ == "__main__":
    import sys
    num_epochs = int(sys.argv[1])
    block_size = int(sys.argv[2])
    batch_size = int(sys.argv[3])


    train_args = dict(
        num_epochs=num_epochs,
        batch_size=batch_size,
        block_size=block_size,
    )

    model_args = dict(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        dropout=dropout,
        bias=bias,
        n_embd=n_embd,
    )

    trainer_args = TrainArgs(**train_args)

    turkgptconfing = ModelArgs(**model_args)
    model = Transformer(turkgptconfing)

    train_data = GPTDataset("train", batch_size=1, block_size=block_size)

    trainer = Trainer(trainer_args, model, train_data)
    trainer.run()
    trainer.oku(1000)