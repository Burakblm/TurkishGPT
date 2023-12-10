import os
import time
import math

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


batch_size = 32
block_size = 32
vocab_size= 32000
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.0
bias = False

train_data = GPTDataset("train", batch_size=batch_size, block_size=block_size)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False, num_workers=0)

val_data = GPTDataset("val", batch_size=batch_size, block_size=block_size)
val_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False, num_workers=0)

split = {"train": train_dataloader,
         "val": val_dataloader}

model_args = dict(block_size=block_size,
                  vocab_size=vocab_size,
                  n_layer=n_layer,
                  n_head=n_head,
                  n_embd=n_embd,
                  dropout=dropout,
                  bias=bias
                  )

turkgptconfing = ModelArgs(**model_args)
model = Transformer(turkgptconfing)

print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


@torch.no_grad()
def calculate_loss():
    out = {}
    model.eval()
    for i in ["train", "val"]:
        dataloader = split[i]
        losses = torch.zeros(eval_iters)
        for i, (inputs, targets) in enumerate(dataloader):
            logits, loss = model(inputs, logits)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


for i, (inputs, targets) in enumerate(train_dataloader):
    if i == 5:
        break
    print("batch number: ",i)

    logits, loss = model(inputs, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    print(loss.item())
    optimizer.step()


def generate(model, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = model(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

inpt = torch.tensor([tokenizer.encode("mustafa kemal atatürk")], dtype=torch.long)
print(tokenizer.decode(generate(model, inpt, max_new_tokens=200)[0].tolist()))