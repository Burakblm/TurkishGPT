import os
import time
import math
from dataclasses import dataclass
from typing import Optional
import random

import torch
from torch.nn import functional as F
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler # verileri tüm gpu lara dağıtmak için
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

from model import ModelArgs, Transformer
from prepare_data import load_tensor, prepare_data
from utils import get_tokenizer
from data_loader import GPTDataset

from utils import get_tokenizer

tokenizer = get_tokenizer()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ddp = int(os.environ.get("RANK", -1)) != -1


eval_iters = 10
max_iters = 100
block_size = 1024
batch_size = 16
vocab_size= 32000
n_layer = 8
n_head = 8
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
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False
    out_dir: str = "out"


def ddp_setup():
    init_process_group(backend="nccl")

def destroy_ddp():
    destroy_process_group()


class Trainer:

    def __init__(self,
                model: Transformer,
                train_data: DataLoader,
                optimizer: torch.optim.Optimizer,
                save_every: int,
                snapshot_path: str,
                ddp: bool = False,
                eval_iters: int = 100,
                device: Optional[str] = None,
                args: Optional[TrainArgs] = None,
                val_data: Optional[Dataset] = None,
                ) -> None:
        
        self.args = args
        self.ddp = ddp
        self.device = device
        self.eval_iters = eval_iters

        if self.ddp:
            self.gpu_id = int(os.environ["LOCAL_RANK"])
            self.model = model.to(self.gpu_id)
            self.model = DDP(self.model, device_ids=[self.gpu_id])

        self.model = model.to(self.device)
        self.epochs_run = 0
        self.train_data = train_data
        self.val_data = val_data
        self.split_data = {"train": self.train_data,
                           "val": self.val_data}
        self.optimizer = optimizer
        self.save_every = save_every
        if os.path.exists(snapshot_path):
            print("Loading Snapshot")
            self._load_snapshot(snapshot_path)

    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Model training continues from the {self.epochs_run} epoch")

    def _run_batch(self, inputs, targets):
        print("Run Batch funciton: inputs",inputs.device)
        print("Run Batch funciton: targets",targets.device)

        logits, loss = self.model(inputs, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        bs = len(next(iter(self.train_data))[0])
        print(f"[GPU:{self.gpu_id if self.ddp else self.device}] Epoch {epoch} | Batchsize: {bs} | Steps: {len(self.train_data)}")
        for i, (inputs, targets) in enumerate(self.train_data):
            inputs = inputs.to(self.gpu_id if self.ddp else self.device)
            targets = targets.to(self.gpu_id if self.ddp else self.device)
            self._run_batch(inputs, targets)
            if i % eval_iters == 0:
                out = self.calculate_loss()
                print(f"Train loss: {out['train']:.4f}" + (f" | Val loss : {out['val']:.4f}" if self.val_data is not None else ""))

    def _save_snapshot(self, epoch):
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict() if self.ddp else self.model.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        torch.save(snapshot, "snapshot.pt")
        print(f"Epoch {epoch} | training snapshot save at snapshot.pt")
    
    @torch.no_grad()
    def calculate_loss(self):
        out = {}
        self.model.eval()
        for i in ["train", "val"]:
            if self.split_data[i] is not None:
                losses = torch.zeros(self.eval_iters)
                data = iter(self.split_data[i])
                for j in range(self.eval_iters):
                    inputs, targets = next(data)
                    print("Loss : ", inputs.device)
                    print("Loss : ", targets.device)
                    logits, loss = self.model(inputs, targets)
                    losses[j] = loss.item()
            out[i] = losses.mean()
        self.model.train()
        return out
        
    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id if self.ddp else self.device == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)


model_args = dict(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        dropout=dropout,
        bias=bias,
        n_embd=n_embd,
    )
turkgptconfing = ModelArgs(**model_args)

def load_train_objs():
    train_data = GPTDataset("train", batch_size=1, block_size=32)
    val_data = GPTDataset("val", batch_size=1, block_size=32)
    model = Transformer(turkgptconfing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return train_data, val_data, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int):
    if ddp:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(dataset)
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=0
        )

def main(total_epoch: int, save_every: int, snapshot_path: str = "snapshot.pt"):
    if ddp:
        ddp_setup()
    train_data, val_data, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(train_data, batch_size=32)
    val_data = prepare_dataloader(val_data, batch_size=32)
    trainer = Trainer(model=model, train_data=train_data, optimizer=optimizer, ddp=ddp, save_every=save_every, snapshot_path=snapshot_path,eval_iters=eval_iters, device=device)
    trainer.train(total_epoch)
    if ddp:
        destroy_ddp()

if __name__ == "__main__":
    import sys
    total_epochs = int(sys.argv[1])
    save_every = int(sys.argv[2])
    main(total_epoch=total_epochs, save_every=save_every)
