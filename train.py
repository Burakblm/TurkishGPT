import os
import time
import math
from dataclasses import dataclass
from typing import Optional

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

class Trainer:

    def __init__(self,
                args: TrainArgs,
                model: Transformer,
                train_data: Dataset,
                optimizer: torch.optim.Optimizer,
                save_every: int,
                snapshot_path: str,
                val_data: Optional[Dataset] = None,
                ) -> None:
        
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        if os.path.exists(snapshot_path):
            print("Loading Snapshot")
            self._load_snapshot(snapshot_path)
        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Model training continues from the {self.epochs_run} epoch")

    def _run_batch(self, inputs, targets):
        logits, loss = self.model(inputs, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        bs = len(next(iter(self.train_data))[0])
        print(f"[GPU:{self.gpu_id}] Epoch {epoch} | Batchsize: {bs} | Steps: {len(self.train_data)}")
        for inputs, targets in self.train_data:
            inputs = inputs.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(inputs, targets)

    def _save_snapshot(self, epoch):
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        #torch.save(snapshot, "snapshot.pt")
        print(f"Epoch {epoch} | training snapshot save at snapshot.pt")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
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
    model = Transformer(turkgptconfing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return train_data, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def main(total_epoch: int, save_every: int, snapshot_path: str = "snapshot.pt"):
    ddp_setup()
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size=32)
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path)
    trainer.train(total_epoch)
    destroy_process_group()


if __name__ == "__main__":
    import sys
    total_epochs = int(sys.argv[1])
    save_every = int(sys.argv[2])
    snapshot_path = str(sys.argv[3])
    main(total_epoch=total_epochs, save_every=save_every, snapshot_path=snapshot_path)
