import os
import time
import math
from dataclasses import dataclass
from typing import Optional
import random
import re
from tqdm import tqdm
from dotenv import load_dotenv
from prepare_data import load_tensor


import torch
from torch.nn import functional as F
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, Dataset
from torch.distributed.fsdp import fully_sharded_data_parallel as FSDP
import numpy as np
import os

from model import ModelArgs, Transformer
from prepare_data import load_tensor, prepare_data
from utils import get_tokenizer
from data_loader import GPTDataset
from lora import Lora

from utils import get_tokenizer

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
MODEL_PATH = os.getenv("MODEL_PATH")
MODEL_DIR = os.getenv("MODEL_DIR")

tokenizer = get_tokenizer()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ddp = int(os.environ.get("RANK", -1)) != -1
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' 

scaler = GradScaler()


num_samples_for_loss = 100
max_iters = 100
block_size = 1024
batch_size = 16
vocab_size= 32002
n_layer = 12
n_head = 12
n_embd= 768
dropout = 0.0
bias = False
learning_rate = 1e-4

@dataclass
class TrainArgs:
    num_epochs: int = 1
    batch_size: int = 8
    block_size: int = 1024
    num_samples_for_loss: int = 200
    learning_rate: float = 1e-4
    dataset: str = "nutuk"
    vocab_size: int = 32000
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
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
                epochs: int,
                ddp: bool = False,
                num_samples_for_loss: int = 100,
                device: Optional[str] = None,
                args: Optional[TrainArgs] = None,
                val_data: Optional[Dataset] = None,
                ) -> None:
        
        self.args = args
        self.ddp = ddp
        self.device = device
        self.num_samples_for_loss = num_samples_for_loss
        self.epochs = epochs
        snapshot_path = snapshot_path

        if self.ddp:
            self.gpu_id = int(os.environ["LOCAL_RANK"])
            self.model = model.to(self.gpu_id)
            self.model = DDP(self.model, device_ids=[self.gpu_id])
        else:
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
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model.module.load_state_dict(snapshot["MODEL_STATE"])
        else:
            self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Model training continues from the {self.epochs_run} epoch")

    def _run_batch(self, inputs, targets):
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(dtype=torch.float16):
            logits, loss = self.model(inputs, targets)
        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()
        return loss

    def _run_epoch(self, epoch):
        bs = len(next(iter(self.train_data))[0])
        print(f"[GPU:{self.gpu_id if self.ddp else self.device}] Epoch {epoch} | Batchsize: {bs} | Steps: {len(self.train_data)}")
        loop = tqdm(enumerate(self.train_data), total=len(self.train_data), leave=True)
        for i, (inputs, targets) in loop:
            inputs = inputs.to(self.gpu_id if self.ddp else self.device)
            targets = targets.to(self.gpu_id if self.ddp else self.device)
            loss = self._run_batch(inputs, targets)
            loop.set_description(f"Epoch [{epoch}/{self.epochs}]")
            loop.set_postfix(loss = loss.item())
        if ddp:
            if self.gpu_id == 0:
                out = self.calculate_loss()
                print(f"Train loss: {out['train']:.4f}" + (f" | Val loss : {out['val']:.4f}" if self.val_data is not None else ""))
        else:
            out = self.calculate_loss()
            print(f"Train loss: {out['train']:.4f}" + (f" | Val loss : {out['val']:.4f}" if self.val_data is not None else ""))

    def _save_snapshot(self, epoch):
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict() if self.ddp else self.model.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        torch.save(snapshot, os.getcwd() + "/model/snapshot.pt")
        print(f"Epoch {epoch} | training snapshot save at snapshot.pt\n")

    @torch.no_grad()
    def calculate_loss(self):
        out = {}
        self.model.eval()
        for i in ["train", "val"]:
            if self.split_data[i] is not None:
                losses = torch.zeros(self.num_samples_for_loss)
                loop = tqdm(enumerate(self.split_data[i]), total=self.num_samples_for_loss-1, leave=True)
                for j, (inputs, targets) in loop:
                    j += 1
                    inputs = inputs.to(self.gpu_id if self.ddp else self.device)
                    targets = targets.to(self.gpu_id if self.ddp else self.device)
                    logits, loss = self.model(inputs, targets)
                    losses[j % self.num_samples_for_loss] = loss.item()
                    loop.set_description(f"{i} Average Loss")
                    loop.set_postfix(loss = loss.item())
                    if j % self.num_samples_for_loss == 0:
                        break
            out[i] = losses.mean()
        self.model.train()
        return out
        
    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.ddp:
                if self.gpu_id == 0 and epoch % self.save_every == 0:
                    self._save_snapshot(epoch)
            else:
                if epoch % self.save_every == 0:
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

def load_train_objs(training_type = "pretraining", data_path: str = "train.pt", split_rate: float = 0.97):
    data = load_tensor(data_path)
    data_size = len(data)
    print(f"All data size: {data_size}")
    train_data = data[:int(data_size * split_rate)]
    print(f"Train data size: {len(train_data)}")
    val_data = data[int(data_size * split_rate):]
    print(f"Validation data size: {len(val_data)}")
    train_data = GPTDataset(data=train_data, batch_size=1, block_size=block_size)
    val_data = GPTDataset(data=val_data, batch_size=1, block_size=block_size)
    # yeni kodlar
    train_data = GPTDataset("train", batch_size=1, block_size=block_size)

    if training_type == "pretraining":
        model = Transformer(turkgptconfing)
        model = model.to(device)
    else:
        model = Transformer(turkgptconfing)
        mode = model.to(device)
        lora = Lora(model)
        lora.freeze_non_lora_params()
        lora.print_model_parameters()
        lora.enable_disable_lora(enabled=True)
        total_params, trainable_params = lora.count_parameters()
        print(f"Toplam parametre sayısı: {total_params}")
        print(f"Eğitilebilir parametre sayısı: {trainable_params}")
        model = lora.model
        model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
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

def main(total_epoch: int, batch_size: int, save_every: int, training_type: str = "pretraining", snapshot_path: str = "snapshot.pt", data_path="train.pt"):
    if ddp:
        ddp_setup()
    train_data, val_data, model, optimizer = load_train_objs(training_type=training_type, data_path=data_path)
    train_data = prepare_dataloader(train_data, batch_size=batch_size)
    val_data = prepare_dataloader(val_data, batch_size=batch_size)
    trainer = Trainer(model=model, train_data=train_data, val_data=val_data, optimizer=optimizer, epochs=total_epoch, ddp=ddp, save_every=save_every, snapshot_path=snapshot_path, num_samples_for_loss=num_samples_for_loss, device=device)
    #num_of_params = sum(p.numel() for p in model.parameters())
    num_of_params = '{:,}'.format(sum(p.numel() for p in model.parameters())).replace(",", ".")
    print(f"number of model parameters: {num_of_params}")
    trainer.train(total_epoch)
    if ddp:
        destroy_ddp()
if __name__ == "__main__":
    import sys
    total_epochs = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    save_every = int(sys.argv[3])
    snapshot_path = str(sys.argv[4])
    data_path = str(sys.argv[5])
    main(total_epoch=total_epochs, batch_size=batch_size, save_every=save_every, snapshot_path=snapshot_path, data_path=data_path)
