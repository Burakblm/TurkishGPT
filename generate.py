import torch
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F

import os

from utils import get_tokenizer
from model import Transformer, ModelArgs

path = os.getcwd() + "/model/snapshot.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "flaot16"

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12344"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


block_size = 1024
vocab_size= 32000
n_layer = 12
n_head = 12
n_embd= 768
dropout = 0.0
bias = False

tokenizer = get_tokenizer()
model_args = dict(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        dropout=dropout,
        bias=bias,
        n_embd=n_embd,
    )
turkgptconfing = ModelArgs(**model_args)
model = Transformer(turkgptconfing)

snapshot = torch.load(path)
model.load_state_dict(snapshot["MODEL_STATE"])


def generate_text(model, max_token):
    model.eval()
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_token):
            logits, _ = model(idx)
            logits = logits[:, -1, :]
            props = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(props)
            idx = torch.cat((idx, idx_next), dim=1)
            

    idx = idx[0].tolist()
    return tokenizer.decode(idx)


def main(rank: int, world_size: int):
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    ddp_setup(rank=rank, world_size=world_size)
    model.cuda(rank)

    mp.spawn(run, args=(rank, world_size, n_gpus), nprocs=n_gpus)



def run(rank, world_size):
    global model
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    ddp_setup(rank, world_size)
    
    if torch.cuda.is_available():
        model.cuda(rank)
        model = DDP(model, device_ids=[rank])

    gen_text = generate_text(model, 100)

    print(f"GPU {rank} generated text : {gen_text}")
    destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    mp.spawn(run, args=(world_size,), nprocs=world_size)
