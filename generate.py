import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn import functional as F

import os

from utils import get_tokenizer
from model import Transformer, ModelArgs

path = os.getcwd() + "/model/snapshot.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "flaot16"


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

def generate_text(model, max_token: int = 100, temprature: float = 1.0):
    model.eval()
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_token):
            logits, _ = model(idx)
            logits = logits[:, -1, :] / temprature
            props = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(props, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
    return tokenizer.decode(idx[0].tolist())


if __name__ == "__main__":
    import sys
    model.to(device)
    max_token = int(sys.argv[1])
    temprature = float(sys.argv[2])
    print(generate_text(model, max_token=max_token, temprature=temprature))