import torch
import os
from utils import get_tokenizer
from model import Transformer, ModelArgs

path = os.getcwd() + "/snapshot.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

model.to(device)

idx = torch.zeros((1, 1), dtype=torch.long, device=device)
res = model.generate(idx=idx, do_sample=True, top_k=10, temprature=0.8, max_new_tokens=500)[0].tolist()
print(tokenizer.decode(res))
