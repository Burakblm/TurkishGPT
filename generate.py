import torch
import os
from utils import get_tokenizer
from model import Transformer

path = os.getcwd() + "/snapshot.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = get_tokenizer()

model = Transformer()

snapshot = torch.load(path)
model.load_state_dict(snapshot["MODEL_STATE"])


idx = torch.zeros((1, 1), dtype=torch.long, device=device)
res = model.generate(idx=idx, do_sample=True, top_k=10, temprature=0.8, max_new_tokens=500)[0].tolist()
print(tokenizer.decode(res))
