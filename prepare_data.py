import torch
from utils import get_tokenizer
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")

tokenizer = get_tokenizer()

def prepare_data(path: str = "data.txt", split_rate: float = 0.9):
    with open(path, "r") as f:
        data = f.read()
    
    print("Data is being prepared...")
    data = tokenizer.encode(data)
    data = torch.tensor(data, dtype=torch.int16)
    data_size = len(data)
    train_data = data[:int(data_size * split_rate)]
    print(f"Train data size: {len(train_data)}")
    val_data = data[int(data_size * split_rate):]
    print(f"Validation data size: {len(val_data)}")

    save_tensor(train_data, "train")
    save_tensor(val_data, "val")
    

def save_tensor(tensor: torch.Tensor, save_name: str):
    tensor = tensor.to(torch.int16)
    save_name = "data/" + save_name + ".pt"
    torch.save(tensor, save_name)

def load_tensor(path: str = "train"):
    path = DATA_PATH
    #path = os.getcwd() + "/data/" + path + ".pt"
    data = torch.load(path)
    return data


if __name__ == "__main__":
    import sys
    data_path = str(sys.argv[1])
    split_rate = float(sys.argv[2])
    prepare_data(data_path, split_rate)
