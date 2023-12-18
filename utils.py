import torch
from transformers import AutoTokenizer
import os


def install_tokenizer(path: str = "meta-llama/Llama-2-7b-chat-hf"):
    out_path = os.getcwd() + "/model"
    print(out_path)
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    tokenizer.save_pretrained(out_path)


def get_tokenizer():
    model_path = os.getcwd() + "/model"
    tokenizer =  AutoTokenizer.from_pretrained(model_path, use_fast=True)
    return tokenizer

