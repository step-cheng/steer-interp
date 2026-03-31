import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

PROJECT_DIR = "/nfshomes/scheng03/steer-interp"

def load_model_and_tokenizer(model_name: str, device_map: str = "auto", revision: str = "main", dtype=torch.bfloat16):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device_map, dtype=dtype, revision=revision
    ).eval()
    if not model.model.is_gradient_checkpointing:
        print('setting gradient checkpointing')
        model.model.gradient_checkpointing_enable()
        model.model.config.use_cache=False
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

