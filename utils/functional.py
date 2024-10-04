import yaml
import torch

def gen_velocity(m: torch.Tensor):
    dm = m[:, 1:] - m[:, :-1]
    return dm

def read_config(fname: str):
    with open(fname) as f:
        config = yaml.safe_load(f)
    return config