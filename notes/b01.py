from typing import List, Tuple, Callable

import torch
from torch import nn


def make_encoder_and_decoder(chars: List[str]):
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
    return encode, decode


def make_train_valid_data(text: str, encode: Callable):
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data


def load_input_text():
    with open("./notes/data/input.txt", "r", encoding="utf-8") as f:
        return f.read()

def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    data: torch.Tensor,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    device: str = "cpu",
) -> float:
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data, block_size, batch_size, device)
        _, loss = model(X, Y)
        losses[k] = loss.item()
    return losses.mean()
