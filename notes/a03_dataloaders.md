# Dataloaders + Datasets

Loading portions of the data a little at a time

```python
import torch
from typing import Sequence


def read_in_shakespear_data(file: str) -> str:
    """returns a long string buffer"""
    with open(file, "r", encoding="utf-8") as f:
        return f.read()


class MyDataUtils:
    def __init__(self, text_file: str):
        bigtext = read_in_shakespear_data(text_file)
        print("total length of characters: {:,}".format(len(bigtext)))

        self.bigtext = bigtext
        self.unique_characters = sorted(list(set(bigtext)))
        self.vocab_size = len(self.unique_characters)
        print("".join(self.unique_characters))

        self.encoder = {char: j for j, char in enumerate(self.unique_characters)}
        self.decoder = {j: char for j, char in enumerate(self.unique_characters)}

    def encode(self, some_str: str) -> Sequence[int]:
        return [self.encoder[char] for char in some_str]

    def decode(self, sequence: Sequence[int]) -> str:
        chars = [self.decoder[num] for num in sequence]
        return "".join(chars)
    
    def get_tokens(self, text: str) -> torch.LongTensor:
        token_ids = torch.tensor(
            self.encode(text),
            dtype=torch.long,
        )
        print(token_ids.shape, token_ids.dtype)
        print(token_ids[:100])
        return token_ids 
    
my_data_utils = MyDataUtils("./notes/data/input.txt")
token_ids = my_data_utils.get_tokens(my_data_utils.bigtext)

train_cutoff = int(0.9 * len(token_ids))
train_data = token_ids[: train_cutoff]
valid_data = token_ids[train_cutoff:]
```

Dumps the following:

```
total length of characters: 1,115,394

 !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
torch.Size([1115394]) torch.int64
tensor([18, 47, 56, 57, 58,  1, 15, 47, 58, 47, 64, 43, 52, 10,  0, 14, 43, 44,
        53, 56, 43,  1, 61, 43,  1, 54, 56, 53, 41, 43, 43, 42,  1, 39, 52, 63,
         1, 44, 59, 56, 58, 46, 43, 56,  6,  1, 46, 43, 39, 56,  1, 51, 43,  1,
        57, 54, 43, 39, 49,  8,  0,  0, 13, 50, 50, 10,  0, 31, 54, 43, 39, 49,
         6,  1, 57, 54, 43, 39, 49,  8,  0,  0, 18, 47, 56, 57, 58,  1, 15, 47,
        58, 47, 64, 43, 52, 10,  0, 37, 53, 59])
```


## About dataloads

Often it is useful too look at a subset of a larger sequence


```python
block_size = 8
token_ids[:block_size + 1]
```

```
tensor([18, 47, 56, 57, 58,  1, 15, 47, 58])
```

### Problem setup:

A few characters will be provided, and the problem will be the predict what the next character will be. So for example:

Input:

```
[18] -> try to predict [47]
[18, 47] -> try to predict [ 56]
[18, 47, 56] -> try to predict [ 57]
[18, 47, 56, 57] -> try to predict [ 58]
```

The below code will demonstrate how to do this

```python
x = train_data[: block_size]
y = train_data[1: block_size + 1]
for t in range(block_size):
    context = x[: t + 1]
    target = y[t]
    print(f"when input is {context} the target: {target}")
```

And the output looks something like the following:

```
when input is tensor([18]) the target: 47
when input is tensor([18, 47]) the target: 56
when input is tensor([18, 47, 56]) the target: 57
when input is tensor([18, 47, 56, 57]) the target: 58
when input is tensor([18, 47, 56, 57, 58]) the target: 1
when input is tensor([18, 47, 56, 57, 58,  1]) the target: 15
when input is tensor([18, 47, 56, 57, 58,  1, 15]) the target: 47
when input is tensor([18, 47, 56, 57, 58,  1, 15, 47]) the target: 58
```

## Using pytorch's built in classes

```python
import torch
from typing import Tuple
from torch.utils.data import DataLoader, Dataset

class MyTorchDataset(Dataset):
    def __init__(self, window_size: int, long_seq: torch.LongTensor):
        self.seq = long_seq
        self.window_size = window_size

    def __len__(self):
        return self.seq.shape[0] - self.window_size - 1

    def __getitem__(self, idx) -> Tuple[torch.LongTensor, torch.LongTensor]: 
        x = self.seq[idx: idx + self.window_size]
        y = self.seq[idx + 1 : idx + 1 + self.window_size]
        return x, y
```

As an example if we do the following:

```python
batch_size = 4
block_size = 8

train_dataset = MyTorchDataset(block_size, train_data)
valid_dataset = MyTorchDataset(block_size, valid_data)

train_dataloader = DataLoader(train_dataset, batch_size)
valid_dataloader = DataLoader(valid_dataset, batch_size)

next(iter(train_dataloader))
```

The output from the dataloader is
```python
[tensor([[18, 47, 56, 57, 58,  1, 15, 47],
         [47, 56, 57, 58,  1, 15, 47, 58],
         [56, 57, 58,  1, 15, 47, 58, 47],
         [57, 58,  1, 15, 47, 58, 47, 64]]),
 tensor([[47, 56, 57, 58,  1, 15, 47, 58],
         [56, 57, 58,  1, 15, 47, 58, 47],
         [57, 58,  1, 15, 47, 58, 47, 64],
         [58,  1, 15, 47, 58, 47, 64, 43]])]
```
