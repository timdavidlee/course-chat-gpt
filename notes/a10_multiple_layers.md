# Adding multiple layers

https://youtu.be/kCc8FmEb1nY?t=5201


## We want to stack multiple Attention + Feedforward processing


```python
import torch
from torch import nn

class AttnFFBlock(nn.Module)::
    def __init__(self, n_emb: int, n_head: int):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_emb)

    def forward(self, x):
        x = self.sa(x)
        x = self.ffwd(x)
        return x
```

And in the model, this will look like the following:

```python
...
self.token_embedding_table = nn.Embedding(...)
self.position_embedding_table = nn.Embedding(...)
self.blocks = nn.Sequential(
    Block(n_embed, n_head=4),
    Block(n_embed, n_head=4),
    Block(n_embed, n_head=4)
)
```

This will help a lot, but also at this point we will be building a "deeper" network. The dangerous thing about this is that the deeper networks can get more unstable. To address this, we will implement some additional techniques.

## Residual Connections

This idea was taken from RESNET. The main idea is there is a clear pathway that skips data manipulation. 

- in the beginning, these residual pathways are initialized do not adjust the input
- but as training progresses, the residual blocks come online + start learning
- this allows for much deeper networks
- this helps with stability of model training

Implementation

```python
class AttnFFBlock(nn.Module):
    def __init__(self, n_emb: int, n_head: int):
        super().__init__()
        head_size = n_emb // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_emb)

    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x
```

The multi-head attention must be updated too

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, n_emb, *args, **kwargs):
        super().__init__()
        self.heads = nn.ModuleList([Head(*args, **kwargs)])
        self.proj = nn.Linear(n_emb, n_emb)

    def forward(self, x):
        out = torch.cat(
            [h(x) for h in self.heads],
            dim=-1
        )
        out = self.proj(out)
        return out
```

The feed forward will also need to be updated.

- according to the paper, the embedding dim should be 4x
- A projection layer will be added

```python
class FeedForward(nn.Module):
    def __init__(self, n_emb: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb), # projection layer
        )
    def forward(self, x):
        return self.net(x)
```