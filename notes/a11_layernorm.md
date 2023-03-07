## Layer Norm and Batch Norm

### Batch Normalization

Across the batch dimension, any individual neuron has unit gaussian distribution, meaning:

- mean is 0
- stdev is 1

A quick example (will use pre-built)

```python
import torch
from torch import nn

modz = nn.BatchNorm1d(num_features=100)
x = torch.randn(32, 100) # batch size of 32 x 100 dim vecs
x = modz(x)
x.shape, x[0, :].mean(), x[0, :].std(), x[1, :].mean(), x[1, :].std()
```

```
(torch.Size([32, 100]),
 tensor(0.0474, grad_fn=<MeanBackward0>),
 tensor(1.0037, grad_fn=<StdBackward0>),
 tensor(0.0294, grad_fn=<MeanBackward0>),
 tensor(1.1358, grad_fn=<StdBackward0>))
```

Layer Norm

```python
import torch

class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.out = None

    def __call__(self, x: torch.Tensor):
        # forward pass (this will be looking row-wise)
        xmean = x.mean(1, keepdim=True) # get the batch mean
        xvar = x.var(1, keepdim=True) # batch variance
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)  # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]
```

Now that we have implemented this. Lets add it in:

- general comment: very few transformer details have changed over the years
- the orignal paper, layer norm is applied after the transformation through the attention head
- in current times its more common to apply the norm BEFORE the attention is applied


```python
class AttnFFBlock(nn.Module):
    def __init__(self, n_emb: int, n_head: int):
        super().__init__()
        head_size = n_emb // n_head
        self.sa = MultiHeadAttention(n_head, n_emb, head_size)
        self.ffwd = FeedForward(n_emb)
        self.ln1 = nn.LayerNorm(n_emb)  # adding layer norm, has trainable
        self.ln2 = nn.LayerNorm(n_emb)  # adding layer norm, has trainable

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # normalize the features
        x = x + self.ffwd(self.ln2(x))  # normalize the features
        return x
```

Will add layernorm after the block stack, in the Lagnague model

```python
        self.blocks = nn.Sequential(
            AttnFFBlock(n_emb, n_head=4),
            AttnFFBlock(n_emb, n_head=4),
            AttnFFBlock(n_emb, n_head=4),
            nn.LayerNorm(n_emb) # adding here
        )
        self.lm_head = nn.Linear(n_emb, vocab_size)
```

## Dropout

Takes neural network, drops out certain weights so the model will not memorize the data. Makes it more robust and essentially adds regularization

```
```


## Note that the training at this point will be SIGNIFICANTLY SLOWER

- There's a slow startup time (it will hang)
- Then afterwards, i swapped to a GPU to do the training

Here's the output after predicting:

```
KING RICHARD II:
Shall be stir a senators again:
Brief Mercutio
My arm that us all to become as a funward,
And follow me to heaven from the heart more.
You, beg it end, I heard it to move thee.

WARWICK:
Ay, just, leave you well us wonder yourself.

KING HENRY VI:
He came at last your mistress than
And your white is your maids, and hie my means,--

KING RICHARD III:
From your present! my lord, I leave,
Your Prince, Signio; I speak too, is a royal thing:
For this county, and your battle may lose
```