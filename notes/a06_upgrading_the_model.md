# Adding some changes


## 1. Adding an embedding dimension

Previously

```python
self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
```

Paraphase: storing the general likelihood of next character. So given `A`, whats the chance of the other characters?

Now:

```python
self.token_embedding_table = nn.Embedding(vocab_size, n_emb)
self.lm_head = nn.Linear(n_embed, vocab_size)
```

Paraphrase: Each letter has some data related to it `n_emb` float numbers. Then some combination of those numbers will be multiplied down to the frequencies of the next character

To accommodate this, need to make changes in the `forward` function as well:

```python
token_emb = self.token_embedding_table(idx) # (B, T, C)
logits = self.lm_head(tok_emb)  # (B, T, vocab_size)
```

## 2. Encode the position


```
self.position_embedding_table = nn.Embedding(block_size, n_emb)
```

and in the `forward` part of the function

```
pos_emb = self.position_embedding_table(torch.arange(T, device=device))
x = token_emb + pos_emb
```


```python
import torch
import torch.nn as nn
import numpy as np

from torch.nn import functional as F
torch.manual_seed(1337)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, n_emb: int = 32):
        super().__init__()
        
        # vocab size square this is almost like attention
        # should note this is considered to be row-wise
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb)

        # encoding where the character is in the overall
        self.position_embedding_table = nn.Embedding(block_size, n_emb)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets = None):
        B, T = idx.shape
        # Batch | Time | Channel
        # batch=4, time=8, channel is NOW the embed
        
        # get the token embedding
        token_emb = self.token_embedding_table(idx) # (B, T, C)

        # get the position embedding
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = token_emb + pos_emb
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # logits = scores of what will be next in a sequence

        # batch, target, channels
        B, T, C = logits.shape
        if targets is None:
            return logits, None

        # unwinding
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx: torch.LongTensor, max_new_tokens: int):
        """
        Args:
            idx: a sequence of long ints representing an input
        """
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            
            # focus only on the last time step
            logits = logits[:, -1, :]
            
            # get the softmax probabilities
            probs = F.softmax(logits, dim=-1)

            # sample from the distribtuion (so its not deterministic)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # extends the sequence
            idx = torch.cat((idx, idx_next), dim=1) 

        return idx 
```