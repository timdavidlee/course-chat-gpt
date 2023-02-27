## The first Bigram model

```python
import torch
import torch.nn as nn
import numpy as np

from torch.nn import functional as F
torch.manual_seed(1337)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        
        # vocab size square this is almost like attention
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets = None):
        logits = self.token_embedding_table(idx) # (B, T, C)
        
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


A sample of how to interact with the model

```python
m = BigramLanguageModel(my_data_class.vocab_size)
xb1, yb1 = next(iter(train_dataloader))
print(xb1.shape, yb1.shape)
out, _ = m(xb1, yb1)
print(out.shape)  

expected_loss = np.log(1/ 65)
print("average expected loss: {:.04f}".format(expected_loss))
```

The output is the following:

```python
torch.Size([4, 8]) torch.Size([4, 8])
torch.Size([32, 65])
average expected loss: -4.1744
```

Trying out the generative method:

```python
starter_idx = torch.zeros((1, 1), dtype=torch.long)
predicted_token_ids = m.generate(starter_idx, max_new_tokens=100)[0].tolist()
output = my_data_class.decode(predicted_token_ids)
print("[untrained model output]: {}".format(output))
```

Note that this model has not been trained on anything, so we expect the output to be terrible

```
[untrained model output]: 
o,3foB$MVkzM'Q&!.iFlttRjXKoiw3y?c;iQJaSYI!poBdttRnYC My,WWThP:X3pEH-wNA'zr'XnQhfp
S-b3fp
jCJyhsm.WUG
```

## Let's train the model

```python
# training
# training
m = BigramLanguageModel(my_data_utils.vocab_size)
optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)

batch_size = 32
last_loss = 999999
train_dataloader = DataLoader(train_dataset, batch_size)
for steps in range(1, 20_000):
    xb, yb = next(iter(train_dataloader))
    
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if steps % 500 == 0:
        print("steps: {:,} | {:.04f}".format(steps, loss.item()))
        imp = (last_loss - loss.item()) / last_loss
        if imp < 0.01:
            print("loss improvement: {}".format(imp))
            break
        else:
            last_loss = loss.item()
```

The following is the output:

```
steps: 500 | 4.0185
steps: 1,000 | 3.1753
steps: 1,500 | 2.4656
steps: 2,000 | 1.9072
steps: 2,500 | 1.5012
steps: 3,000 | 1.2300
steps: 3,500 | 1.0615
steps: 4,000 | 0.9599
steps: 4,500 | 0.8977
steps: 5,000 | 0.8580
steps: 5,500 | 0.8316
steps: 6,000 | 0.8134
steps: 6,500 | 0.8006
steps: 7,000 | 0.7912
steps: 7,500 | 0.7843
loss improvement: 0.00871082619848534
```

What happens if we run the generative function now?

```python
starter_idx = torch.zeros((1, 1), dtype=torch.long)
predicted_token_ids = m.generate(starter_idx, max_new_tokens=300)[0].tolist()
output = my_data_utils.decode(predicted_token_ids)
print("[untrained model output]: {}".format(output))
```


```python
[untrained model output]: 
Bed an:
Npre an:
Bee we any prororororgBeny we pwefured wefoce Cit any we we any any prstitize prororoceny any proceny Citire proceGPPPPPTZhy Cizen:
Be Cize Cize an:
Be wed furoreeed we an:
Be Cit forstZpred ween:
Be any any foce Cizen:
Be fororst prorstize weeforst Firoree foce wen:
Ben:
Beny wed f
```

