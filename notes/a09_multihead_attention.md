# What is multi-head attention?

Training multiple heads of attention in parallel, so we will create an abstraction that allows for multiple heads.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, *args, **kwargs):
        super().__init__()
        self.heads = nn.ModuleList([Head(*args, **kwargs)])

    def forward(self, x):
        return torch.cat(
            [h(x) for h in self.heads],
            dim=-1
        )
```

This will add more throughput to the processing. Can look at attention in detail.

In the paper a feed-forward layer is added to give the model more time to think:

```python
class FeedForward(nn.Module):
    def __init__(self, n_emb: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, n_emb),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
```

Sample generative output

```
And ther thidcow and is and the airen bobe to: anvirtand me?
I ands ar hiphe us hat vet?
F dilth ane aw crup and not com onour
Yowns
Moof is her!
Amil;
And buaes ifee-sen cin lat Heriviov the and Wing.

DWAFNWIO:
Wel lind teall thus wonchiry:
Aur Maiss hawty.

KIBEUSIR:
I patelives
Mom
my wod mothake on indo wher eiicks withe dourive wies ime st so mower; the the danderupt for ar igis! muf thin inled
Atatfif Pried my of.

HINY IER:
Widby ake adsal ther ghe thidin cour ay andy Iry to chan the!
Ay
```