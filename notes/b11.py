import torch
from torch import nn
from torch.nn import functional as F

from .b10 import FeedForward


class HeadV1(nn.Module):
    """one head of attention"""
    def __init__(
        self,
        head_size: int,
        n_emb: int,
        block_size: int,
        dropout: float,
        device: str = "cpu"
    ):
        super().__init__()
        self.device = device

        # linear layers
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)

        # not a param of the module
        # this is a buffer, assigning to module
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size).to(device))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # attention
        wei = q @ k.transpose(-2, -1) * C **-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  #(B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v 
        return out


class MultiHeadAttentionV1(nn.Module):
    def __init__(
        self,
        num_heads: int,
        n_emb: int,
        head_size: int,
        block_size: int,
        dropout: float,
        *args,
        **kwargs
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                HeadV1(
                    head_size=head_size,
                    n_emb=n_emb,
                    block_size=block_size,
                    dropout=dropout,
                    *args, **kwargs
                ) for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(n_emb, n_emb)

    def forward(self, x):
        out = torch.cat(
            [h(x) for h in self.heads],
            dim=-1
        )
        out = self.proj(out)
        return out


class AttnFFBlockV1(nn.Module):
    def __init__(self, n_emb: int, n_head: int, block_size: int, dropout: float, device: str):
        super().__init__()
        head_size = n_emb // n_head
        self.sa = MultiHeadAttentionV1(
            n_head,
            n_emb,
            head_size,
            block_size,
            dropout,
            device=device
        )
        self.ffwd = FeedForward(n_emb)
        self.ln1 = nn.LayerNorm(n_emb)  # adding layer norm, has trainable
        self.ln2 = nn.LayerNorm(n_emb)  # adding layer norm, has trainable

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # normalize the features
        x = x + self.ffwd(self.ln2(x))  # normalize the features
        return x


class BigramAttentionLanguageModelV4(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_emb: int,
        block_size: int,
        dropout: float,
        device: str = "cpu"
    ):
        super().__init__()
        self.device = device
        
        # vocab size square this is almost like attention
        # should note this is considered to be row-wise
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb)
        self.block_size = block_size

        # encoding where the character is in the overall
        self.position_embedding_table = nn.Embedding(block_size, n_emb)
        self.blocks = nn.Sequential(
            AttnFFBlockV1(n_emb, n_head=4, block_size=block_size, dropout=dropout, device=device),
            AttnFFBlockV1(n_emb, n_head=4, block_size=block_size, dropout=dropout, device=device),
            AttnFFBlockV1(n_emb, n_head=4, block_size=block_size, dropout=dropout, device=device),
            nn.LayerNorm(n_emb)
        )
        self.lm_head = nn.Linear(n_emb, vocab_size)

    def forward(self, idx, targets = None):
        B, T = idx.shape

        # get the token embedding
        token_emb = self.token_embedding_table(idx) # (B, T, C)
        # get the position embedding
        pos_emb = self.position_embedding_table(torch.arange(T).to(self.device))
        x = token_emb + pos_emb
        x = self.blocks(x)
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
            # crop IDx to the last block size tokens
            idx_cond = idx[:, -self.block_size:]

            # get the predictions
            logits, loss = self(idx_cond)
            
            # focus only on the last time step
            logits = logits[:, -1, :]
            
            # get the softmax probabilities
            probs = F.softmax(logits, dim=-1)

            # sample from the distribtuion (so its not deterministic)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # extends the sequence
            idx = torch.cat((idx, idx_next), dim=1) 

        return idx 
