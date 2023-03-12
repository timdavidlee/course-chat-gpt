import torch
import torch.nn as nn
import numpy as np

from torch.nn import functional as F
torch.manual_seed(1337)


class Head(nn.Module):
    """one head of attention"""
    def __init__(
        self,
        head_size: int,
        n_emb: int,
        block_size: int = 8,
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
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # attention
        wei = q @ k.transpose(-2, -1) * C **-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  #(B, T, T)

        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v 
        return out


class BigramAttentionLanguageModelV1(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_emb: int = 32,
        block_size: int = 8,
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
        self.sa_head = Head(
            head_size=n_emb,
            n_emb=n_emb,
            block_size=block_size,
            device=device
        )
        self.lm_head = nn.Linear(n_emb, vocab_size)

    def forward(self, idx, targets = None):
        B, T = idx.shape

        # get the token embedding
        token_emb = self.token_embedding_table(idx) # (B, T, C)
        # get the position embedding
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = token_emb + pos_emb
        x = self.sa_head(x)  # apply one head of self-attention
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
