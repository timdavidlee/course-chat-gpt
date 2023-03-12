import torch
from torch import nn
import torch.nn.functional as F

from .b09 import Head


class AttnFFBlock(nn.Module):
    def __init__(self, n_emb: int, n_head: int):
        super().__init__()
        head_size = n_emb // n_head
        self.sa = MultiHeadAttention(n_head, n_emb,head_size)
        self.ffwd = FeedForward(n_emb)

    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, n_emb: int, head_size: int, *args, **kwargs):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                Head(head_size=head_size, n_emb=n_emb, *args, **kwargs) for _ in range(num_heads)
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


class BigramAttentionLanguageModelV3(nn.Module):
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
        self.blocks = nn.Sequential(
            AttnFFBlock(n_emb, n_head=4),
            AttnFFBlock(n_emb, n_head=4),
            AttnFFBlock(n_emb, n_head=4),
        )
        self.lm_head = nn.Linear(n_emb, vocab_size)

    def forward(self, idx, targets = None):
        B, T = idx.shape

        # get the token embedding
        token_emb = self.token_embedding_table(idx) # (B, T, C)
        # get the position embedding
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
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
