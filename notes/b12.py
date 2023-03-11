import torch
from torch import nn
from torch.nn import functional as F
from .b11 import AttnFFBlockV1


class BigramAttentionLanguageModelV5(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_emb: int,
        n_layer: int,
        n_head: int,
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
            *[AttnFFBlockV1(n_emb, n_head=n_head, block_size=block_size, dropout=dropout, device=device)
            for _ in range(n_layer)]
        )
        self.layer_norm = nn.LayerNorm(n_emb)
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
