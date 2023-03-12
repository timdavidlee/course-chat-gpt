import torch
from torch import nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        
        # vocab size square this is almost like attention
        # should note this is considered to be row-wise
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets  = None):
        # Batch | Time | Channel
        # batch=4, time=8, channel is vocab size=65
        logits = self.token_embedding_table(idx) # (B, T, C)
        
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