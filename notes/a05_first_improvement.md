# First improvement

First a diagnosis:
- one of the reasons the generative model is doing so poorly is it is only looking at the last character (single) to do the prediction. Note the following:

```python
    def generate(
        self,
        idx: torch.LongTensor,
        max_new_tokens: int
    ):
        """
        Args:
            idx: a sequence of long ints representing an input
        """
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            
            # focus only on the last time step
            logits = logits[:, -1, :]
            ...
```

