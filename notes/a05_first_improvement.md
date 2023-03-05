# First improvement: Using History

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

### When generating future tokens + words:

When considering a sequence:

```
A-B-C-D-?
```

The next letter can use the preceeding tokens as information to help the next item in the sequence. But considering training data:

```
A-B-C-D-[E]-F-G
```

Our prediction slot should NOT know the future, in this case `F` and `G`.


### Average the history

What is the easiest way for tokens to communicate to each other?

The easiest way to communicate the past is to a average of all the preceding token. So considering the example before

```
A-B-C-D-?
```

Can use all the information at `D` and average the information of `A-B-C-D` somehow to be useful. The following is how to write it out in pytorch. Consider the following:

```python
# consider the following example:

torch.manual_seed(1337)
B, T, C = 4, 8, 2 # batch, time, channels

# generate some random data
x_dummy = torch.randint(0, 5, size=(B, T, C), dtype=torch.float)
x_dummy
```

The output looks like the following:

```python
tensor([[[0., 2.],
         [2., 0.],
         [0., 3.],
         [0., 0.],
         [4., 0.],
         [2., 0.],
         [2., 1.],
         [0., 3.]],

        [[1., 4.],
         [4., 0.],
         [3., 1.],
         [2., 0.],
         [2., 1.],
         [1., 0.],
         [4., 4.],
         [0., 4.]],

        [[1., 4.],
         [4., 4.],
         [4., 3.],
         [4., 3.],
         [4., 3.],
         [3., 0.],
         [4., 3.],
         [3., 3.]],

        [[4., 1.],
         [1., 3.],
         [1., 0.],
         [0., 3.],
         [4., 3.],
         [3., 1.],
         [1., 1.],
         [2., 1.]]])
```

Now consider a small inefficient loop to calculate the mean(s):


```python
# x bag-of-words, starts empty
# essentially ignoring the order, the information is important
xbow = torch.zeros((B, T, C))

for b in range(B):
    # for each record in B
    for t in range(T):
        # for each timestep, get average vectors
        # everything before + including current token
        xprev = x_dummy[b, : t + 1]
        xbow[b, t] = torch.mean(xprev, 0)
```

Looking at the output, lets check some of the work:

```python
tensor([[[0.0000, 2.0000],
         [1.0000, 1.0000],
         [0.6667, 1.6667],
         [0.5000, 1.2500],
         [1.2000, 1.0000],
         [1.3333, 0.8333],
         [1.4286, 0.8571],
         [1.2500, 1.1250]],

        [[1.0000, 4.0000],
         [2.5000, 2.0000],
         [2.6667, 1.6667],
         [2.5000, 1.2500],
         [2.4000, 1.2000],
         [2.1667, 1.0000],
         [2.4286, 1.4286],
         [2.1250, 1.7500]],

        [[1.0000, 4.0000],
         [2.5000, 4.0000],
         [3.0000, 3.6667],
         [3.2500, 3.5000],
         [3.4000, 3.4000],
         [3.3333, 2.8333],
         [3.4286, 2.8571],
         [3.3750, 2.8750]],

        [[4.0000, 1.0000],
         [2.5000, 2.0000],
         [2.0000, 1.3333],
         [1.5000, 1.7500],
         [2.0000, 2.0000],
         [2.1667, 1.8333],
         [2.0000, 1.7143],
         [2.0000, 1.6250]]])
```

Observations:

1. the first row stays the same, because there's nothing before it in terms of history
2. the decimal complexity further increases farther into the sequence because at step 2, the denomiator is 2, and at step 7, its 7

A quick deep dive:

```python
# x_dummy
tensor([[[0., 2.],
         [2., 0.],
         [0., 3.],
         ...

# xbow
tensor([[[0.0000, 2.0000],
         [1.0000, 1.0000],
         [0.6667, 1.6667], # <---- consider this row
         [0.5000, 1.2500],
         [1.2000, 1.0000],
         [1.3333, 0.8333],
         [1.4286, 0.8571],
         [1.2500, 1.1250]],
```

## About efficiency

As shown by the above, the loop is very inefficient, lets try and fix this with matrix multiplication

```python
import torch
torch.manual_seed(42)
a = torch.ones(3, 3)
b = torch.randint(0, 10, (3, 2), dtype=torch.float)
c = a @ b
print("a={}".format(a))
print("b={}".format(b))
print("c={}".format(c))
```

```
a=tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]])
b=tensor([[2., 7.],
        [6., 4.],
        [6., 5.]])
c=tensor([[14., 16.],
        [14., 16.],
        [14., 16.]])
```


Note that the differing dimensions of what is being multiplied:

```
[3 x 3] @ [3 x 2] = [3 x 2]
```

And this is out the math works

```
14 = 2 x 1 + 6 x 1 + 6 x 1
16 = 7 x 1 + 4 x 1 + 5 x 1
```

# lets a different pattern instead of all ones

This time we will us the `lower triangle` function or `tril` to get the following:

```python
torch.tril(torch.ones(3, 3))
```

```python
a = tensor([[1., 0., 0.],
            [1., 1., 0.],
            [1., 1., 1.]])
c = a @ b
```

The output of `a @ c` is as follows:

```
tensor([[ 2.,  7.],
        [ 8., 11.],
        [14., 16.]])
```

The first row is simply a copy, the second row is a rolling sum

``` 
8 = 2 + 6
11 = 7 + 4
```

## Converting the above into averages

```python
a = torch.tril(torch.ones(3, 3))

# normalizes the rolling sum
a = a / torch.sum(a, 1, keepdim=True)
```

notice that every row sums up to 1. Also think about this as "evenly spreading out attention"

```
tensor([[1.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000],
        [0.3333, 0.3333, 0.3333]])
```

Now consider the same `a @ c`:

```
tensor([[2.0000, 7.0000],
        [4.0000, 5.5000],
        [4.6667, 5.3333]])
```

The first row only has 1 set of values to look at, so its the same vals

```
2 = 2 x 1.
7 = 7 x 1.

4 = 2 x 0.5 + 6 x 0.5
5.5 = 7 x 0.5 + 4 x 0.5
```

## Back to weight calculation

Lets setup our averaging matrix for a larger size `T = 8`

```python
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
wei
```

```python
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000],
        [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000],
        [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.0000],
        [0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250]])
```

Using the above matrix to do rolling averages:

```python
# (T, T) @ (B, T, C) --> (B, T, C)
# will apply this by batch [(T, T) @ (T, C)]
xbow2 = wei @ xdummy

# compare the two
torch.allclose(xbow, xbow2)
# >>> true
```


## Using Softmax V3 for calculating average history

```python
## Softmax approach
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))

# using the triangle as a max
wei = wei.masked_fill(tril == 0, float('-inf'))

# take a softmax along every dimension
# softmax is a normalization operation
# softmax exponentiates each of the values + normalizes
# exp(0) = 1, exp(-inf) = 0
wei = F.softmax(wei, dim=-1)
```

```python
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000],
        [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000],
        [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.0000],
        [0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250]])
```