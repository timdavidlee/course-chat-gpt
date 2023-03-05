# WTF is self-attention

Remember from before:

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

The above calculation results in an array like below:

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

### Think about a sentence, some words are more important than others

```
"From the market, I purchased some apples"
```

The more imporant words in the above sentence is `market` and `apples`.

While averaging is not a bad idea, most times, we don't want uniform consideration.


## How does self-attention solve this?

Every item in a sequence will emit 2 vectors:

`Query` - "the query vector", what am i looking for
`Key` - "what do i contain"

So how do the different items between the keys + the querys get affinity?

`Query [dot product] keys of all other tokens`, and that result will be come `wei`

If the affinity is high, the dot product result should be high.


### How to implement

Implement a single `head` of self-attention. There's something in this field called `head size`


```python
torch.manual_seed(1337)
B, T, C = 4, 8, 32 # batch , time, embedding channels
x = torch.randn(B, T, C)

# making a single head of self-attention
head_size = 16

# note that these values will be "learned" and are not constant
key = nn.Linear(C, head_size, bias=False)  # (B, T, 16)
query = nn.Linear(C, head_size, bias=False)  # (B, T, 16)

# so whats created here is a Key + Query for each of the inputs
# no communication has happened yet
k = key(x)  # (B, T, 16)
q = query(x)  # (B, T, 16)

# this wei will represent the affinities between the different possibilities
wei = q @ k.transpose(-2, -1)  # (B, T, 16) @ (B, 16, T) = (B, T, T)

# the above linear layers lets non-uniform emphasis to be made
# the below still enables the moving history
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)
out = wei @ x
```

If the non-uniform `wei` is examined:

```python
torch.set_printoptions(linewidth=200, threshold=100_000, sci_mode=False)
```

Should note that the results are not uniform anymore

```
tensor([[[1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
         [0.16, 0.84, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
         [0.21, 0.16, 0.63, 0.00, 0.00, 0.00, 0.00, 0.00],
         [0.58, 0.12, 0.19, 0.11, 0.00, 0.00, 0.00, 0.00],
         [0.03, 0.11, 0.05, 0.03, 0.79, 0.00, 0.00, 0.00],
         [0.02, 0.27, 0.02, 0.01, 0.68, 0.00, 0.00, 0.00],
         [0.17, 0.41, 0.04, 0.04, 0.10, 0.20, 0.03, 0.00],
         [0.02, 0.08, 0.06, 0.23, 0.06, 0.07, 0.24, 0.24]]
```

### What about `value`? 

So from above, there's a learnable distribution of "what external information might be useful?"

But note, that is not the same thing as retrieving the actual value. Collecting those "outside" values is established by the following:

```python
torch.manual_seed(1337)
B, T, C = 4, 8, 32 # batch , time, embedding channels
x = torch.randn(B, T, C)

# making a single head of self-attention
head_size = 16

key = nn.Linear(C, head_size, bias=False)  # (B, T, 16)
query = nn.Linear(C, head_size, bias=False)  # (B, T, 16)

# adding layer
value = nn.Linear(C, head_size, bias=False)  # (B, T, 16)

k = key(x)  # (B, T, 16)
q = query(x)  # (B, T, 16)

# this wei will represent the affinities between the different possibilities
wei = q @ k.transpose(-2, -1)  # (B, T, 16) @ (B, 16, T) = (B, T, T)

# the above linear layers lets non-uniform emphasis to be made
# the below still enables the moving history
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)

# the other elements we aggregate
v = value(x)
out = wei @ v
```

So consider `x` as `private` information about this token. 

So if im the 5th item in a series. 

- my private information is in `x`
- im interested in info stored in `q`
- i am offering information in `k`
- actual value that ill return in `v`

# Comments on Attention + Positional encoding

Attention is a communication mechanism, almost like a network. And attention gives a certain weight to information connected to our current location. A quick rehash:

```
A->B->C->D->E

A is connected to nothing (but itself)
B is connected to A
C is connected to B and A, etc.
And E is connected to all the previously letters
```

The the issue with the above is for letter `E`, there's no concept of "space", there's no difference to the location of A + B, it just knows that its connected to it.

### The batch dimension is isolated

Everything in the batch dimension should be isolated. Data across the batch dimension should not interact with one another.


### what if all the tokens need to talk to one another?

- Sentiment analysis
- fill in the blank (need to look before + after)
- Encoder block of self-attention, run the above, but remove this one line
    ```
    wei = wei.masked_fill(tril == 0, float("-inf"))
    ```

### What is cross attention

What we have described here is `self-attention` because the keys, queries, and values come from the same data. but they don't have to come from the same section

- Encoder <> Decoders example:
    - the queries are produced from X, but keys + values are produced from a separate section
    - when there's separate source of nodes

### Dividing by the headsize AKA scaled attention

The reason for denominator value (from the paper) is the following:

- softmax tends to go to one-hot encoding at extreme positive and negative values
- so dividing by the number of heads will curb some of that to ensure the calculated distribution is not so extreme

```
```