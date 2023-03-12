# Text Basics

How a machine understands text

### Requirements

- `pytorch`
- `mlp`
- `python`


### Getting started with transformers

There's a google colab that can be used as reference: [Colab Link](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbVRONkNLYTFXY0kxUXhKcVlHUGdid2pocTFMZ3xBQ3Jtc0tsWTM4LXNjXzJicG9MSVRoOHJiTUFwRGNUQWktZGlQRHV6UHQ5WE1DRWRQWnR1OU5MdEFTVzA0ZWFteXdlUVNZMjRidDlURXBhTjY2TVFnZDRpWm54SEZPRzFLd0JpVWlrb3MxdkZjblBWQThYdUJFMA&q=https%3A%2F%2Fcolab.research.google.com%2Fdrive%2F1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-%3Fusp%3Dsharing&v=kCc8FmEb1nY)


### Get the data

```sh
!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

### Opening the data

Open the text file as a really long single string variable

```python
def read_in_shakespear_data(file: str) -> str:
    """reads the simple text file from disk as a long string"""
    with open(file, "r", encoding="utf-8") as f:
        return f.read()

file = "./notes/data/input.txt"
bigtext = read_in_shakespear_data(file)
```

Long is our source text?

```python
print("total length of characters: {:,}".format(len(bigtext)))
# >>> total length of characters: 1,115,394
```

What is a sample of the text?

```python
print(bigtext[:1000])
```

```
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.
```


What are the unique characters found in the text?

```python
unique_characters = sorted(list(set(bigtext)))
print("".join(unique_characters))
```

```
>>>  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
```

Should also note that `newline` or `\n` is also one of those characters, but cannot be visually seen because of the `print`

How many unique characters are there? 

```python
print(len(unique_characters))
>>> 65
```

So for the task we are developing, at any given time there are 65 possible choices to select for the next word

### Tokenizing the data

`tokenizing` refers to converting text into some series of numbers. There's mulitiple methods, libraries + approaches for doing this conversion. 

- tokenizing = translating text -> numbers

Our approach here, will be to assign numbers to each unique charater. And then any word can be converted to a series of integers.

```python
from typing import Sequence

encoder = {char: j for j, char in enumerate(unique_characters)}
decoder = {j: char for j, char in enumerate(unique_characters)}


def encode(some_str: str) -> Sequence[int]:
    return [encoder[char] for char in some_str]


def decode(sequence: Sequence[int]) -> str:
    chars = [decoder[num] for num in sequence]
    return "".join(chars)
```

An example of encoding:

```python
encode("ABCabc")

>>> [13, 14, 15, 39, 40, 41]
```

A couple of observations:
- since the encoder is sorted in abc order, similar characters are only a 1 step away
- **uppercase** vs. **lowercase** will be treated differently in this model


An example of decoding:

```python
decode([0, 1, 2, 3])
```

```
>>> '\n !$'
```

A few other examples

```python
encode("my name is bob")

>>> [51, 63, 1, 52, 39, 51, 43, 1, 47, 57, 1, 40, 53, 40]
```

```python
decode([51, 63, 1, 52, 39, 51, 43, 1, 47, 57, 1, 40, 53, 40])

>>> 'my name is bob'
```
 
### Other approaches

`SentencePiece` that google uses a different vocabulary and a different approach. This is a `sub-word` tokenizer.

#### What does sub-word mean?

It means you are not encoding whole words, but at the same time not encoding characters either. `Subword` is pretty typical in practice. A quick example:

```python
orignal = "rerunning"
subwork = "re#", "run", "#ing"
```

The above is an example of breaking up a word into `prefix`, `suffix`, and core `word` parts.


Google's Encoder: [Sentence Piece Github](https://github.com/google/sentencepiece)

OpenAI's Encoder: [TikToken Github](https://github.com/openai/tiktoken)

#### Example from OpenAI's tokenizer:

```python
import tiktoken

encoder = tiktoken.get_encoding("gpt2")
encoder.encode("my name is bob")
# >>> [1820, 1438, 318, 29202]

print(encoder.n_vocab)
# >>> 50257
```

Random:
- its funny that `bob` has its own token, but maybe thats more for "something the water **bobs** up and down"
- the vocabulary is a LOT larger


## Tokenize the entire dataset:

```python
import torch

def tokenize_shakespear(bigtext: str) -> Sequence[int]:
    token_ids = torch.tensor(
        encode(bigtext),
        dtype=torch.long,
    )
    return token_ids


token_ids = tokenize_shakespear(bigtext)
print(token_ids.shape, token_ids.dtype)
print(token_ids[:100])
```

A few general comments:

`encode(text)` - encodes the text into integers
`torch.tensor(encode(text))` - then creates a large `long` 1D array to store the data and the data. FYI `long` is another way of saying `big integer`.


```
torch.Size([1115394]) torch.int64
tensor([18, 47, 56, 57, 58,  1, 15, 47, 58, 47, 64, 43, 52, 10,  0, 14, 43, 44,
        53, 56, 43,  1, 61, 43,  1, 54, 56, 53, 41, 43, 43, 42,  1, 39, 52, 63,
         1, 44, 59, 56, 58, 46, 43, 56,  6,  1, 46, 43, 39, 56,  1, 51, 43,  1,
        57, 54, 43, 39, 49,  8,  0,  0, 13, 50, 50, 10,  0, 31, 54, 43, 39, 49,
         6,  1, 57, 54, 43, 39, 49,  8,  0,  0, 18, 47, 56, 57, 58,  1, 15, 47,
        58, 47, 64, 43, 52, 10,  0, 37, 53, 59])
```


## Splitting data into train / and val

Will split our long sequence into train + val. 

`train` will be what the model sees
`valid` will be what is used to see how the model does on new material.

The ratio can depend on the size, but `80 / 20` or `90 / 10` is pretty common for smaller datasets.

```python
train_cutoff = int(0.9 * len(token_ids))
train_data = token_ids[:train_cutoff]
vali_date = token_ids[valid_cutoff:]
```





