# Introduction

1. For any one prompt, any give any different examples
2. [LearnGPT.com]

### Language Model

Models a sequence of words, and in its frame of reference it is completing a sequence of words.

### What is the neural network underlying chat GPT

It comes from the paper `Attention is All you Need` paper from 2017. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

- Landmark paper that proposed the transformer architecture

### What does GPT stand for?

GPT stands for generatively, pretrained transformer. Transformer is the neural network. Transformers were used widely in the next few years. 

- GPT is trained on a large part of the internet
- GPT has a lot of fine tuning involved in it

###  What we will do together instead is make a cha racter level model

- tiny shakespeare
    - all of shakespear concatenated together
    - tensorflow: `https://www.tensorflow.org/datasets/catalog/tiny_shakespeare`
    - Model how all these characters follow each other

- problem setup:
    - given a chunk of these characters + some context in the past
    - will try and predict the next character that will come next
    - can generate infinite shakespeare-like language

### Where's the code?

The code base is already in a repository call `nano-gpt`: [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)

### Let's begin (in the next markdown)
