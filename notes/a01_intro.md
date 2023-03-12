# Class notes from Andrej Karpathy's Video Series on Chat GPT

### Links + Code

**Youtube link**: [https://www.youtube.com/watch?v=kCc8FmEb1nY](https://www.youtube.com/watch?v=kCc8FmEb1nY)

The code base is already in a repository called:

- For the more simplistic code with step-by-step [https://github.com/karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture)

- For the code-complete w/training additions : [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)

### What is a Language Model?

Models a sequence of words, and in its frame of reference it is completing a sequence of words.

### What is the neural network underlying chat GPT?

It comes from the paper `Attention is All you Need` paper from 2017. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

- Landmark paper that proposed the transformer architecture

### What does GPT stand for?

GPT stands for Generatively, Pretrained Transformer. Transformer is the neural network architecture. Transformers were used widely in the next few years. 

- GPT is trained on a large part of the internet
- GPT has a lot of fine tuning involved in it

###  What we will do together instead is make a character level model (will cover the differences later)

- Data source: tiny shakespeare
    - all of shakespear concatenated together
    - tensorflow: `https://www.tensorflow.org/datasets/catalog/tiny_shakespeare`
    - Model how all these characters follow each other

- problem setup:
    - given a chunk of these characters + some context in the past
    - will try and predict the next character that will come next
    - can generate infinite shakespeare-like language


## Contributing:

Easiest way to run the local site is:

```sh
mkdocs serve

# INFO     -  Documentation built in 0.89 seconds
# INFO     -  [20:53:27] Watching paths for changes: 'notes', 'mkdocs.yml'
# INFO     -  [20:53:27] Serving on http://127.0.0.1:8000/
```

Then to publish to github

```sh
mkdocs gh-deploy
```