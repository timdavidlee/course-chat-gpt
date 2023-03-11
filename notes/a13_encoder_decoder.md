# About Encoder Decoders

Everything that we have done is concerning `decoder`

The key difference here is the following comparison:

- the Queries come from the generative start (x)
- the Keys + and th Values come from the encoder network
- this is cross attention, in that all the phrases are available
- so when generating input, it is based on the past + the full availability of input
- this is called `conditioning`

# Nano GPT

`Training` will be much differenc e because of the following:

- adding checkpoints during model training
- training on multi-gpu
- pretrained weights
- decaying the learning rate
- more options

The `model` should be very similar.

- What is different is the Multi-head attention. Instead of separating out two heads, make a 4D array instead.
- GeLU

# ChatGPT

What if we wanted to train ourselves?

- `Pretraining` stage: learn from the internet, and train it to "babble" about the internet. Massive infrastructure challenge. 1000's of GPUs to train this.
    - this trains a document completer. It will create articles + documents, trying to complete the sequence.
    - undefined behavior, try to complete some news article
- `2ndstage` condition on assistance
    - Part A: documents: `question + answer`, on the order of thousands of examples. Fine tune the model on documents like this. Sample efficient on fine-tuning
    - Part B: Let the model respond in different ways, than raters will select which answer they prefer, then will use this as a `reward` piece, predict using a different network, to guess which response would be preferred
    - Part C: then run `PPO` policy gradient reinforcement, to fine tune the answers, are expected to score a high reward. There is a whole fine tuning stage.
    - Much harder to replicate this stage

 
