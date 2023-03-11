# About Encoder Decoders

Everything that we have done is concerning `decoder`

The key difference here is the following comparison:

- the Queries come from the generative start (x)
- the Keys + and th Values come from the encoder network
- this is cross attention, in that all the phrases are available
- so when generating input, it is based on the past + the full availability of input
- this is called `conditioning`

# Nano GPT

Training will be much differenc e because of the following:

- adding checkpoints during model training
- training on multi-gpu
- pretrained weights
- decaying the learning rate
- more options