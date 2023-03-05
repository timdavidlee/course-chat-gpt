"""python -m notes.b10b_train"""
import torch

from tqdm import tqdm

from .b01 import make_encoder_and_decoder, make_train_valid_data, load_input_text, estimate_loss, get_batch
from .b10 import BigramAttentionLanguageModelV3


def main():
    batch_size = 32
    block_size = 8
    learning_rate = 1e-3
    max_iters = 5000
    eval_iters = 200
    n_emb = 32
    device="cpu"

    text = load_input_text()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    encoder, decoder = make_encoder_and_decoder(chars)
    train_data, valid_data = make_train_valid_data(text, encoder)

    m = BigramAttentionLanguageModelV3(vocab_size, n_emb=n_emb, block_size=block_size)
    optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)

    for iter in tqdm(range(max_iters)):
        if iter % eval_iters == 0 or iter == max_iters - 1:
            train_loss = estimate_loss(m, train_data, eval_iters, batch_size, block_size)
            valid_loss = estimate_loss(m, valid_data, eval_iters, batch_size, block_size)
            print("iter: {:,}\t| train_loss: {:.04f}\t| valid_loss: {:.04f}".format(iter, train_loss, valid_loss))

    
        xb, yb = get_batch(train_data, block_size, batch_size)

        _, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decoder(m.generate(context, max_new_tokens=500)[0].tolist()))


if __name__ == "__main__":
    main()
