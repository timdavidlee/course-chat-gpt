"""python -m notes.b08b_train --gpu"""
import torch

from tqdm import tqdm

from .b01 import make_encoder_and_decoder, make_train_valid_data, load_input_text, estimate_loss, get_batch
from .b08 import BigramAttentionLanguageModelV1
from .util import make_arg_parser, print_banner


def main(use_gpu: bool):
    batch_size = 32
    block_size = 8
    learning_rate = 1e-3
    max_iters = 5000
    eval_iters = 500
    n_emb = 32
    device = "cuda" if use_gpu else "cpu"

    print_banner("using : {}".format(device))
    text = load_input_text()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    encoder, decoder = make_encoder_and_decoder(chars)
    train_data, valid_data = make_train_valid_data(text, encoder)

    m = BigramAttentionLanguageModelV1(vocab_size, n_emb=n_emb, block_size=block_size, device=device)
    m = m.to(device)

    optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)

    if next(m.parameters()).is_cuda:
        print_banner("model is on cuda!")

    optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)

    for iter in tqdm(range(max_iters)):
        if iter % eval_iters == 0 or iter == max_iters - 1:
            train_loss = estimate_loss(m, train_data, eval_iters, batch_size, block_size, device=device)
            valid_loss = estimate_loss(m, valid_data, eval_iters, batch_size, block_size, device=device)
            print("iter: {:,}\t| train_loss: {:.04f}\t| valid_loss: {:.04f}".format(iter, train_loss, valid_loss))

    
        xb, yb = get_batch(train_data, block_size, batch_size, device=device)

        _, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decoder(m.generate(context, max_new_tokens=500)[0].tolist()))


if __name__ == "__main__":
    parser = make_arg_parser()
    args = parser.parse_args()
    main(args.gpu)