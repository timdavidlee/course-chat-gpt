"""python -m notes.b11b_train --gpu"""

import torch

from datetime import datetime
from tqdm import tqdm

from .b01 import make_encoder_and_decoder, make_train_valid_data, load_input_text, estimate_loss, get_batch
from .b11 import BigramAttentionLanguageModelV4
from .util import make_arg_parser, print_banner


def main(use_gpu: bool):
    batch_size = 64
    block_size = 256
    max_iters = 5000
    eval_iters = 500
    learning_rate = 3e-4
    n_emb = 384
    device = "cuda" if use_gpu else "cpu"
    dropout = 0.2

    print_banner("using : {}".format(device))
    text = load_input_text()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    encoder, decoder = make_encoder_and_decoder(chars)
    train_data, valid_data = make_train_valid_data(text, encoder)

    m = BigramAttentionLanguageModelV4(
        vocab_size,
        n_emb=n_emb,
        block_size=block_size,
        dropout=dropout,
        device=device,
    )
    m = m.to(device)

    if next(m.parameters()).is_cuda:
        print_banner("model is on cuda!")

    optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)
    global_start = datetime.now()
    checkin_start = datetime.now()
    for iter in tqdm(range(max_iters)):
        if iter % eval_iters == 0 or iter == max_iters - 1:
            total_time = datetime.now() - global_start
            checkin_time = datetime.now() - checkin_start
            train_loss = estimate_loss(m, train_data, eval_iters, batch_size, block_size, device=device)
            valid_loss = estimate_loss(m, valid_data, eval_iters, batch_size, block_size, device=device)
            print("iter: {:,}\t| train_loss: {:.04f}\t| valid_loss: {:.04f} cycle: {} total_time: {}".format(
                iter, train_loss, valid_loss, checkin_time, total_time, 
            ))
            checkin_start = datetime.now()
    
        xb, yb = get_batch(train_data, block_size, batch_size, device=device)

        _, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    total_time = datetime.now() - global_start
    print("total runtime: {}".format(total_time))
    print_banner("Generative Output!")

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decoder(m.generate(context, max_new_tokens=500)[0].tolist()))


if __name__ == "__main__":
    arg_parser = make_arg_parser()
    args = arg_parser.parse_args()
    main(use_gpu=args.gpu)
