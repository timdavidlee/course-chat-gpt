"""python -m notes.b08b_train"""
import torch

from .b02 import MyDataUtils
from .b03 import MyTorchDataset, DataLoader
from .b08 import BigramAttentionLanguageModelV1


def main():
    batch_size = 32
    block_size = 8
    learning_rate = 1e-3
    eval_iters = 200
    n_emb = 32

    my_data_utils = MyDataUtils("./notes/data/input.txt")
    vocab_size = my_data_utils.vocab_size
    token_ids = my_data_utils.get_tokens(my_data_utils.bigtext)
    
    train_cutoff = int(0.9 * len(token_ids))
    train_data = token_ids[: train_cutoff]
    valid_data = token_ids[train_cutoff:]

    train_dataset = MyTorchDataset(block_size, train_data)
    valid_dataset = MyTorchDataset(block_size, valid_data)

    train_dataloader = DataLoader(train_dataset, batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size)

    m = BigramAttentionLanguageModelV1(vocab_size, n_emb=n_emb, block_size=block_size)
    optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)


    last_loss = 999999
    train_dataloader = DataLoader(train_dataset, batch_size)
    for steps in range(1, 20_000):
        xb, yb = next(iter(train_dataloader))
        
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if steps % 500 == 0:
            print("steps: {:,} | {:.04f}".format(steps, loss.item()))
            imp = (last_loss - loss.item()) / last_loss
            if imp < 0.01:
                print("loss improvement: {}".format(imp))
                break
            else:
                last_loss = loss.item()

if __name__ == "__main__":
    main()
