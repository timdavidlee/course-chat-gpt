import torch
from typing import Sequence


def read_in_shakespear_data(file: str) -> str:
    """returns a long string buffer"""
    with open(file, "r", encoding="utf-8") as f:
        return f.read()


class MyDataUtils:
    def __init__(self, text_file: str):
        bigtext = read_in_shakespear_data(text_file)
        print("total length of characters: {:,}".format(len(bigtext)))

        self.bigtext = bigtext
        self.unique_characters = sorted(list(set(bigtext)))
        self.vocab_size = len(self.unique_characters)
        print("".join(self.unique_characters))

        self.encoder = {char: j for j, char in enumerate(self.unique_characters)}
        self.decoder = {j: char for j, char in enumerate(self.unique_characters)}

    def encode(self, some_str: str) -> Sequence[int]:
        return [self.encoder[char] for char in some_str]

    def decode(self, sequence: Sequence[int]) -> str:
        chars = [self.decoder[num] for num in sequence]
        return "".join(chars)
    
    def get_tokens(self, text: str) -> torch.LongTensor:
        token_ids = torch.tensor(
            self.encode(text),
            dtype=torch.long,
        )
        print(token_ids.shape, token_ids.dtype)
        print(token_ids[:100])
        return token_ids 
