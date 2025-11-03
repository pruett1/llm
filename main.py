from helpers.text_handler import jsonl_to_texts
from components.tokenizer_cpp import BlBPETokenizer

from model.transformer import Transformer
from model.train import train_model

import torch

def main():
    texts = jsonl_to_texts('./corpuses/easy_array_python_0.6_8_1755876977.jsonl')
    print("Loaded texts:", len(texts))

    tokenizer = BlBPETokenizer(vocab_size=10000, special_tokens=["<|OUTPUT|>", "<|PAD|>", "<|DESC|>", "<|EXAMPLES|>", "<|CONSTRAINTS|>"])
    print("Training tokenizer with vocab size:", tokenizer.get_vocab_size())
    tokenizer.train(texts)
    print("Tokenizer trained.")

    token_data = [tokenizer.encode(text) for text in texts]

    model = Transformer(vocab_size = tokenizer.get_vocab_size(),
                        d_model = 512,
                        max_seq_len = 128,
                        n_heads = 8,
                        n_layers = 6,
                        ff_mult = 4)

    train_model(model, token_data, tokenizer, epochs=5, lr=1e-4, batch_size=32)

    text = "<|DESC|>Write a function that returns the sum of two numbers.<|EXAMPLES|>Input: 2, 3 Output: 5 Input: -1, 1 Output: 0<|CONSTRAINTS|>The function should handle integer inputs.<|OUTPUT|>"
    input_ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long)
    generated_ids = model.generate(input_ids, max_length=50)
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    print(generated_text)

if __name__ == "__main__":
    main()