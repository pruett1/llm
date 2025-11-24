from helpers.json_to_s_exp import json_to_s_exp
from helpers.datasets import TokenDataset
from helpers.text_handler import jsonl_to_texts, csv_to_texts
from components.tokenizer_cpp import BlBPETokenizer

from model.transformer import Transformer
from model.train import train_model

import torch
from math import ceil
import time
import random
import gc
import os

def train_random_sample_texts(m_path: str, alt_texts: list[str], tokenizer: BlBPETokenizer, p: float = 0.01) -> None:
    print(f"Training tokenizer with random {p * 100}% sample of data...")
    start = time.time()

    sample = []
    with open(m_path, 'r', encoding='utf-8') as f:
        for line in f:
            if random.random() < p:
                sample.append(line.strip())

    sample_len = len(sample)
    for path in alt_texts:
        if path.endswith(".jsonl"):
            texts = jsonl_to_texts(path)
            sample.extend(texts)
        elif path.endswith(".csv"):
            texts = csv_to_texts(path, "output")
            if len(texts) > sample_len:
                texts = texts[:sample_len]
            sample.extend(texts)

    print(f"Loaded entire sample dataset of len {len(sample)}")
    
    tokenizer.train(sample)
    sample_len = len(sample)
    del sample
    gc.collect()

    end = time.time()
    print(f"Tokenizer trained with vocab size of {tokenizer.get_vocab_size()} on texts of length {sample_len} in {end - start:.2f} sec")

def encode_texts_to_token_data(in_path: str, out_path: str, tokenizer: BlBPETokenizer, skip_if_exists: bool = True):
    if os.path.exists(out_path) and skip_if_exists:
        print("Output file already exists, passing...")
        return

    start = time.time()
    vocab_ids = set(range(tokenizer.get_vocab_size()))
    max_len = total_len = count = 0
    with open(in_path, 'r', encoding='utf-8') as input_file, open(out_path, 'w', encoding='utf-8') as output_file:
        for line in input_file:
            line = line.strip()
            if not line:
                continue
            token_ids = tokenizer.encode(line)

            L = len(token_ids)
            if L > max_len:
                max_len = L
            total_len += L
            count += 1

            output_file.write(' '.join(map(str, token_ids)) + '\n')
    
    end = time.time()
    print(f"Encoded {count} texts in {end - start:.2f} sec to '{out_path}'. Max length: {max_len}, Average length: {total_len / count:.2f}")
    
def main():
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # conver json ASTs to s-expressions
    json_to_s_exp('corpuses/python100k_train.json', 'corpuses/pretrain_s_exp.txt', limit=-1)

    tokenizer = None

    if os.path.exists('checkpoints/tokenizer.bin'):
        print("loading tokenizer...")
        tokenizer = BlBPETokenizer.load('checkpoints/tokenizer.bin') 
        # print(tokenizer.curr_vocab_size())
    else:
        tokenizer = BlBPETokenizer(vocab_size=10000, special_tokens=["<|OUTPUT|>", "<|PAD|>", "<|DESC|>", "<|EXAMPLES|>", "<|CONSTRAINTS|>"])

    if tokenizer.curr_vocab_size() != tokenizer.get_vocab_size():
        train_random_sample_texts('corpuses/pretrain_s_exp.txt', ["corpuses/mbpp.jsonl", "corpuses/python_code_instruction.csv"], tokenizer, p=0.05)
        encode_texts_to_token_data('corpuses/pretrain_s_exp.txt', 'corpuses/pretrain_token_data.txt', tokenizer)

    model = Transformer(vocab_size = tokenizer.get_vocab_size(),
                        d_model = 512,
                        max_seq_len = 1024,
                        n_heads = 8,
                        n_layers = 8,
                        ff_mult = 4)
    
    token_data = TokenDataset('corpuses/pretrain_token_data.txt')
    print(len(token_data), "samples in token dataset")
    
    EPOCHS = 1000
    BATCH_SIZE = 32

    peak_lr = 1e-3
    steps_per_epoch = ceil(len(token_data) // BATCH_SIZE)
    WARMUP_STEPS = steps_per_epoch * (EPOCHS * 0.05) # 5% of total epochs for warmup
    LR = peak_lr * (512 ** -0.5)

    train_model(model, token_data, tokenizer, epochs=EPOCHS, lr=LR, warmup_steps=WARMUP_STEPS, batch_size=BATCH_SIZE, resume=True)
    model.save('models/pretrain.pt')

if __name__ == "__main__":
    main()