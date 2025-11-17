from helpers.json_to_s_exp import json_to_s_exp
from helpers.file_dataset import StreamingFileDataset
from components.tokenizer_cpp import BlBPETokenizer

from model.transformer import Transformer
from model.train import train_model

import torch
from math import ceil
import time
import random
import gc

def train_random_sample_texts(path, tokenizer, p = 0.01) -> None:
    print(f"Training tokenizer with random {p * 100}% sample of data...")
    start = time.time()

    sample = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if random.random() < p:
                sample.append(line.strip())
    
    tokenizer.train(sample)
    del sample
    gc.collect()

    end = time.time()
    print(f"Tokenizer trained with vocab size of {tokenizer.get_vocab_size()} in {end - start:.2f} sec")

def encode_texts_to_token_data(in_path: str, out_path: str, tokenizer: BlBPETokenizer):
    start = time.time()
    vocab_ids = set(range(tokenizer.get_vocab_size()))
    max_len = total_len = count = 0
    split_tokens = 0
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
            split_tokens += sum(1 for tok in token_ids if tok not in vocab_ids)

            output_file.write(' '.join(map(str, token_ids)) + '\n')
    
    end = time.time()
    print(f"Encoded {count} texts in {end - start:.2f} sec to '{out_path}'. Max length: {max_len}, Average length: {total_len / count:.2f}")
    print(f"Coverage: {1 - total_len / split_tokens:.4f}")
    
def main():
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # conver json ASTs to s-expressions
    json_to_s_exp('corpuses/python100k_train.json', 'corpuses/pretrain_s_exp.txt', limit=1000)

    tokenizer = BlBPETokenizer(vocab_size=10000, special_tokens=["<|OUTPUT|>", "<|PAD|>", "<|DESC|>", "<|EXAMPLES|>", "<|CONSTRAINTS|>"])
    
    train_random_sample_texts('corpuses/pretrain_s_exp.txt', tokenizer, p=0.5)

    encode_texts_to_token_data('corpuses/pretrain_s_exp.txt', 'corpuses/pretrain_token_data.txt', tokenizer)
    

    model = Transformer(vocab_size = tokenizer.get_vocab_size(),
                        d_model = 512,
                        max_seq_len = 1024,
                        n_heads = 8,
                        n_layers = 8,
                        ff_mult = 4)
    
    EPOCHS = 1000
    BATCH_SIZE = 32

    peak_lr = 1e-3
    steps_per_epoch = ceil(100_000 // BATCH_SIZE)
    WARMUP_STEPS = steps_per_epoch * (EPOCHS * 0.05) # 5% of total epochs for warmup
    LR = peak_lr * (512 ** -0.5)

    token_data = StreamingFileDataset('corpuses/pretrain_token_data.txt', sample_frac = 0.1)

    train_model(model, token_data, tokenizer, epochs=EPOCHS, lr=LR, warmup_steps=WARMUP_STEPS, batch_size=BATCH_SIZE, resume=True)
    model.save('checkpoints/pretrain.pt')

if __name__ == "__main__":
    main()