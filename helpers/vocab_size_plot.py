from .text_handler import jsonl_to_texts
from components.tokenizer import BlBPETokenizer
import matplotlib.pyplot as plt

def compression_ratio_avg(texts: list[str], tokenizer: BlBPETokenizer) -> float:
    return sum(len(t) / len(tokenizer.encode(t)) for t in texts) / len(texts)

def tokenizer_vocab_size_test(sample_text_path: str, sizes: list[int] = [300, 500, 1000, 2000, 5000, 10000, 20000]):
    texts = jsonl_to_texts(sample_text_path)
    ratios = []

    for vocab_size in sizes:
        print(f"Training tokenizer with vocab size {vocab_size}...")
        tokenizer = BlBPETokenizer(vocab_size=vocab_size, special_tokens=["<|OUTPUT|>"])
        tokenizer.train(texts)

        ratios.append(compression_ratio_avg(texts, tokenizer))

    print(sizes, ratios)

    plt.plot(sizes, ratios, marker='o')
    plt.xlabel('Vocab Size')
    plt.ylabel('Compression Ratio Avg. (Original Length / Tokenized Length)')
    plt.title('Tokenizer Compression Ratio vs Vocab Size')
    plt.show()

# tokenizer_vocab_size_test('../corpuses/easy_array_python_0.6_8_1755876977.jsonl')