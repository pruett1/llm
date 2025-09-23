import helpers.vocab_size_plot as vs_plot
import helpers.text_handler as th
from components.tokenizer import BlBPETokenizer
from components.tokenizer_cpp import BlBPETokenizer as CPPBlBPETokenizer
import time

def main():
    # vs_plot.tokenizer_vocab_size_test('corpuses/easy_array_python_0.6_8_1755876977.jsonl', 
    #                                   sizes=[1000, 2000, 3000, 4000, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 10000, 10500, 11000, 11500, 12000, 12500, 13000, 13500, 14000, 14500, 15000, 15500, 16000, 16500, 17000, 17500, 18000, 18500, 19000, 19500, 20000])
    texts = th.jsonl_to_texts('corpuses/easy_array_python_0.6_8_1755876977.jsonl')

    print("Benchmarking Python vs C++ Tokenizer...")
    print("Start python tokenizer...")
    start_py = time.time()
    tokenizer_py = BlBPETokenizer(vocab_size=1000, special_tokens=["<|OUTPUT|>"])
    tokenizer_py.train(texts)
    tokenizer_py_encode = tokenizer_py.encode(texts[0])
    tokenizer_py_decode = tokenizer_py.decode(tokenizer_py_encode)
    end_py = time.time()

    print("Start C++ tokenizer...")
    start_cpp = time.time()
    tokenizer_cpp = CPPBlBPETokenizer(vocab_size=1000, special_tokens=["<|OUTPUT|>"])
    tokenizer_cpp.train(texts)
    tokenizer_cpp_encode = tokenizer_cpp.encode(texts[0])
    tokenizer_cpp_decode = tokenizer_cpp.decode(tokenizer_cpp_encode)
    end_cpp = time.time()

    print(f"Python Tokenizer Time: {end_py - start_py:.4f} seconds")
    print(f"C++ Tokenizer Time: {end_cpp - start_cpp:.4f} seconds")
    print(tokenizer_py_encode == tokenizer_cpp_encode)
    print(tokenizer_py_decode == tokenizer_cpp_decode)


if __name__ == "__main__":
    main()