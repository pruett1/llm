import helpers.vocab_size_plot as vs_plot

def main():
    vs_plot.tokenizer_vocab_size_test('corpuses/easy_array_python_0.6_8_1755876977.jsonl', 
                                      sizes=[1000, 2000, 3000, 4000, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 10000, 10500, 11000, 11500, 12000, 12500, 13000, 13500, 14000, 14500, 15000, 15500, 16000, 16500, 17000, 17500, 18000, 18500, 19000, 19500, 20000])

if __name__ == "__main__":
    main()