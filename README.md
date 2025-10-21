# LLM

Using PyTorch to build an LLM

## Built the following components
- Byte-Pair Encoding Tokenizer (done in C++ to significantly speed up)
- Token Embedding
- Rotational Positional Embedding (based on RoFormer, Su et al. 2023)
- Multi-Headed Streaming Self-Attention (based on Attetion is All You Need, Vaswani et al. 2017)
- Add and Norm Layer

## Training
Hoping to train on data that was scraped from leetcode using https://github.com/pruett1/leetcode-solution-scraper

However, this might not be enough data and if so I will try to find publicly available free training set to use