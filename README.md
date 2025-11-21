# LLM

Using PyTorch to build an LLM

## Built the following components
- Byte-Pair Encoding Tokenizer (done in C++ to significantly speed up)
- Token Embedding
- Rotational Positional Embedding (based on RoFormer, Su et al. 2023)
- Multi-Headed Streaming Self-Attention (based on Attetion is All You Need, Vaswani et al. 2017)
- Add and Norm Layer

## Training
Using https://www.kaggle.com/datasets/veeralakrishna/150k-python-dataset/data for pretraining

For fine tuning on python syntax I will be using the output tab of https://www.kaggle.com/datasets/thedevastator/python-code-instruction-dataset

Hoped to train on data that was scraped from leetcode using https://github.com/pruett1/leetcode-solution-scraper

This dataset proved to be too small so switched to using https://github.com/google-research/google-research/blob/master/mbpp/mbpp.jsonl, which was released as part of Austin et al. (2021), _Program synthesis with large language models_, arXiv:2108.07732