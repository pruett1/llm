import pytest
import torch

from components.tokenizer import CharTokenizer, BlBPETokenizer
from components.token_embedding import TokenEmbedding
from components.positional_embedding import RoPositionalEmbedding
from components.tokenizer_cpp import BlBPETokenizer as CppBlBPETokenizer


def test_char_tokenizer_roundtrip():
    t = CharTokenizer()
    s = "hello"
    tokens = t.encode(s)
    assert isinstance(tokens, list)
    assert all(isinstance(x, int) for x in tokens)
    out = t.decode(tokens)
    assert out == s

def test_blbpe_tokenizer_basic_train_encode_decode():
    texts = ["hello world", "hello there"]
    b = BlBPETokenizer(vocab_size=260, special_tokens=["<|OUTPUT|>"])
    # train should not raise
    b.train(texts)
    encoded = b.encode("hello<|OUTPUT|>")
    assert 256 in encoded
    assert isinstance(encoded, list)
    assert all(isinstance(x, int) for x in encoded)
    decoded = b.decode(encoded)
    assert isinstance(decoded, str)
    assert decoded == "hello<|OUTPUT|>"

def test_cpp_blbpe_tokenizer_basic_train_encode_decode():
    texts = ["hello world", "hello there"]
    b = CppBlBPETokenizer(260, ["<|OUTPUT|>"])
    # train should not raise
    b.train(texts)
    encoded = b.encode("hello<|OUTPUT|>")
    assert 256 in encoded
    assert isinstance(encoded, (list))
    assert all(isinstance(x, int) for x in encoded)
    decoded = b.decode(encoded)
    assert isinstance(decoded, str)
    assert decoded == "hello<|OUTPUT|>"

def test_python_cpp_consistent():
    b_py = BlBPETokenizer(256, ["<|OUTPUT|>"])
    b_cpp = BlBPETokenizer(256, ["<|OUTPUT|>"])
    encoded_py = b_py.encode("hello<|OUTPUT|>")
    encoded_cpp = b_cpp.encode("hello<|OUTPUT|>")
    assert encoded_py == encoded_cpp
    assert b_py.decode(encoded_py) == b_cpp.decode(encoded_cpp)

def test_token_embedding_forward():
    vocab_size = 100
    d_model = 16
    embed = TokenEmbedding(vocab_size, d_model)
    tokens = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.long)
    out = embed(tokens)
    assert out.shape == (2,3,d_model)

def test_ro_positional_embedding_instantiation():
    p = RoPositionalEmbedding()
    assert p is not None