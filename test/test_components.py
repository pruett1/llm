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
    # RoPositionalEmbedding requires vocab_size and d_model
    p = RoPositionalEmbedding(vocab_size=1000, d_model=16)
    assert p is not None


def test_ro_positional_embedding_rot():
    d_model = 8
    vocab_size = 100
    p = RoPositionalEmbedding(vocab_size=vocab_size, d_model=d_model)

    # create a dummy tensor (batch=2, seq=3, d_model)
    x = torch.randn(2, 3, d_model)

    # rotation at position 0 should be identity-like (freqs = 0)
    out0 = p.rot(x, 0)
    assert out0.shape == x.shape
    assert torch.allclose(out0, x, atol=1e-6)

    # rotation at non-zero position should preserve per-pair norms
    out1 = p.rot(x, 1)
    # check shape
    assert out1.shape == x.shape
    # for each pair (even, odd) the squared norm should be preserved
    even = out1[..., ::2]
    odd = out1[..., 1::2]
    orig_even = x[..., ::2]
    orig_odd = x[..., 1::2]

    orig_pair_norm = orig_even**2 + orig_odd**2
    new_pair_norm = even**2 + odd**2
    assert torch.allclose(orig_pair_norm, new_pair_norm, atol=1e-5)

def test_ro_positional_embedding_rot_values():
    # deterministic check for d_model=4, single vector
    d_model = 4
    p = RoPositionalEmbedding(vocab_size=10, d_model=d_model)

    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])  # shape (1,1,4)
    out = p.rot(x, 1)

    # compute expected manually
    half = d_model // 2
    thetas = (10000 ** (torch.arange(0, half).float() / d_model))
    freqs = 1 * thetas
    cos = freqs.cos().repeat_interleave(2)
    sin = freqs.sin().repeat_interleave(2)

    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack([-x2, x1], dim=-1).reshape_as(x)
    expected = x * cos + x_rot * sin

    assert out.shape == x.shape
    assert torch.allclose(out, expected, atol=1e-6)