import pytest
import torch

from tiny_llm.dataset import TextDataset
from tiny_llm.model import ModelConfig, TinyLLM
from tiny_llm.tokenizer import CharacterTokenizer


def test_tokenizer_roundtrip_and_unknown():
    tokenizer = CharacterTokenizer.from_text("日本語の文章")
    assert tokenizer.decode(tokenizer.encode("日本語")) == "日本語"
    with pytest.raises(ValueError, match="not in training vocabulary"):
        tokenizer.encode("猫")
    with pytest.raises(ValueError, match="outside vocabulary"):
        tokenizer.decode([-1])


def test_dataset_shifts_targets():
    dataset = TextDataset([0, 1, 2, 3, 4], 3)
    x, y = dataset[0]
    assert x.tolist() == [0, 1, 2]
    assert y.tolist() == [1, 2, 3]
    assert len(dataset) == 2


def test_causal_attention_cannot_see_future():
    torch.manual_seed(1)
    model = TinyLLM(ModelConfig(8, 4, 8, 2, 2, 0.0)).eval()
    with torch.no_grad():
        first = model(torch.tensor([[1, 2, 3, 4]]))
        second = model(torch.tensor([[1, 2, 6, 7]]))
    assert first.shape == (1, 4, 8)
    torch.testing.assert_close(first[:, :2], second[:, :2])
