"""Load a checkpoint and sample the next character repeatedly."""
import torch
from torch.nn import functional as F

from tiny_llm.model import ModelConfig, TinyLLM
from tiny_llm.tokenizer import CharacterTokenizer


def generate(checkpoint_path: str, prompt: str, length: int = 100,
             temperature: float = 1.0, seed: int = 42) -> str:
    if not prompt:
        raise ValueError("Prompt cannot be empty")
    if length < 0 or temperature <= 0:
        raise ValueError("length must be non-negative and temperature must be positive")
    torch.manual_seed(seed)
    torch.set_num_threads(1)  # Avoid thread overhead for tiny CPU inference.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    tokenizer = CharacterTokenizer(checkpoint["characters"])
    config = ModelConfig(**checkpoint["model_config"])
    if len(tokenizer.characters) != config.vocab_size:
        raise ValueError("Checkpoint vocabulary does not match model")
    model = TinyLLM(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    with torch.no_grad():
        for _ in range(length):
            logits = model(tokens[:, -config.block_size:])[:, -1, :] / temperature
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)
    return tokenizer.decode(tokens[0].tolist())
