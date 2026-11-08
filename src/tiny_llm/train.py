"""Train a model and save everything required for inference."""
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tiny_llm.config import TrainConfig
from tiny_llm.dataset import TextDataset
from tiny_llm.model import ModelConfig, TinyLLM
from tiny_llm.tokenizer import CharacterTokenizer


def train(config: TrainConfig) -> None:
    torch.manual_seed(config.seed)
    torch.set_num_threads(1)  # Small examples are faster and more predictable on CPU.
    text = Path(config.data_path).read_text(encoding="utf-8")
    split = int(len(text) * 0.9)
    if split <= config.block_size or len(text) - split <= config.block_size:
        raise ValueError("Both training and validation text must exceed block_size; use more text")
    # Build the vocabulary from both partitions so validation characters are representable.
    tokenizer = CharacterTokenizer.from_text(text)
    training = TextDataset(tokenizer.encode(text[:split]), config.block_size)
    validation = TextDataset(tokenizer.encode(text[split:]), config.block_size)
    loader = DataLoader(training, batch_size=config.batch_size, shuffle=True)
    validation_loader = DataLoader(validation, batch_size=config.batch_size)
    model_config = ModelConfig(len(tokenizer.characters), config.block_size, config.n_embd,
                               config.n_head, config.n_layer, config.dropout)
    model = TinyLLM(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    batches = iter(loader)
    for step in range(1, config.steps + 1):
        try:
            x, y = next(batches)
        except StopIteration:
            batches = iter(loader)
            x, y = next(batches)
        model.train()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, model_config.vocab_size), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % config.eval_interval == 0 or step == config.steps:
            model.eval()
            total_loss = 0.0
            count = 0
            with torch.no_grad():
                for vx, vy in validation_loader:
                    v_logits = model(vx)
                    total_loss += F.cross_entropy(v_logits.reshape(-1, model_config.vocab_size),
                                                  vy.reshape(-1), reduction="sum").item()
                    count += vy.numel()
            print(f"step {step}: train_loss={loss.item():.4f} val_loss={total_loss / count:.4f}")
    path = Path(config.output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "model_config": model.checkpoint_config(),
                "characters": tokenizer.characters, "model_state": model.state_dict()}, path)
    print(f"Saved checkpoint: {path}")
