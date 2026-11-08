"""A decoder-only Transformer with causal self-attention."""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float

    def __post_init__(self):
        if min(self.vocab_size, self.block_size, self.n_embd, self.n_head, self.n_layer) <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be between 0 and 1")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.projection = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(batch, length, self.n_head, channels // self.n_head).transpose(1, 2)
                   for t in (q, k, v))
        scores = q @ k.transpose(-2, -1) * (q.size(-1) ** -0.5)
        scores = scores.masked_fill(self.mask[:, :, :length, :length] == 0, float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        result = (weights @ v).transpose(1, 2).contiguous().view(batch, length, channels)
        return self.dropout(self.projection(result))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.n_embd)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd), nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd), nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


class TinyLLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.n_layer)))
        self.norm = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.size(1) == 0 or tokens.size(1) > self.config.block_size:
            raise ValueError("Expected [batch, sequence] with 0 < sequence <= block_size")
        positions = torch.arange(tokens.size(1), device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        return self.head(self.norm(self.blocks(x)))

    def checkpoint_config(self) -> dict:
        return asdict(self.config)
