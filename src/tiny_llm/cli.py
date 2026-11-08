"""Command-line interface for training and generation."""
import argparse

from tiny_llm.config import load_config
from tiny_llm.generate import generate
from tiny_llm.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and explore a tiny character language model")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--config", default="configs/tiny.yaml")
    generation = commands.add_parser("generate")
    generation.add_argument("--checkpoint", required=True)
    generation.add_argument("--prompt", required=True)
    generation.add_argument("--length", type=int, default=100)
    generation.add_argument("--temperature", type=float, default=1.0)
    generation.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        if args.command == "train":
            train(load_config(args.config))
        else:
            print(generate(args.checkpoint, args.prompt, args.length, args.temperature, args.seed))
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
