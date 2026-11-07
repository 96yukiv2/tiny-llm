import subprocess
import sys

import yaml


def test_cli_train_save_and_generate(tmp_path):
    data = tmp_path / "sample.txt"
    data.write_text("こんにちは世界。\n" * 30, encoding="utf-8")
    checkpoint = tmp_path / "model.pt"
    config = {"data_path": str(data), "output_path": str(checkpoint), "seed": 42,
              "block_size": 4, "n_embd": 8, "n_head": 2, "n_layer": 1, "dropout": 0.0,
              "batch_size": 4, "steps": 2, "learning_rate": 0.001, "eval_interval": 1}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = subprocess.run([sys.executable, "-m", "tiny_llm.cli", "train", "--config",
                             str(config_path)], check=True, text=True, capture_output=True)
    assert "val_loss=" in result.stdout
    assert checkpoint.is_file()
    result = subprocess.run([sys.executable, "-m", "tiny_llm.cli", "generate", "--checkpoint",
                             str(checkpoint), "--prompt", "こんにちは", "--length", "4"],
                            check=True, text=True, capture_output=True)
    assert result.stdout.startswith("こんにちは")
    assert len(result.stdout.strip()) >= len("こんにちは") + 3  # Generated text may contain newlines.
