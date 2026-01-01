from __future__ import annotations

import argparse
import os
import sys

# Allow running without installing package (dev convenience)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from lunarlander_ppo.training.train_loop import train
from lunarlander_ppo.utils.io import load_config


def main():
    parser = argparse.ArgumentParser(description="Train PPO on LunarLander-v3")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = train(cfg)
    print(f"\nRun complete. Outputs in: {run_dir}\n")


if __name__ == "__main__":
    main()
