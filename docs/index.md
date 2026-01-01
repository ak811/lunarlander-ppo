# LunarLander PPO

This repository implements Proximal Policy Optimization (PPO) for `LunarLander-v3` using Gymnasium + PyTorch.

- Training entry: `python scripts/train.py --config configs/default.yaml`
- Evaluation: `python scripts/evaluate.py --config configs/default.yaml --checkpoint <path>`
- Video: `python scripts/record_video.py --config configs/default.yaml --checkpoint <path>`
- Sweep: `python scripts/sweep.py --sweep configs/sweep.yaml`
