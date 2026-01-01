import os
import shutil

from lunarlander_ppo.training.train_loop import train


def test_train_smoke(tmp_path):
    cfg = {
        "project_name": "test",
        "env_name": "LunarLander-v3",
        "seed": 0,
        "device": "cpu",
        "max_episodes": 2,
        "max_timesteps": 50,
        "eval_every": 1,
        "eval_episodes": 1,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "update_epochs": 1,
        "mini_batch_size": 16,
        "entropy_coef": 0.01,
        "value_loss_coef": 0.5,
        "max_grad_norm": 0.5,
        "hidden_dim": 32,
        "experiments_dir": str(tmp_path),
        "run_name": "smoke",
    }

    run_dir = train(cfg)
    assert os.path.isdir(run_dir)
    assert os.path.isfile(os.path.join(run_dir, "config.yaml"))
    assert os.path.isfile(os.path.join(run_dir, "train_metrics.csv"))
    assert os.path.isdir(os.path.join(run_dir, "models"))
    assert os.path.isdir(os.path.join(run_dir, "plots"))
