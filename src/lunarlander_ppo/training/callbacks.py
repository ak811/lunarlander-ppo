from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class BestModelTracker:
    """
    Tracks best reward and writes best checkpoint.
    """
    best_reward: float = float("-inf")
    best_path: Optional[str] = None

    def maybe_save(self, reward: float, save_dir: str, save_fn) -> bool:
        """
        save_fn(path) should save the model.
        Returns True if saved as new best.
        """
        if reward > self.best_reward:
            self.best_reward = reward
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, "best_model.pth")
            save_fn(path)
            self.best_path = path
            return True
        return False
