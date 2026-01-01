from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunningMean:
    """
    Tiny helper for smoothing.
    """
    alpha: float = 0.1
    value: float = 0.0
    initialized: bool = False

    def update(self, x: float) -> float:
        if not self.initialized:
            self.value = x
            self.initialized = True
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value
