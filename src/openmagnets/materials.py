from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _vec3(value: np.ndarray | list[float] | tuple[float, float, float] | None) -> np.ndarray:
    if value is None:
        return np.zeros(3, dtype=float)
    return np.asarray(value, dtype=float).reshape(3)


@dataclass(slots=True)
class Material:
    name: str
    mu_r: float = 1.0
    current_density: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    magnetization: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))

    def __post_init__(self) -> None:
        self.mu_r = float(self.mu_r)
        if self.mu_r <= 0.0:
            raise ValueError("mu_r must be positive")
        self.current_density = _vec3(self.current_density)
        self.magnetization = _vec3(self.magnetization)
