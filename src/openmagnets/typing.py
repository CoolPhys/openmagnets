from __future__ import annotations

from typing import Sequence, TypeAlias
import numpy as np

VectorLike: TypeAlias = Sequence[float] | np.ndarray
MatrixLike: TypeAlias = Sequence[Sequence[float]] | np.ndarray
ArrayLike: TypeAlias = Sequence[float] | Sequence[Sequence[float]] | np.ndarray
