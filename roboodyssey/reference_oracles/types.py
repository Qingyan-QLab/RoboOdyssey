"""Lightweight types shared by optional reference algorithms."""
from typing import Literal

Point = tuple[float, float]
Status = Literal["ok", "invalid", "unreachable", "indeterminate"]
StateSource = Literal["planned", "runtime"]
