"""T2: skills and primitives. Every skill implements contracts.Skill."""

from .loader import load_skill
from .pick_place import PickPlace

__all__ = ["PickPlace", "load_skill"]
