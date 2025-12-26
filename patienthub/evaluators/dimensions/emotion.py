from typing import List
from .base import Dimension, Aspect
from dataclasses import dataclass, field


@dataclass
class EmotionalDepth(Dimension):
    name: str = "emotional_depth"
    description: str = (
        "Evaluates the depth of emotional expression in the client's responses"
    )
    target: str = "client"
    aspects: List[Aspect] = field(default_factory=lambda: [])
