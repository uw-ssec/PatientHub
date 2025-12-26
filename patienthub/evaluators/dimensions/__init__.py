from .base import Dimension, Aspect
from .consistency import Consistency
from .resistance import Resistance
from .pedagogical import PedagogicalValue
from .emotion import EmotionalDepth


DIMENSION_REGISTRY = {
    "consistency": Consistency,
    "resistance": Resistance,
    "pedagogical_value": PedagogicalValue,
    "emotional_depth": EmotionalDepth,
}


def get_dimensions(dimensions):
    return [DIMENSION_REGISTRY[name]() for name in dimensions]


__all__ = ["Dimension", "Aspect", "get_dimensions"]
