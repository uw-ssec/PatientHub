from typing import List
from .base import Dimension, Aspect
from dataclasses import dataclass, field


@dataclass
class ProfileFactual(Aspect):
    name: str = "profile_factual"
    description: str = "Factual consistency with the character profile"
    guidelines: str = "Check if stated facts (age, job, family) match the profile"


@dataclass
class ConvFactual(Aspect):
    name: str = "conv_factual"
    description: str = "Factual consistency within the conversation history"
    guidelines: str = "Check for self-contradictions across turns"


@dataclass
class Behavioral(Aspect):
    name: str = "behavioral"
    description: str = "Behavioral consistency between profile and responses"
    guidelines: str = "Check if actions/reactions match personality traits"


@dataclass
class Emotional(Aspect):
    name: str = "emotional"
    description: str = "Emotional consistency between profile and responses"
    guidelines: str = "Check if emotional expressions match the profile's affect"


@dataclass
class Consistency(Dimension):
    name: str = "consistency"
    description: str = "Evaluates whether the client's responses are consistent"
    target: str = "client"
    aspects: List[Aspect] = field(
        default_factory=lambda: [
            ProfileFactual(),
            ConvFactual(),
            Behavioral(),
            Emotional(),
        ]
    )
