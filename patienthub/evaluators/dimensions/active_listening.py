from typing import List
from .base import Dimension, Aspect
from dataclasses import dataclass, field


@dataclass
class EmpatheticUnderstanding(Aspect):
    name: str = "empathetic_understanding"
    description: str = "Therapist's demonstration of empathic understanding"
    guidelines: str = (
        "Assess how well the therapist demonstrates empathy in their responses"
    )


@dataclass
class UnconditionalRegard(Aspect):
    name: str = "unconditional_regard"
    description: str = (
        "Therapist's acceptance and support of the client without judgment"
    )
    guidelines: str = "Evaluate how the therapist expresses acceptance without judgment"


@dataclass
class Congruence(Aspect):
    name: str = "congruence"
    description: str = "Therapist's authenticity and transparency in interactions"
    guidelines: str = "Look for signs of the therapist being genuine and transparent"


@dataclass
class ActiveListening(Dimension):
    name: str = "active_listening"
    description: str = "Evaluates the therapist's active listening skills"
    target: str = "therapist"
    aspects: List[Aspect] = field(
        default_factory=lambda: [
            EmpatheticUnderstanding(),
            UnconditionalRegard(),
            Congruence(),
        ]
    )
