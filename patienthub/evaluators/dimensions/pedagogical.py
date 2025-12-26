from typing import List
from .base import Dimension, Aspect
from dataclasses import dataclass, field


@dataclass
class Cues(Aspect):
    name: str = "cues"
    description: str = "Identification of pedagogical cues in the conversation"
    guidelines: str = (
        "Look for instances where the client displays symptoms and cues from their profile (e.g., emotional expressions, behavioral patterns)."
    )


@dataclass
class Feedback(Aspect):
    name: str = "feedback"
    description: str = "Usefulness of the client's feedback to the therapist"
    guidelines: str = (
        "Assess how the client's feedback can help the therapist improve their skills (e.g., clarity, relevance)."
    )


@dataclass
class PedagogicalValue(Dimension):
    name: str = "pedagogical_value"
    description: str = "Evaluates the pedagogical value of the client's responses"
    target: str = "client"
    aspects: List[Aspect] = field(
        default_factory=lambda: [
            Cues(),
            Feedback(),
        ]
    )
