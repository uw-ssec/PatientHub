from typing import List
from .base import Dimension, Aspect
from dataclasses import dataclass, field


@dataclass
class Engagement(Aspect):
    name: str = "engagement"
    description: str = "Client's engagement level in therapy sessions"
    guidelines: str = "Assess how actively the client participates in sessions"


@dataclass
class Agreeableness(Aspect):
    name: str = "agreeableness"
    description: str = "Client's willingness to cooperate with the therapist"
    guidelines: str = "Evaluate how open the client is to suggestions and feedback"


@dataclass
class SelfCuring(Aspect):
    name: str = "self_curing"
    description: str = "Client's tendency to self-cure or avoid deep exploration"
    guidelines: str = (
        "Look for signs of the client downplaying issues or rushing to solutions"
    )


@dataclass
class Realism(Aspect):
    name: str = "realism"
    description: str = "Realism of the client's resistance behaviors"
    guidelines: str = "Compare the client's behaviors to those of a real human client"


@dataclass
class Resistance(Dimension):
    name: str = "resistance"
    description: str = "Evaluates the client's resistance to therapy"
    target: str = "client"
    aspects: List[Aspect] = field(
        default_factory=lambda: [
            Engagement(),
            Agreeableness(),
            SelfCuring(),
            Realism(),
        ]
    )
