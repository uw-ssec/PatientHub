from typing import List
from .base import Dimension, Aspect
from dataclasses import dataclass, field


@dataclass
class Identification(Aspect):
    name: str = "identification"
    description: str = "The therapist's ability to identify core beliefs"
    guidelines: str = (
        "Look for instances where the therapist accurately identifies the client's cognitive distortions and emotional responses."
    )


@dataclass
class SocraticQuestioning(Aspect):
    name: str = "socratic_questioning"
    description: str = "Effectiveness of the therapist's Socratic questioning"
    guidelines: str = (
        "Assess how well the therapist uses Socratic questioning to guide the client towards self-discovery and cognitive restructuring."
    )


@dataclass
class HomeworkAssignment(Aspect):
    name: str = "homework_assignment"
    description: str = "Appropriateness of homework assignments given by the therapist"
    guidelines: str = (
        "Evaluate whether the homework assignments are relevant to the client's issues and promote skill development."
    )


@dataclass
class CognitiveRestructuring(Aspect):
    name: str = "cognitive_restructuring"
    description: str = (
        "Effectiveness of cognitive restructuring techniques used by the therapist"
    )
    guidelines: str = (
        "Look for how effectively the therapist helps the client challenge and change maladaptive thought patterns."
    )


@dataclass
class CBT(Dimension):
    name: str = "cbt"
    description: str = (
        "Evaluates the therapist's Cognitive Behavioral Therapy (CBT) skills"
    )
    target: str = "therapist"
    aspects: List[Aspect] = field(
        default_factory=lambda: [
            Identification(),
            SocraticQuestioning(),
            HomeworkAssignment(),
            CognitiveRestructuring(),
        ]
    )
