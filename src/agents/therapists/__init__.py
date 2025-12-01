from .basic import BasicTherapist
from .eliza import ElizaTherapist
from .user import UserTherapist

from omegaconf import DictConfig


def get_therapist(configs: DictConfig):
    agent_type = configs.agent_type
    print(f"Loading {agent_type} therapist agent...")
    if agent_type == "basic":
        return BasicTherapist(configs=configs)
    elif agent_type == "eliza":
        return ElizaTherapist(configs=configs)
    elif agent_type == "user":
        # name = input("Enter your name: ")
        return UserTherapist(configs=configs)
    else:
        raise ValueError(f"Unknown therapist agent type: {agent_type}")


__all__ = [
    "BasicTherapist",
    "ElizaTherapist",
    "UserTherapist",
]
