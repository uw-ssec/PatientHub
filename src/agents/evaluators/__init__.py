from .client import get_client_evaluator
from .therapist import get_therapist_evaluator

from omegaconf import DictConfig


def get_evaluator(configs: DictConfig):
    agent_type = configs.agent_type
    if agent_type == "therapist":
        return get_therapist_evaluator(configs)
    elif agent_type == "client":
        return get_client_evaluator(configs)
    else:
        raise ValueError(f"Evaluation for {agent_type} is not supported")


__all__ = [
    "get_evaluator",
]
