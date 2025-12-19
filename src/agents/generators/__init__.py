from .agent_files import AgentFilesGenerator
from .student import StudentClientGenerator
from .psyche import PsycheGenerator
from .clientCast import ClientCastGenerator

from omegaconf import DictConfig

GENERATORS = {
    "agent_files": AgentFilesGenerator,
    "student": StudentClientGenerator,
    "psyche": PsycheGenerator,
    "clientCast": ClientCastGenerator,
}


def get_generator(configs: DictConfig):
    agent_type = configs.generator.agent_type
    print(f"Loading {agent_type} generator...")
    if agent_type in GENERATORS:
        return GENERATORS[agent_type](configs=configs)
    else:
        raise ValueError(f"Unknown generator type: {agent_type}")


__all__ = [
    "get_generator",
    "AgentFilesGenerator",
    "StudentClientGenerator",
    "PsycheGenerator",
    "ClientCastGenerator",
]
