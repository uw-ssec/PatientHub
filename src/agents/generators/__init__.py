from .student import StudentClientGenerator
from .psyche import PsycheGenerator
from .agent_files import AgentFilesGenerator

from omegaconf import DictConfig

GENERATORS = {
    "student": StudentClientGenerator,
    "psyche": PsycheGenerator,
    "agent_files": AgentFilesGenerator,
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
    "StudentClientGenerator",
    "PsycheGenerator",
    "AgentFilesGenerator",
]
