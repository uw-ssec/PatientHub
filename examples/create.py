"""
A script for creating new agents.
It requires:
    - agent_type: either "client" or "therapist"
    - agent_name: the name of the new agent to be created
It creates:
- The agent implementation at `src/agents/{agent_type}s/{agent_name}.py`.
- The configuration file at `configs/{agent_type}s/{agent_name}.yaml`.
- A prompt template file at `data/prompts/{agent_type}s/{agent_name}.yaml`.
It also adds the new agent to the corresponding `__init__.py` file.
The configurations for the new agent are specified in the `configs/create.yaml` file.
"""

import hydra
from omegaconf import DictConfig

from src.agents import get_inference_agent


@hydra.main(version_base=None, config_path="../configs", config_name="create")
def main(configs: DictConfig) -> None:
    file_creator = get_inference_agent(category="generator", configs=configs)
    if configs.gen_agent_type not in ["client", "therapist"]:
        raise ValueError(
            "Can only generate files for 'client' or 'therapist' agents for now..."
        )
    file_creator.generate_files()


if __name__ == "__main__":
    main()
