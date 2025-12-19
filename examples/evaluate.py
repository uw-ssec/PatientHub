"""
An example for evaluation.
It requires:
    - An evaluator agent
    - Data (e.g., conversation history, profile data)
"""

import hydra
from omegaconf import DictConfig

from src.utils import load_json, save_json
from src.agents import get_inference_agent


@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def evaluate(configs: DictConfig):
    evaluator = get_inference_agent(category="evaluator", configs=configs)
    conv_data = load_json(configs.input_dir)
    res = evaluator.evaluate(data=conv_data)

    save_json(res, configs.output_dir, overwrite=True)


if __name__ == "__main__":
    evaluate()
