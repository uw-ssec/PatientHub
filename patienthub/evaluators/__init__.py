from .rating import RatingEvaluator, RatingEvaluatorConfig
from .inspect import InspectEvaluator, InspectEvaluatorConfig

from omegaconf import DictConfig

EVALUATOR_REGISTRY = {
    "rating": RatingEvaluator,
    "inspect": InspectEvaluator,
}
EVALUATOR_CONFIG_REGISTRY = {
    "rating": RatingEvaluatorConfig,
    "inspect": InspectEvaluatorConfig,
}


def get_evaluator(configs: DictConfig, lang: str = "en"):
    configs.lang = lang
    eval_type = configs.eval_type
    print(f"Loading {eval_type} evaluator agent...")
    if eval_type in EVALUATOR_REGISTRY:
        return EVALUATOR_REGISTRY[eval_type](configs)
    else:
        raise ValueError(f"Evaluation for {eval_type} is not supported")


def register_evaluator_configs(cs):
    for name, config_cls in EVALUATOR_CONFIG_REGISTRY.items():
        cs.store(group="evaluator", name=name, node=config_cls)


__all__ = [
    "get_evaluator",
    "register_evaluator_configs",
]
