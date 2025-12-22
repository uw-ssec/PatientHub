from hydra.core.config_store import ConfigStore
from src.clients import register_client_configs
from src.therapists import register_therapist_configs
from src.evaluators import register_evaluator_configs
from src.generators import register_generator_configs


def register_configs(config_name: str, config_class):
    cs = ConfigStore.instance()
    cs.store(name=config_name, node=config_class)

    # Register configs
    register_client_configs(cs)
    register_therapist_configs(cs)
    register_evaluator_configs(cs)
    register_generator_configs(cs)
