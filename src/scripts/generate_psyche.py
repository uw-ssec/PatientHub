import hydra
from omegaconf import DictConfig

from src.utils import append_json
from src.agents.generators.psyche import PsycheGenerator


@hydra.main(
    version_base=None, config_path="../configs/generators", config_name="psyche"
)
def generate_psyche_case(configs: DictConfig) -> None:
    generator = PsycheGenerator(configs=configs)

    diagnosis = configs.diagnosis
    age = configs.age
    sex = configs.sex

    case = generator.generate_case(diagnosis=diagnosis, age=age, sex=sex)
    append_json(case, configs.output_dir)


if __name__ == "__main__":
    generate_psyche_case()
