from typing import Dict, List
from src.agents import InferenceAgent
from src.utils import load_prompts, get_model_client
from pydantic import BaseModel
from omegaconf import DictConfig
from langchain_core.language_models import BaseChatModel


class BasicEvaluator(InferenceAgent):
    def __init__(self, configs: DictConfig, feedback_schema: BaseModel):
        self.model_client = get_model_client(configs)
        self.prompt = load_prompts(
            role="evaluator", agent_type="basic", lang=configs.lang
        )["prompt"]

    def generate(self, messages: List[Dict]):
        # messages = [f"{msg['role']}: {msg['content']}" for msg in messages]
        # model_client = self.model_client.with_structured_output(self.feedback_schema)
        # self.prompt = self.prompt.render(dialogue_history="\n".join(messages))
        # res = model_client.invoke(self.prompt)

        # manually write res
        res = {"content": "Good job!"}

        return res

    def reset(self):
        pass
