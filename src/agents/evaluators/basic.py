from typing import Dict, List
from src.agents import InferenceAgent
from src.utils import load_prompts, get_chat_model
from pydantic import BaseModel
from omegaconf import DictConfig
from langchain_core.language_models import BaseChatModel


class BasicEvaluator(InferenceAgent):
    def __init__(self, configs: DictConfig, feedback_schema: BaseModel):
        self.chat_model = get_chat_model(configs)
        self.prompt = load_prompts(
            role="evaluator", agent_type="basic", lang=configs.lang
        )["prompt"]

    def generate(self, messages: List[Dict]):
        # messages = [f"{msg['role']}: {msg['content']}" for msg in messages]
        # chat_model = self.chat_model.with_structured_output(self.feedback_schema)
        # self.prompt = self.prompt.render(dialogue_history="\n".join(messages))
        # res = chat_model.invoke(self.prompt)

        # manually write res
        res = {"content": "Good job!"}

        return res

    def reset(self):
        pass
