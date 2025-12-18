from typing import Any, Dict, List
from src.agents import InferenceAgent
from src.utils import load_json, load_prompts, get_chat_model
from omegaconf import DictConfig
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# This model does not support response formatting currently
class EeyoreClient(InferenceAgent):
    def __init__(self, configs: DictConfig):
        self.configs = configs

        self.data = load_json(configs.data_path)[configs.data_idx]
        self.name = self.data.get("name", "Client")
        self.profile = self.data.get("profile", {})

        self.chat_model = get_chat_model(configs)

        self.prompts = load_prompts(
            role="client", agent_type="eeyore", lang=configs.lang
        )

        system_content = self.prompts["system_prompt"].render(profile=self.profile)
        self.messages = [SystemMessage(content=system_content)]

    def set_therapist(
        self,
        therapist: Dict[str, Any],
        prev_sessions: List[Dict[str, str]] | None = None,
    ):
        self.therapist = therapist.get("name", "Therapist")

    def generate(self, messages: List[Any]):
        return self.chat_model.invoke(messages)

    def generate_response(self, msg: str):
        self.messages.append(HumanMessage(content=msg))
        res = self.generate(self.messages)
        self.messages.append(AIMessage(content=res.content))
        return res

    def reset(self):
        self.therapist = None
        system_content = self.prompts["system_prompt"].render(profile=self.profile)
        self.messages = [SystemMessage(content=system_content)]
