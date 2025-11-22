import random
from typing import Any, Dict, List

from agents import BaseAgent
from pydantic import BaseModel, Field
from utils import load_json, load_prompts
from utils.eeyore_local import EeyoreLocalModel
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class Response(BaseModel):
    content: str = Field(
        description="The content of your generated response in this turn",
    )


class EeyoreClient(BaseAgent):
    def __init__(
        self,
        model_client: BaseChatModel,
        data: Dict[str, Any] | None = None,
        lang: str = "en",
    ):
        self.role = "client"
        self.agent_type = "eeyore"
        self.lang = lang
        self.data = data or {}
        self.name = self.data.get("name", "Eeyore Client")

        self.local_model = EeyoreLocalModel.from_env_or_default()

        self.prompts = load_prompts(
            role=self.role,
            agent_type=self.agent_type,
            lang=self.lang,
        )

        self._init_profile_and_system_message()

    def _init_profile_and_system_message(self):
        profiles = load_json("data/characters/Eeyore Profile Cognitive Model.json")
        profile_entry = random.choice(profiles)
        self.profile = profile_entry.get("profile", {})

        system_content = self.prompts["system_prompt"].render(profile=self.profile)
        self.messages = [SystemMessage(content=system_content)]

    def generate(self, messages: List[Any], response_format: type[BaseModel]):
        chat_messages: List[Dict[str, str]] = []
        for m in messages:
            if isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, HumanMessage):
                role = "user"
            else:
                role = "assistant"
            chat_messages.append({"role": role, "content": m.content})

        text = self.local_model.generate(chat_messages)
        return response_format(content=text)

    def set_therapist(
        self,
        therapist: Dict[str, Any] | Any,
        prev_sessions: List[Dict[str, str]] | None = None,
    ):
        if isinstance(therapist, dict):
            self.therapist = therapist.get("name", "Therapist")
        else:
            self.therapist = getattr(therapist, "name", "Therapist")

    def generate_response(self, msg: str):
        self.messages.append(HumanMessage(content=msg))
        res = self.generate(self.messages, response_format=Response)
        self.messages.append(AIMessage(content=res.content))
        return res

    def reset(self):
        self.therapist = None
        self._init_profile_and_system_message()
