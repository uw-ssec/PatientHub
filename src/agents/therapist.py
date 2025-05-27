from .base import BaseAgent
from pydantic import BaseModel, Field
from prompts import get_prompt
from utils import parse_json_response, to_bullet_list
from langchain_core.output_parsers import PydanticOutputParser
from typing import Dict, List, Any


class MentalState(BaseModel):
    Emotion: str = Field(
        description="Client's current emotion based on Ekman's 8 emotions (label only)",
        default="Unknown",
    )
    Beliefs: str = Field(
        description="Client's current beliefs (1-2 sentences)", default="Unknown"
    )
    Desires: str = Field(
        description="Client's current desires (1-2 sentences)", default="Unknown"
    )
    Intents: str = Field(
        description="Client's current intentions (1-2) sentences", default="Unknown"
    )
    Trust_Level: int = Field(
        description="Client's current level of trust towards you (0-100)",
        default=0,
    )


class BaseTherapistResponse(BaseModel):
    reasoning: str = Field(
        description="reason about your perspective about client's thoughts and feelings in 3-5 sentences."
    )
    client_mental_state: Dict[str, MentalState] = Field(
        description="Current mental state of the client"
    )
    response: str = Field(
        description="Your generated response based on your reasoning and the client's mental state"
    )


class BasicTherapist(BaseAgent):
    def __init__(self, model, profile_data):
        super().__init__(role="therapist", model=model)
        self.parser = PydanticOutputParser(pydantic_object=BaseTherapistResponse)
        self.agent_name = "basic"
        self.client_mental_state = {
            "Emotion": "None",
            "Beliefs": "None",
            "Desires": "None",
            "Intents": "None",
            "Trust Level": 0,
        }
        self.profile_data = profile_data
        self.set_prompt()

    def set_prompt(self):
        self.sys_prompt = get_prompt(self.role, self.agent_name).render(
            data=self.profile_data,
            response_format=self.parser.get_format_instructions(),
            mode="conversation",
        )

        self.messages = [{"role": "system", "content": self.sys_prompt}]

    def receive_message(self, msg):
        self.messages.append(
            {
                "role": "user",
                "content": msg,
            }
        )

        return self.generate_response()

    def generate_response(self):
        res = self.model.invoke(self.messages)
        res = parse_json_response(res.content)
        self.client_mental_state = res["client_mental_state"]
        self.messages.append({"role": "assistant", "content": res["response"]})

        return (
            f"([{res['reasoning']}] -> {res['response']}",
            res["client_mental_state"],
        )
