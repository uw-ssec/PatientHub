from .base import BaseAgent
from pydantic import BaseModel, Field
from prompts import get_prompts
from typing import Dict, List, Any
from brain import MentalState
from langchain_core.language_models import BaseChatModel


class ClientResponse(BaseModel):
    mental_state: Dict[str, MentalState] = Field(
        description="Your current mental state"
    )
    content: str = Field(
        description="The content of your generated response based on your profile and mental state"
    )


class SessionFeedback(BaseModel):
    identification: int = Field(
        "How well did the therapist identify your problems? (1-5)"
    )
    coherence: int = Field(
        "How coherent were the therapist's responses to your messages? (1-5)"
    )
    empathy: int = Field("How empathetic was the therapist? (1-5)")
    comforting: int = Field("How comforting were the therapist's responses? (1-5)")
    suggestions: int = Field("How helpful were the therapist's suggestions? (1-5)")
    overall: int = Field("Overall rating of the session (1-5)")


class BasicClient(BaseAgent):
    def __init__(self, model_client: BaseChatModel, data: Dict[str, Any]):
        self.role = "Client"
        self.agent_type = "basic"
        self.model_client = model_client
        self.data = data
        self.prompts = get_prompts(self.role)
        self.mental_state = MentalState()
        self.therapist = None
        self.messages = [
            {
                "role": "system",
                "content": self.prompts["profile"].render(data=self.data),
            }
        ]

    def generate(self, messages: List[str], response_format: BaseModel):
        model_client = self.model_client.with_structured_output(response_format)
        res = model_client.invoke(messages)
        return res

    def set_therapist(self, therapist, prev_sessions: List[Dict[str, str] | None] = []):
        self.therapist_data = therapist.data["demographics"]
        self.messages[0]["content"] += "\n" + self.prompts["therapy"].render(
            therapist=self.therapist,
            previous_sessions=prev_sessions,
        )

    def init_mental_state(self):
        # pt = self.prompts["mental_state"].render()
        # self.mental_state = self.generate(
        #     messages=self.messages + [{"role": "user", "content": pt}],
        #     response_format=MentalState,
        # )
        self.mental_state = MentalState(
            Emotion="Anxious",
            Beliefs="I need to meet everyone's expectations, but I'm not sure if I can handle it all.",
            Desires="To find a way to manage my workload without feeling overwhelmed.",
            Intents="Seeking strategies to balance work and personal life.",
            Trust_Level=0,
        )

    def generate_response(self, msg: str):
        if len(self.messages) == 1:
            self.messages[0]["content"] += "\n" + self.prompts["conversation"].render(
                data=self.data
            )

        self.messages.append({"role": "user", "content": msg})
        res = self.generate(self.messages, response_format=ClientResponse)
        self.messages.append({"role": "assistant", "content": res.model_dump_json()})

        return res

    def generate_feedback(self):
        pt = self.prompts["feedback"].render()
        feedback = self.generate(
            messages=self.messages + [{"role": "user", "content": pt}],
            response_format=SessionFeedback,
        )

        return feedback

    def reset(self):
        self.agent.reset()
