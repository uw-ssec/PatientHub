from .base import BaseAgent
from pydantic import BaseModel, Field
from prompts import get_prompt
from utils import parse_json_response, to_bullet_list
from langchain_core.output_parsers import PydanticOutputParser
from typing import Dict, List, Any


class MentalState(BaseModel):
    Emotion: str = Field(
        description="You current emotion based on Ekman's 8 emotions (label only)",
        default="Unknown",
    )
    Beliefs: str = Field(
        description="Your current beliefs (1-2 sentences)", default="Unknown"
    )
    Desires: str = Field(
        description="Your current desires (1-2 sentences)", default="Unknown"
    )
    Intents: str = Field(
        description="Your current intentions (1-2 sentences)", default="Unknown"
    )
    Trust_Level: int = Field(
        description="Your current level of trust towards the therapist (0-100)",
        default=0,
    )


class BasicClientResponse(BaseModel):
    mental_state: Dict[str, MentalState] = Field(
        description="Current mental state of the client"
    )
    response: str = Field(
        description="Your generated response based on your mental state"
    )


class BasicClient(BaseAgent):
    def __init__(self, model, profile_data):
        super().__init__(role="client", model=model)
        self.agent_name = "basic"
        self.mental_state = {
            "Emotion": "Unknown",
            "Beliefs": "Unknown",
            "Desires": "Unknown",
            "Intents": "Unknown",
            "Trust Level": 0,
        }
        self.profile_data = profile_data
        self.sys_prompt = get_prompt(self.role, self.agent_name)
        # self.init_mental_state(self.profile_data)
        self.set_prompt()

    def set_prompt(self, mode: str = "mental state"):
        self.parser = PydanticOutputParser(pydantic_object=BasicClientResponse)
        self.sys_prompt = self.sys_prompt.render(
            data=self.profile_data,
            response_format=self.parser.get_format_instructions(),
            # mode=mode,
            therapist={"name": "Sarah", "specialization": "CBT"},
        )
        self.messages = [{"role": "system", "content": self.sys_prompt}]

    def init_mental_state(self, data):
        #
        # self.set_prompt(mode="mental_state")
        # res = self.model.invoke(self.messages)
        # self.mental_state = parse_json_response(res.content)
        # print(res)
        pass

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
        print("Client: ", res)
        # self.mental_state = res["mental_state"]
        # self.messages.append(
        #     {
        #         "role": "assistant",
        #         "content": res["response"],
        #     }
        # )
        # return res["response"], self.mental_state
        return res, self.mental_state

    def reset_agent(self):
        self.__init__(self.model)


class PatientPsiResponse(BaseModel):
    content: str = Field(description="Your generated response")


class PatientPsi(BaseAgent):
    r"""
    Patient-{Psi} Agent
    Based on the paper "PATIENT-Ψ: Using Large Language Models to Simulate Patients for Training Mental Health Professionals"
    """

    def __init__(self, client):
        super().__init__(role="patient", client=client)
        self.agent_name = "patient-psi"
        self.sys_prompt = get_prompt(self.role, self.agent_name)
        self.parser = PydanticOutputParser(pydantic_object=PatientPsiResponse)

    def set_prompt(self, data):
        sys_prompt = self.sys_prompt.render(
            data=data,
            patientTypeContent="You should try your best to act like a patient who talks a lot: 1) you may provide detailed responses to questions, even if directly relevant, 2) you may elaborate on personal experiences, thoughts, and feelings extensively, and 3) you may demonstrate difficulty in allowing the therapist to guide the conversation. But you must not exceed 8 sentences each turn. Attention: The most important thing is to be as natural as possible and you should be verbose in some turns and be concise in other turns. You could listen to the therapist more as the session goes when you feel more trust in the therapist.",
        )
        self.messages = [
            {
                "role": "system",
                "content": sys_prompt,
            }
        ]

    def receive_message(self, msg):
        self.messages.append(
            {
                "role": "user",
                "content": msg,
            }
        )
        return self.generate_response()

    def generate_response(self):
        res = self.client.invoke(self.messages)
        # res = parse_json_response(res.content)
        return res.content
