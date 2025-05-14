import os
import yaml
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Dict
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv("./.env")

client = ChatOpenAI(
    model=os.environ.get("MODEL_NAME"),
    base_url=os.environ.get("API_URL"),
    api_key=os.environ.get("API_KEY"),
    temperature=0.6,
)

agent_configs = {}
with open("src/config/agents.yaml") as stream:
    try:
        agent_configs = yaml.safe_load(stream)
    except yaml.YAMLError as e:
        print(e)


class MentalState(BaseModel):
    beliefs: str = Field(description="Current beliefs")
    emotions: str = Field(description="Current emotions")
    desires: str = Field(description="Current desires")
    intents: str = Field(description="Current intents")


class PatientResponse(BaseModel):
    emotion: str = Field(description="Current emotion. Could be either sad/angry/happy")
    content: str = Field(description="Your generated response based on your emotions")


class TherapistResponse(BaseModel):
    reasoning: str = Field(
        description="reason about the patients' thoughts and feelings in 1-2 sentences."
    )
    patient_mental_state: Dict[str, MentalState] = Field(
        description="Current mental state of the patient"
    )
    content: str = Field(description="Your generated response based on your reasoning")


def to_bullet_list(data):
    res = ""

    # If it's a list
    if isinstance(data, list):
        res += "\n".join([f"- {item}" for item in data])
    # If it's a dict
    elif isinstance(data, dict):
        for k, v in data.items():
            res += f"- {k.capitalize()}: {v}\n"
    else:
        print("Invalid data type for to_bullet_list")

    return res


def parse_json_response(res):
    try:
        res = res.replace("```json", "").replace("```", "")
        res = json.loads(res)
        return res
    except json.JSONDecodeError:
        return res


class Patient:
    def __init__(self):
        self.model = client
        self.sys_prompt = agent_configs["patient"]["sys_prompt"]
        self.messages = []
        self.role = "patient"
        self.parser = PydanticOutputParser(pydantic_object=PatientResponse)

    def fill_prompt(self, profile):
        demographics = profile["demographics"]
        personality = profile["personality"]
        current_issue = profile["current_issue"]
        coping_mechanisms = current_issue["coping_mechanisms"]
        mechanisms = coping_mechanisms["Negative"] + coping_mechanisms["Positive"]

        self.sys_prompt = self.sys_prompt.format(
            demographics=to_bullet_list(demographics),
            personality=to_bullet_list(personality),
            social_circle=len(profile["social_circle"]),
            description=current_issue["description"],
            duration=current_issue["duration"],
            severity=current_issue["severity"],
            triggers=(", ").join(current_issue["triggers"]),
            coping_mechanisms=to_bullet_list(mechanisms),
        )
        self.sys_prompt += "\n # Output\n\n" + self.parser.get_format_instructions()
        self.messages = [{"role": "system", "content": self.sys_prompt}]

    def reset_messages(self):
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
        return f"{res['emotion']} -> {res['content']}"
        # return res.content


class Therapist:
    def __init__(self):
        self.model = client
        self.sys_prompt = agent_configs["therapist"]["sys_prompt"]
        self.messages = []
        self.role = "therapist"
        self.parser = PydanticOutputParser(pydantic_object=TherapistResponse)

    def fill_prompt(self, profile):
        demographics = profile["demographics"]
        personality = profile["personality"]
        self.sys_prompt = self.sys_prompt.format(
            demographics=to_bullet_list(profile["demographics"]),
            personality=to_bullet_list(profile["personality"]),
        )
        self.sys_prompt += "\n # Output\n\n" + self.parser.get_format_instructions()
        self.messages = [{"role": "system", "content": self.sys_prompt}]

    def reset_messages(self):
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
        print(res)

        return f"[{res['reasoning']}] -> {res['content']}"
