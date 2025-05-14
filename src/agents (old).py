import os
import time
import random
from dotenv import load_dotenv
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types import RoleType

from pydantic import BaseModel, Field

load_dotenv("./.env")


class PatientResponse(BaseModel):
    reasoning: Field("reason about your thoughts and feelings")
    emotion: Field("Current emotion. Could be either sad/angry/happy")
    response: Field("Your generated response")


def create_llm(model_type: ModelType, response_format: BaseModel):
    return ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type=model_type,
        api_key=os.environ.get("API_KEY"),
        url=os.environ.get("API_URL"),
        model_config_dict={
            "temperature": 0.6,
            "max_tokens": 4096,
            "response_format": response_format,
        },
    )


class Patient(ChatAgent):
    def __init__(self):
        # self.demographics = {}
        # self.set_demographics()  # Set default demographics (temp testing)
        self.sys_prompt = BaseMessage(
            role_name="Patient",
            role_type=RoleType.ASSISTANT,
            meta_dict=None,
            content=f"You are a helpful assistant.",
        )
        # self.disorder = disorder
        # self.symptom_severity = symptom_severity  # 0-10 scale
        # self.coping_mechanism = coping_mechanism
        # self.session_history = []
        super().__init__(self.sys_prompt, model=base_model_api)

    def gen_res(self, msg):
        # Generate a response using the CAMEL-AI chat agent
        res = self.step(msg)
        return res.msgs[0].content

    def set_demographics(
        self,
        name="Sam",
        age=26,
        gender="Male",
        ethinicity="White",
        education="Postgraduate",
    ):
        self.demographics = {
            "name": name,
            "age": age,
            "gender": gender,
            "ethnicity": ethnicity,
            "education": education,
            "social_status": "",
        }

    def update_state(self, therapy_effect, adherence):
        # Simulate life between sessions
        if adherence and random.random() < 0.7:  # 70% chance to follow advice
            self.symptom_severity = max(0, self.symptom_severity - therapy_effect)
            self.coping_mechanism = "diary"
        else:
            self.symptom_severity = min(
                10, self.symptom_severity + random.randint(0, 2)
            )
            self.coping_mechanism = "smoking"
        self.session_history.append(self.symptom_severity)
        print(
            f"{self.name}'s new symptom severity: {self.symptom_severity}/10, Coping: {self.coping_mechanism}"
        )


# Define Therapist Agent
class Therapist(ChatAgent):
    def __init__(self, name, expertise, experience_level):
        self.name = name
        self.expertise = expertise
        self.experience_level = experience_level
        self.effectiveness = self.experience_level * 0.5

        # Define role and system message for CAMEL-AI
        system_message = BaseMessage(
            role_name="Therapist",
            role_type=RoleType.ASSISTANT,
            meta_dict=None,
            content=f"I am {name}, a therapist specializing in {expertise} with {experience_level} years of experience.",
        )
        super().__init__(
            system_message, model_type="gpt-3.5-turbo"
        )  # Requires API key setup

    def conduct_session(self, patient):
        # Simulate therapy session using CAMEL-AI chat
        patient_input = BaseMessage(
            role_name="Patient",
            role_type=RoleType.USER,
            meta_dict=None,
            content=f"Hi, I’m {patient.name}. I’ve been feeling {patient.symptom_severity}/10 due to my {patient.disorder}.",
        )
        response = self.step(patient_input)
        print(f"\n--- Session Start ---")
        print(f"Patient: {patient_input.content}")
        print(f"Therapist: {response.msg.content}")
        print(f"--- Session End ---")
        return self.effectiveness
