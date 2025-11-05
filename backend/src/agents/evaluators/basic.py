from ..base import BaseAgent
from utils import load_prompts
from typing import Dict, List, Any
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class DimensionFeedback(BaseModel):
    strengths: List[str] = Field(
        description="A list of specific strengths demonstrated by the therapist in this dimension (2-3 items)",
    )
    areas_for_improvement: List[str] = Field(
        description="A list of specific areas for improvement for the therapist in this dimension (2-3 items)",
    )


class ActiveListeningDimensions(BaseModel):
    empathic_understanding: DimensionFeedback = Field(
        description="Evaluation of the therapist's demonstration of empathic understanding",
    )
    unconditional_positive_regard: DimensionFeedback = Field(
        description="Evaluation of the therapist's demonstration of unconditional positive regard",
    )
    congruence: DimensionFeedback = Field(
        description="Evaluation of the therapist's demonstration of congruence",
    )


class ActiveListeningFeedback(BaseModel):
    overall_score: int = Field(
        description="Overall score for the therapist's active listening skills in this session",
    )
    summary: str = Field(
        description="A brief summary of the therapist's performance regarding active listening skills  (you should reference what the therapist said/do in the session that makes you give this rating)",
    )
    dimension_feedback: ActiveListeningDimensions = Field(
        description="Detailed feedback for each active listening dimension"
    )


class CBTDimensions(BaseModel):
    identification_of_core_beliefs: DimensionFeedback = Field(
        description="Evaluation of the therapist's ability to identify core beliefs",
    )
    socratic_questioning: DimensionFeedback = Field(
        description="Evaluation of the therapist's use of Socratic questioning",
    )
    actionable_homework: DimensionFeedback = Field(
        description="Evaluation of the therapist's collaboration on actionable homework",
    )
    cognitive_restructuring: DimensionFeedback = Field(
        description="Evaluation of the therapist's use of cognitive restructuring techniques",
    )


class CBTFeedback(BaseModel):
    overall_score: int = Field(
        description="Overall score for the therapist's CBT skills in this session",
    )
    summary: str = Field(
        description="A brief summary of the therapist's performance (you should reference what the therapist said/do in the session that makes you give this rating)",
    )
    dimension_feedback: CBTDimensions = Field(
        description="Detailed scores for each CBT dimension"
    )


class GeneralEvaluator(BaseAgent):
    def __init__(self, model_client: BaseChatModel):
        self.role = "evaluator"
        self.agent_type = "basic"
        self.model_client = model_client
        self.prompt = """You are an expert Cognitive Behavioral Therapy (CBT) 
        Your task is to analyze a transcript of a counseling session between a novice therapist and a client.
        Accordingly you should provide structured, constructive feedback based on core CBT principles.

        Scoring Guideline:

        9-10: Excellent, nuanced application of CBT. Effectively identifies core beliefs, uses Socratic questioning masterfully, and collaborates on actionable homework.

        7-8: Good, solid CBT practice. Identifies automatic thoughts and some intermediate beliefs, but could go deeper.

        5-6: Adequate. Shows basic understanding but may rely on reassurance or miss key cognitive distortions.

        3-5: Needs significant development. Missed opportunities, overly advice-giving, or failed to engage with the cognitive model.

        1-2: The conversation is too short to evaluate OR the performance is Poor, Lacks understanding of CBT principles, fails to identify thoughts/beliefs, and does not engage the client effectively OR

        Based on the following conversation between the therapist and client, evaluate the therapist's performance in exhibiting these core conditions.
        {dialogue_history}

        用中文回复
        """
        self.messages = []

    def generate(self, messages: List[Dict]):
        messages = [f"{msg['role']}: {msg['content']}" for msg in messages]
        model_client = self.model_client.with_structured_output(CBTFeedback)
        self.prompt = self.prompt.format(dialogue_history="\n".join(messages))
        res = model_client.invoke(self.prompt)
        return res

    def reset(self):
        self.messages = []


class SkillEvaluator(BaseAgent):
    def __init__(self, model_client: BaseChatModel):
        self.role = "evaluator"
        self.agent_type = "basic"
        self.model_client = model_client
        self.prompt = """You are an expert clinical supervisor specializing in client-centered therapy based on Carl Rogers' principles. 
        Your task is to analyze a transcript of a counseling session, focusing exclusively on the therapist's demonstration of three core conditions: 
        1) Empathic Understanding
        2) Unconditional Positive Regard
        3) Congruence

        Scoring Guideline (per dimension and overall):

        9-10: The condition was consistently and skillfully demonstrated, creating a palpable sense of safety and understanding.

        7-8: The condition was often present and effective, with only minor lapses.

        5-6: The condition was intermittently present but inconsistent; the therapist understands the concept but struggles with fluent application.

        3-5: The condition was largely absent; responses were frequently judgmental, incongruent, or misattuned.

        1-2: The condition was completely absent throughout the session OR the conversation is too short to evaluate.

        Based on the following conversation between the therapist and client, evaluate the therapist's performance in exhibiting these core conditions.
        {dialogue_history}

        IMPORTANT: You should focus on what the therapist said and did in the session!

        用中文回复
        """
        self.messages = []

    def generate(self, messages: List[Dict]):
        messages = [f"{msg['role']}: {msg['content']}" for msg in messages]
        model_client = self.model_client.with_structured_output(ActiveListeningFeedback)
        self.prompt = self.prompt.format(dialogue_history="\n".join(messages))
        res = model_client.invoke(self.prompt)
        return res

    def reset(self):
        self.messages = []
