from typing import Dict, List
from pydantic import BaseModel, Field


class DimensionFeedback(BaseModel):
    strengths: List[str] = Field(
        description="A list of specific strengths demonstrated by the therapist in this dimension (2-3 items)",
    )
    areas_for_improvement: List[str] = Field(
        description="A list of specific areas for improvement for the therapist in this dimension (2-3 items)",
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
