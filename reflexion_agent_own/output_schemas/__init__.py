from pydantic import BaseModel, Field
from typing import List

class CritiqueOutput(BaseModel):
    missing: List[str] = Field(description="List of elements missing from the content")
    superfluous: List[str] = Field(description="List of superfluous elements in the content")

class QualityAssessmentOutput(BaseModel):
    needs_improvement: bool = Field(description="Whether the answer needs another iteration of improvement")
    reasoning: str = Field(description="Explanation of the assessment")