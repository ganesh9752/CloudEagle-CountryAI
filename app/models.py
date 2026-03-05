from typing import Optional
from pydantic import BaseModel, Field


class HistoryEntry(BaseModel):
    question: str
    answer: str


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    history: list[HistoryEntry] = Field(
        default_factory=list,
        max_length=5,
        description="Last few Q&A pairs for follow-up context",
    )


class QuestionResponse(BaseModel):
    answer: str
    countries: list[str] = []
    fields: list[str] = []
    error: Optional[str] = None
