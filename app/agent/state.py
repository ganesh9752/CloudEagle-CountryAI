from typing import TypedDict, Optional


class AgentState(TypedDict):
    query: str
    history: list[dict]  # previous Q&A pairs for follow-up context
    countries: list[str]
    fields: list[str]
    raw_data: list[dict]
    answer: str
    error: Optional[str]
