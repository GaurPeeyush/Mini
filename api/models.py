from pydantic import BaseModel
from typing import Optional, List

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    source: str  # "kb" or "llm"
    matchedQuestion: Optional[str] = None
    score: Optional[float] = None
    trace: Optional[List[str]] = None

class ClearHistoryResponse(BaseModel):
    success: bool