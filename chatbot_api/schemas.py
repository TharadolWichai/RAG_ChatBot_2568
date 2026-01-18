from pydantic import BaseModel
from typing import Optional, List

class ChatRequest(BaseModel):
    question: str
    strict_mode: Optional[bool] = False
    return_contexts: Optional[bool] = False

class ChatResponse(BaseModel):
    answer: str
    intent: Optional[str] = None
    confidence: Optional[float] = None
    contexts: Optional[List[str]] = None
