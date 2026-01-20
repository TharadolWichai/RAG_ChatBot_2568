from pydantic import BaseModel
from typing import Optional, List


class ChatRequest(BaseModel):
    question: str
    strict_mode: Optional[bool] = False
    return_contexts: Optional[bool] = False
    return_debug: Optional[bool] = True   # ✅ เพิ่ม: ขอ debug จาก API


class ChatResponse(BaseModel):
    answer: str
    intent: Optional[str] = None
    confidence: Optional[float] = None
    contexts: Optional[List[str]] = None
    debug_output: Optional[str] = None    # ✅ เพิ่ม: debug สำหรับหน้าเว็บ