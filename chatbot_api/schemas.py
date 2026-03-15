from pydantic import BaseModel
from typing import Optional, List


class ChatRequest(BaseModel):
    question: str
    strict_mode: Optional[bool] = False
    return_contexts: Optional[bool] = False
    return_debug: Optional[bool] = True   # ✅ เพิ่ม: ขอ debug จาก API
    model: Optional[str] = None           # ✅ เพิ่ม: เลือกโมเดล LLM


class ChatResponse(BaseModel):
    answer: str
    intent: Optional[str] = None
    confidence: Optional[float] = None
    contexts: Optional[List[str]] = None
    debug_output: Optional[str] = None    # ✅ เพิ่ม: debug สำหรับหน้าเว็บ


# ============================================
# Feedback Schemas (NEW)
# ============================================

class FeedbackRequest(BaseModel):
    """ข้อมูลที่ส่งมาจาก frontend เมื่อ user ให้ feedback"""
    session_id: str              # Session ID ของ user
    question: str                # คำถามที่ถาม
    answer: str                  # คำตอบที่ได้
    model: Optional[str] = None  # โมเดลที่ใช้
    intent: Optional[str] = None # Intent ที่จับได้
    rating: int                  # คะแนน 1-5 ดาว
    comment: str = ""            # ความคิดเห็น (optional)


class FeedbackResponse(BaseModel):
    """Response หลังบันทึก feedback สำเร็จ"""
    status: str
    feedback_id: str
    message: str = "Feedback saved successfully"