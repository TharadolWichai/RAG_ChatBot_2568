from fastapi import FastAPI, HTTPException
from chatbot_api.schemas import ChatRequest, ChatResponse
from chatbot_api.deps import get_chatbot
from chatbot_api.feedback import router as feedback_router  # ✅ เพิ่ม

import io
from contextlib import redirect_stdout, redirect_stderr

app = FastAPI(
    title="CoC Unified RAG Chatbot API",
    version="1.0.0",
    description="API สำหรับเรียกใช้งาน Unified RAG Chatbot (AstraDB + Manual QA Chain)"
)

# ✅ เพิ่ม Feedback Router
app.include_router(feedback_router, tags=["Feedback"])

@app.get("/health")
def health():
    """Health check endpoint for Railway"""
    return {"status": "healthy", "service": "backend"}

@app.get("/api/v1/health")
def health_v1():
    """API v1 health check"""
    return {"status": "ok"}

@app.post("/api/v1/chat/completions")
def chat(req: ChatRequest):
    try:
        # Get chatbot with specified model (or use default)
        bot = get_chatbot(model=req.model)

        output_buffer = io.StringIO()
        with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
            # automated version ใช้ hybrid classifier ภายใน bot อยู่แล้ว
            # ส่ง flags เข้าไปเฉพาะใน API layer (ถ้า bot ไม่รองรับ flags นี้ จะ fallback เองด้านล่าง)
            try:
                result = bot.answer(
                    req.question,
                    strict_mode=req.strict_mode,
                    return_contexts=req.return_contexts
                )
            except TypeError:
                # ถ้า bot.answer ไม่รับพารามิเตอร์เพิ่ม ให้เรียกแบบเดิม
                result = bot.answer(req.question)

        debug_output = output_buffer.getvalue()

        # รองรับทั้งกรณีคืน str หรือ dict
        answer = None
        intent = None
        confidence = None
        contexts = None

        if isinstance(result, str):
            answer = result
        elif isinstance(result, dict):
            answer = result.get("answer") or result.get("text") or ""
            intent = result.get("intent")
            confidence = result.get("confidence")
            contexts = result.get("contexts")
        else:
            answer = str(result)

        # ถ้า request ไม่ต้องการ contexts ให้ปิดทิ้ง
        if not req.return_contexts:
            contexts = None

        # ส่ง debug_output กลับไปให้หน้าเว็บโชว์ได้เหมือนเดิม
        return ChatResponse(
            answer=answer,
            intent=intent,
            confidence=confidence,
            contexts=contexts,
            debug_output=debug_output if debug_output else None
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/chatbot/meta")
def chatbot_meta():
    bot = get_chatbot()  # Use default model for meta info

    chatbot_map = getattr(bot, "chatbot_map", {}) or {}

    agents = []
    for _, cfg in chatbot_map.items():
        agents.append({
            "name": cfg.get("name"),
            "collection": cfg.get("collection"),
            "icon": cfg.get("icon", "📦")
        })

    return {
        "agents": agents,
        "total": len(agents)
    }