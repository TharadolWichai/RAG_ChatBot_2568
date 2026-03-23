import asyncio
import io
import os
from contextlib import redirect_stdout, redirect_stderr
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from chatbot_api.schemas import ChatRequest, ChatResponse
from chatbot_api.deps import get_chatbot
from chatbot_api.feedback import router as feedback_router

SERVER_ANSWER_TIMEOUT = int(os.getenv("SERVER_ANSWER_TIMEOUT", "150"))
_executor = ThreadPoolExecutor(max_workers=4)

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

def _run_bot_answer(bot, question: str, strict_mode: bool, return_contexts: bool):
    """รัน bot.answer() พร้อมจับ stdout/stderr (ฟังก์ชัน blocking ที่ถูกรันใน thread)"""
    output_buffer = io.StringIO()
    with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
        try:
            result = bot.answer(
                question,
                strict_mode=strict_mode,
                return_contexts=return_contexts,
            )
        except TypeError:
            result = bot.answer(question)
    return result, output_buffer.getvalue()


@app.post("/api/v1/chat/completions")
async def chat(req: ChatRequest):
    try:
        bot = get_chatbot(model=req.model)
        loop = asyncio.get_event_loop()

        try:
            result, debug_output = await asyncio.wait_for(
                loop.run_in_executor(
                    _executor,
                    _run_bot_answer,
                    bot,
                    req.question,
                    req.strict_mode,
                    req.return_contexts,
                ),
                timeout=SERVER_ANSWER_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=(
                    f"การประมวลผลใช้เวลานานเกิน {SERVER_ANSWER_TIMEOUT} วินาที "
                    "กรุณาลองใหม่หรือถามคำถามที่สั้นกว่านี้"
                ),
            )

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

        if not req.return_contexts:
            contexts = None

        return ChatResponse(
            answer=answer,
            intent=intent,
            confidence=confidence,
            contexts=contexts,
            debug_output=debug_output if debug_output else None,
        )

    except HTTPException:
        raise
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