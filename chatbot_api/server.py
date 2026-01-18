from fastapi import FastAPI, HTTPException
from chatbot_api.schemas import ChatRequest, ChatResponse
from chatbot_api.deps import get_chatbot

app = FastAPI(
    title="CoC Unified RAG Chatbot API",
    version="1.0.0",
    description="API สำหรับเรียกใช้งาน Unified RAG Chatbot (AstraDB + Manual QA Chain)"
)

@app.get("/api/v1/health")
def health():
    return {"status": "ok"}

@app.post("/api/v1/chat/completions")
def chat(req: ChatRequest):
    try:
        bot = get_chatbot()

        # automated version ใช้ hybrid classifier ภายใน bot อยู่แล้ว
        answer = bot.answer(req.question)

        # ดึง intent/score จากข้อความ log ออกมาคืนแบบ best-effort (ถ้าไม่เจอให้เป็น None)
        return ChatResponse(
            answer=answer,
            intent=None,
            confidence=None,
            contexts=None
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
