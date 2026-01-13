# api_unified_chatbot.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import UnifiedChatbotHybrid
from main_app.main_unified_chatbot_hybrid import UnifiedChatbotHybrid

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class Question(BaseModel):
    question: str

# Initialize chatbot once
chatbot = UnifiedChatbotHybrid()

@app.get("/")
def root():
    return {"status": "Unified Hybrid RAG Chatbot API is running"}

@app.post("/chat")
def chat(data: Question):
    # ใช้ฟังก์ชัน answer ของ UnifiedChatbotHybrid
    answer = chatbot.answer(data.question)
    return {"answer": answer}
