# Chatbot API (Unified RAG Chatbot)

โฟลเดอร์ `chatbot_api` ทำหน้าที่เป็น **Backend API Layer** สำหรับเรียกใช้งาน  
**Unified RAG Chatbot ของวิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น**

API นี้ถูกออกแบบมาเพื่อให้
- Frontend (เช่น Streamlit, Web App, Mobile App)
- หรือระบบภายนอก (Third-party systems)

สามารถเรียกใช้งาน Chatbot ได้ผ่าน HTTP API อย่างเป็นมาตรฐาน

---

## 📁 โครงสร้างโฟลเดอร์

```text
chatbot_api/
│
├── server.py        # FastAPI application (entry point)
├── deps.py          # Dependency injection (chatbot / classifier singleton)
├── schemas.py       # Pydantic schemas (Request / Response)
└── README.md        # เอกสารอธิบาย API


---

## ⚙️ ความสามารถของ API

- เรียกใช้งาน Unified RAG Chatbot ผ่าน HTTP
- รองรับ Hybrid Intent Classification
- รองรับ Strict / Non-strict mode
- รองรับการส่ง context กลับ (สำหรับ debug / analysis)
- ออกแบบให้ใช้งานต่อได้ง่ายใน Postman และ Frontend

---

## ▶️ การรันระบบ (Run API)

### 1️⃣ เปิดใช้งาน Virtual Environment

```bash
venv\Scripts\activate

2️⃣ รัน FastAPI Server
uvicorn chatbot_api.server:app --reload --host 0.0.0.0 --port 8000

หากรันสำเร็จ จะเห็นข้อความประมาณนี้
Uvicorn running on http://0.0.0.0:8000

ตัวอย่างการเรียก Postman
POST /api/v1/chat/completions

Request Body (JSON)
{
  "question": "ติดต่อวิทยาลัยการคอมพิวเตอร์ได้ช่องทางไหนบ้าง",
  "strict_mode": false,
  "return_contexts": false
}