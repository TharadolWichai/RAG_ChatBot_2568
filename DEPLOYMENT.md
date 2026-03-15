# 🚀 Deployment Guide - Railway.app

## 📋 Overview

โปรเจคนี้ประกอบด้วย 2 services:
1. **Backend API** (FastAPI) - port 8000
2. **Frontend Chatbot** (Streamlit) - port 8501

---

## 🔧 Prerequisites

1. บัญชี Railway.app (Sign up ที่ https://railway.app)
2. GitHub repository (push โค้ดขึ้น GitHub)
3. Environment variables พร้อม:
   - `OPENAI_API_KEY`
   - `OPENAI_BASE_URL`
   - `ASTRA_DB_APPLICATION_TOKEN`
   - `ASTRA_DB_API_ENDPOINT`

---

## 📦 Step 1: Deploy Backend Service

### 1.1 สร้าง Project ใหม่บน Railway
- ไปที่ https://railway.app/new
- เลือก "Deploy from GitHub repo"
- เชื่อม GitHub repository ของคุณ

### 1.2 Configure Backend Service
1. ไปที่ Project Settings
2. กด "New Service" → เลือก repo
3. ตั้งค่า:
   - **Name:** `chatbot-backend`
   - **Root Directory:** `/` (leave as is)
   - **Start Command:** `bash start_backend.sh`

### 1.3 เพิ่ม Environment Variables
ไปที่ **Variables** tab และเพิ่ม:

```
OPENAI_API_KEY=sk_qNEUYR92Kz84GdWRGIWkw29ZdMVRyg80eyvZGI8rplFQOy1DYiuQwrEvOu9XMMEG
OPENAI_BASE_URL=https://gen.ai.kku.ac.th/api/v1
OPENAI_MODEL=gemini-2.5-flash-lite
CHATBOT_MODEL=gpt-5-mini
ASTRA_DB_APPLICATION_TOKEN=<your-token>
ASTRA_DB_API_ENDPOINT=<your-endpoint>
```

### 1.4 เพิ่ม Volume สำหรับ Feedback Data
1. ไปที่ **Storage** tab
2. กด "New Volume"
3. ตั้งค่า:
   - **Mount Path:** `/app/feedbacks`
   - **Size:** 1GB

### 1.5 Deploy
- กด "Deploy"
- รอ deploy เสร็จ (ประมาณ 2-3 นาที)
- **คัดลอก URL ของ Backend** (เช่น `https://chatbot-backend.railway.app`)

---

## 📦 Step 2: Deploy Frontend Service

### 2.1 สร้าง Service ใหม่
1. ใน project เดียวกัน กด "New Service"
2. เลือก repo เดิม
3. ตั้งค่า:
   - **Name:** `chatbot-frontend`
   - **Root Directory:** `/`
   - **Start Command:** `bash start_frontend.sh`

### 2.2 เพิ่ม Environment Variables
ไปที่ **Variables** tab และเพิ่ม:

```
# ใช้ URL จาก Backend service ที่ deploy ไปแล้ว
CHATBOT_API_URL=https://chatbot-backend.railway.app/api/v1/chat/completions
CHATBOT_HEALTH_URL=https://chatbot-backend.railway.app/api/v1/health
CHATBOT_META_URL=https://chatbot-backend.railway.app/api/v1/chatbot/meta
FEEDBACK_API_URL=https://chatbot-backend.railway.app/api/v1/feedback
FEEDBACK_STATS_URL=https://chatbot-backend.railway.app/api/v1/feedback/stats

# KKU API (สำหรับดึงรายชื่อโมเดล)
OPENAI_API_KEY=sk_qNEUYR92Kz84GdWRGIWkw29ZdMVRyg80eyvZGI8rplFQOy1DYiuQwrEvOu9XMMEG
OPENAI_BASE_URL=https://gen.ai.kku.ac.th/api/v1
```

### 2.3 Deploy
- กด "Deploy"
- รอ deploy เสร็จ
- **คัดลอก URL ของ Frontend** (เช่น `https://chatbot-frontend.railway.app`)

---

## ✅ Step 3: ทดสอบ

### 3.1 ทดสอบ Backend
```bash
curl https://chatbot-backend.railway.app/health
```

ควรได้:
```json
{"status":"healthy","service":"backend"}
```

### 3.2 ทดสอบ Frontend
เปิดเบราว์เซอร์ไปที่: `https://chatbot-frontend.railway.app`

---

## 🎯 Step 4: ส่ง URL ให้ผู้ทดสอบ

ส่ง **Frontend URL** ให้ผู้ทดสอบ 50 คน:
```
https://chatbot-frontend.railway.app
```

---

## 📊 Step 5: ดึง Feedback Data

### 5.1 ดูสถิติ Feedback
```bash
curl https://chatbot-backend.railway.app/api/v1/feedback/stats
```

### 5.2 Export Feedback เป็น CSV
ไปที่:
```
https://chatbot-backend.railway.app/api/v1/feedback/export
```

หรือใช้ curl:
```bash
curl -o feedbacks.csv https://chatbot-backend.railway.app/api/v1/feedback/export
```

---

## 💰 Cost Management

### ประมาณการค่าใช้จ่าย:
- **Backend:** ~$5/month (ถ้ารันตลอด 24/7)
- **Frontend:** ~$5/month (ถ้ารันตลอด 24/7)
- **Total:** ~$10/month

### ประหยัดค่าใช้จ่าย:
1. **เปิดเฉพาะตอนทดสอบ:**
   - ไปที่ Service Settings
   - กด "Pause" เมื่อไม่ใช้
   - กด "Resume" เมื่อต้องการใช้

2. **ใช้ Free Trial:**
   - Railway ให้ $5 credit ฟรีสำหรับผู้ใช้ใหม่

---

## 🔒 Security Notes

1. **Environment Variables:**
   - ไม่ต้อง commit `.env` ขึ้น GitHub
   - ตั้งค่าผ่าน Railway dashboard เท่านั้น

2. **API Keys:**
   - ใช้ Railway's Secret Management
   - Rotate keys หลังเสร็จสิ้นการทดสอบ

---

## 🐛 Troubleshooting

### Backend ไม่ทำงาน
1. เช็ค logs: Railway dashboard → Backend service → Logs
2. เช็ค environment variables ครบหรือไม่
3. ทดสอบ health endpoint: `curl https://your-backend.railway.app/health`

### Frontend เชื่อม Backend ไม่ได้
1. เช็คว่า `CHATBOT_API_URL` ถูกต้อง
2. เช็คว่า Backend service ทำงานอยู่
3. เช็ค CORS settings

### Volume ไม่มี feedback data
1. เช็คว่า mount path ถูกต้อง: `/app/feedbacks`
2. ทดสอบส่ง feedback ผ่าน UI
3. เช็ค logs ว่ามี error ไหม

---

## 📞 Support

หากมีปัญหา:
1. เช็ค logs บน Railway dashboard
2. ทดสอบ API endpoints ด้วย curl
3. เช็ค environment variables

---

## 🎉 Next Steps

หลังจาก deploy สำเร็จ:
1. ✅ ทดสอบ chatbot
2. ✅ ทดสอบระบบ feedback
3. ✅ ส่ง URL ให้ผู้ทดสอบ 50 คน
4. ✅ รวบรวม feedback
5. ✅ Export CSV และวิเคราะห์ผล
