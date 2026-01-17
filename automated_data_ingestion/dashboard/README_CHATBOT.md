# Chatbot Tab in Dashboard

## 📋 ภาพรวม

Dashboard ได้เพิ่ม **Tab ใหม่ "💬 Chatbot"** ที่ให้คุณสามารถใช้งาน RAG Chatbot ได้โดยตรงผ่าน Web Interface!

## 🚀 การใช้งาน

### 1. เริ่มต้นใช้งาน

1. รัน Dashboard:
   ```bash
   streamlit run automated_data_ingestion/dashboard/app.py
   ```

2. ไปที่ Tab **"💬 Chatbot"**

3. คลิก **"🚀 Initialize Chatbot"** เพื่อเริ่มต้นระบบ

4. รอให้ระบบโหลดเสร็จ (จะแสดง "✅ Chatbot initialized successfully!")

### 2. ถามคำถาม

- พิมพ์คำถามในช่อง **"พิมพ์คำถามของคุณ..."** ที่ด้านล่าง
- กด Enter หรือคลิกส่ง
- ระบบจะตอบคำถามโดยใช้ Hybrid Intent Classification

### 3. Features

- ✅ **Chat History**: เก็บประวัติการสนทนาไว้ใน session
- ✅ **Clear Chat**: ลบประวัติการสนทนา
- ✅ **Reload**: รีโหลด chatbot (เมื่อเพิ่ม collection ใหม่)
- ✅ **Export Chat**: ดาวน์โหลดประวัติการสนทนาเป็น JSON

## 🎯 ตัวอย่างคำถาม

- "อาจารย์สมชาย" → อาจารย์และบุคลากร
- "ติดต่อวิทยาลัย" → ข้อมูลติดต่อ
- "ลิงก์จองห้องประชุม" → ลิงก์และระบบ
- "ทุนการศึกษา" → ทุนการศึกษา
- "กลุ่มวิจัย AIDA" → กลุ่มวิจัย

## 🔧 Technical Details

### Chatbot Integration

- ใช้ `UnifiedChatbotAutomated` จาก `main_app/main_unified_chatbot_automated.py`
- Auto-discover collections จาก AstraDB
- สร้าง retrievers และ QA chains แบบ dynamic

### Session State

Dashboard เก็บข้อมูลใน `st.session_state`:
- `chatbot`: Chatbot instance
- `chat_history`: ประวัติการสนทนา

### Output Capture

ระบบจะ capture debug output จาก chatbot เพื่อแสดงใน expander "🔍 Debug Info"

## ⚠️ ข้อควรระวัง

1. **Initialize ก่อนใช้**: ต้อง initialize chatbot ก่อนถามคำถาม
2. **Collections**: ต้องมี collections ใน AstraDB ก่อน
3. **Performance**: การ initialize อาจใช้เวลาสักครู่ (ขึ้นอยู่กับจำนวน collections)
4. **Session**: Chat history จะหายเมื่อ refresh หน้าเว็บ (แต่สามารถ export ได้)

## 🔄 Workflow

```
1. สร้าง Collections ผ่าน Tab "📝 Create New Job"
   ↓
2. ไปที่ Tab "💬 Chatbot"
   ↓
3. Initialize Chatbot
   ↓
4. ถามคำถาม
   ↓
5. ระบบจะค้นหาจาก collections ที่เกี่ยวข้อง
   ↓
6. แสดงคำตอบ
```

## 💡 Tips

- **Reload เมื่อเพิ่ม Collection ใหม่**: หลังจากสร้าง collection ใหม่ ให้คลิก "🔄 Reload" เพื่อให้ chatbot รู้จัก collection ใหม่
- **Export Chat History**: ใช้เพื่อเก็บประวัติการสนทนาไว้
- **Debug Info**: เปิดดู debug info เพื่อเข้าใจว่า chatbot ทำงานอย่างไร

## 🐛 Troubleshooting

### Chatbot ไม่ initialize

- ตรวจสอบว่า collections มีอยู่ใน AstraDB
- ตรวจสอบ environment variables (.env)
- ดู error details ใน expander

### ไม่พบคำตอบ

- ตรวจสอบว่า collection มีข้อมูล
- ลองถามคำถามใหม่หรือใช้คำอื่น
- ดู debug info เพื่อเข้าใจปัญหา

### Performance ช้า

- ลดจำนวน collections
- ใช้ metadata filter
- ปรับ chunk size

## 📚 Related Files

- `main_app/main_unified_chatbot_automated.py` - Chatbot implementation
- `automated_data_ingestion/dashboard/app.py` - Dashboard with chatbot tab
- `main_app/README_AUTOMATED_CHATBOT.md` - Chatbot documentation

