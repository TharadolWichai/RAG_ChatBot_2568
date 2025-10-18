# 🔧 Troubleshooting API Issues for RAGAS Evaluation

## ปัญหาที่พบจาก Terminal Output

### ❌ **Error: "Incorrect API key provided: sk-or-v1..."**

```
AuthenticationError(Error code: 401 - {'error': {'message': 'Incorrect API key provided: sk-or-v1...
```

**สาเหตุ:** RAGAS พยายามใช้ OpenRouter API key (`sk-or-v1-...`) กับ OpenAI endpoint โดยตรง

---

## 🎯 **วิธีแก้ปัญหา**

### **วิธีที่ 1: ใช้ Native OpenAI API Key (แนะนำที่สุด)**

1. ไปที่ https://platform.openai.com/api-keys
2. สร้าง API key ใหม่ (รูปแบบ: `sk-proj-xxxxx`)
3. เพิ่มใน `.env`:

```env
# Native OpenAI API Key (for RAGAS evaluation)
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxx

# OpenRouter API Key (for chatbot)
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxx
OPENAI_API_BASE=https://openrouter.ai/api/v1
```

**ข้อดี:**
- ✅ ทำงานได้ทันที ไม่มีปัญหา
- ✅ RAGAS ใช้ OpenAI key, Chatbot ใช้ OpenRouter key
- ✅ แยก billing ชัดเจน

**ข้อเสีย:**
- ❌ ต้องมี 2 API keys
- ❌ ต้องจ่ายค่าใช้จ่ายทั้ง 2 ที่

---

### **วิธีที่ 2: ใช้ OpenRouter เดียว (ที่เราพยายามทำ)**

ปัจจุบันโค้ดถูกปรับให้รองรับ OpenRouter แล้ว แต่อาจจะยังมีปัญหา:

1. ตรวจสอบ `.env`:

```env
OPENAI_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxx
OPENAI_API_BASE=https://openrouter.ai/api/v1
```

2. ทดสอบ API:

```bash
cd evaluation
python test_gpt4o_api.py
```

3. ถ้าทดสอบผ่าน แต่ RAGAS ยังไม่ได้:

```bash
# ลบ cache และทดสอบใหม่
rm -rf __pycache__
python evaluate_chatbots.py
```

**ข้อดี:**
- ✅ ใช้ API key เดียว
- ✅ จ่ายค่าใช้จ่ายที่เดียว

**ข้อเสีย:**
- ❌ RAGAS อาจจะมีปัญหากับ OpenRouter
- ❌ ต้องตั้งค่าให้ถูกต้อง

---

### **วิธีที่ 3: ใช้โมเดลอื่นแทน GPT-4o**

ถ้า GPT-4o ไม่ได้ผลผ่าน OpenRouter, ลองใช้โมเดลอื่น:

แก้ไขใน `evaluate_chatbots.py`:

```python
llm = ChatOpenAI(
    model="openai/gpt-3.5-turbo",  # เปลี่ยนกลับเป็น 3.5
    # หรือ
    model="anthropic/claude-3-sonnet",  # ลอง Claude
    temperature=0.1,
    api_key=api_key,
    base_url=base_url
)
```

---

## 📊 **วิเคราะห์ผลลัพธ์ที่ได้**

### **ก่อนแก้ไข (GPT-3.5-turbo + OpenRouter ครึ่งทำงาน):**
```
Faithfulness:       1.0000  ✅
Answer Relevancy:   nan     ❌ (API error)
Context Precision:  0.0000  ⚠️
Context Recall:     1.0000  ✅
```

### **หลังแก้ไข (GPT-4o + OpenRouter ใช้ไม่ได้):**
```
Faithfulness:       0.0000  ❌ (API error ทั้งหมด)
Answer Relevancy:   nan     ❌ (API error)
Context Precision:  0.0000  ❌ (API error)
Context Recall:     0.0000  ❌ (API error)
```

**สรุป:** GPT-4o ยังใช้งานผ่าน OpenRouter ไม่ได้กับ RAGAS

---

## 🚀 **แนวทางแก้ไขแบบทีละขั้น**

### **Step 1: ทดสอบ API**
```bash
cd evaluation
python test_gpt4o_api.py
```

### **Step 2: ถ้า Step 1 ผ่าน → ปัญหาอยู่ที่ RAGAS**
- RAGAS อาจจะไม่รองรับ OpenRouter ดี
- ลองใช้ Native OpenAI key แทน

### **Step 3: ถ้า Step 1 ไม่ผ่าน → ปัญหาอยู่ที่ API Key**
- API key หมดอายุ
- API key ไม่มีสิทธิ์ใช้ GPT-4o
- OpenRouter ไม่รองรับ GPT-4o

### **Step 4: เลือกทางออก**
```python
# ตัวเลือกที่ 1: Native OpenAI (แนะนำ)
OPENAI_API_KEY=sk-proj-xxxxx

# ตัวเลือกที่ 2: GPT-3.5-turbo ผ่าน OpenRouter
model="openai/gpt-3.5-turbo"

# ตัวเลือกที่ 3: Claude ผ่าน OpenRouter
model="anthropic/claude-3-sonnet"
```

---

## 💡 **คำแนะนำสุดท้าย**

**สำหรับ Production:**
1. ใช้ **Native OpenAI API Key** สำหรับ RAGAS evaluation
2. ใช้ **OpenRouter** สำหรับ Chatbot (ประหยัดกว่า)
3. แยก billing ชัดเจน

**สำหรับ Development/Testing:**
1. ใช้ **GPT-3.5-turbo** ผ่าน OpenRouter (ถูกกว่า)
2. ยอมรับว่า accuracy อาจจะต่ำกว่า GPT-4o

**ปัจจุบัน (2025-01-12):**
- ✅ GPT-3.5-turbo ผ่าน OpenRouter → ทำงานได้ (บางส่วน)
- ❌ GPT-4o ผ่าน OpenRouter → ไม่ได้ผล
- ✅ GPT-4o ผ่าน Native OpenAI → ควรได้ผล (ยังไม่ได้ทดสอบ)

---

## 📞 **ติดต่อขอความช่วยเหลือ**

ถ้ายังแก้ไม่ได้ ให้ส่งข้อมูลนี้:

1. **Terminal output** จาก `python test_gpt4o_api.py`
2. **`.env` configuration** (ซ่อน API key)
3. **Error message** จาก RAGAS

---

**Last Updated:** 2025-01-12  
**Status:** OpenRouter + GPT-4o ยังไม่ทำงานกับ RAGAS  
**Recommendation:** ใช้ Native OpenAI API Key

