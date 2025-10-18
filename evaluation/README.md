# 🧪 Chatbot Evaluation Suite

ระบบประเมินประสิทธิภาพ RAG Chatbot ด้วย RAGAS Framework

---

## 📁 โครงสร้างไฟล์

```
evaluation/
├── README.md                       # คู่มือหลัก (ไฟล์นี้)
├── QUICKSTART.md                   # เริ่มต้นใช้งานแบบเร็ว
├── README_EVALUATION.md            # คู่มือ RAGAS แบบละเอียด
├── requirements_evaluation.txt     # Dependencies
│
├── evaluate_chatbots.py           # 🧪 Main RAGAS evaluation script
├── analyze_results.py             # 📊 วิเคราะห์และสร้างกราฟ
├── simple_test.py                 # ⚡ ทดสอบแบบง่าย (ไม่ใช้ RAGAS)
├── test_questions.json            # 📝 ชุดคำถามทดสอบ
│
└── results/                       # ผลลัพธ์ (สร้างอัตโนมัติ)
    ├── evaluation_results_*.json
    ├── comparison_chart_*.png
    └── evaluation_report_*.md
```

---

## 🚀 Quick Start

### 1️⃣ ติดตั้ง
```bash
pip install -r evaluation/requirements_evaluation.txt
```

### 2️⃣ รัน Evaluation
```bash
cd evaluation
python evaluate_chatbots.py
```

### 3️⃣ วิเคราะห์ผลลัพธ์
```bash
python analyze_results.py
```

**อ่านเพิ่มเติม:** `QUICKSTART.md`

---

## 📊 ไฟล์สำคัญ

### **evaluate_chatbots.py** - Main Evaluation
ประเมินประสิทธิภาพด้วย RAGAS metrics:
- ✅ Faithfulness (ความซื่อสัตย์)
- ✅ Answer Relevancy (ความเกี่ยวข้อง)
- ✅ Context Precision (ความแม่นยำ)
- ✅ Context Recall (ความครอบคลุม)
- ✅ Context Relevancy (ความเกี่ยวข้องของ context)

**ใช้เมื่อ:**
- ต้องการประเมินอย่างละเอียด
- เปรียบเทียบ 3 versions
- สร้างรายงานสำหรับอาจารย์

---

### **analyze_results.py** - Results Analysis
วิเคราะห์ผลลัพธ์และสร้าง:
- 📊 Comparison charts (PNG)
- 📄 Markdown reports
- 💰 Cost analysis
- 🏆 Winner rankings

**ใช้เมื่อ:**
- รัน evaluation เสร็จแล้ว
- ต้องการ visualize ผลลัพธ์
- เตรียม presentation

---

### **simple_test.py** - Quick Manual Test
ทดสอบแบบง่าย โดยไม่ใช้ RAGAS:
- ⚡ เร็วกว่า (ไม่ต้องรอ RAGAS)
- 💰 ไม่เสียค่า LLM (เฉพาะการทดสอบ)
- 📝 ดูผลลัพธ์ได้ทันที

**ใช้เมื่อ:**
- Debug chatbot
- ทดสอบเบื้องต้น
- ไม่ต้องการ detailed metrics

---

## 🎯 Use Cases

### **Case 1: Demo สำหรับอาจารย์**
```bash
# 1. รัน evaluation (Standard test)
python evaluate_chatbots.py
# เลือก option 2

# 2. สร้างกราฟและรายงาน
python analyze_results.py

# 3. นำ comparison_chart.png และ evaluation_report.md ไปใส่ใน slides
```

---

### **Case 2: ทดสอบเร็วๆ**
```bash
# ใช้ simple test (ไม่ต้อง RAGAS)
python simple_test.py
```

---

### **Case 3: วิจัยและเปรียบเทียบอย่างละเอียด**
```bash
# 1. รัน full evaluation
python evaluate_chatbots.py
# เลือก option 3

# 2. วิเคราะห์ผล
python analyze_results.py

# 3. ดูที่ JSON file สำหรับข้อมูลแต่ละคำถาม
```

---

## 📈 Expected Results

### **RAGAS Scores (0.0 - 1.0):**

| Metric | Good | Fair | Poor |
|--------|------|------|------|
| Faithfulness | ≥ 0.90 | 0.70-0.89 | < 0.70 |
| Answer Relevancy | ≥ 0.85 | 0.70-0.84 | < 0.70 |
| Context Precision | ≥ 0.80 | 0.65-0.79 | < 0.65 |
| Context Recall | ≥ 0.85 | 0.70-0.84 | < 0.70 |
| Context Relevancy | ≥ 0.80 | 0.65-0.79 | < 0.65 |

### **Performance:**

| Version | Avg Response Time | Cost/100Q | Quality |
|---------|-------------------|-----------|---------|
| Rule-Based | ~2-3s | $0 | Good |
| LLM-Based | ~5-7s | ~$0.02 | Excellent |
| **Hybrid** | **~3-4s** | **~$0.005** | **Very Good** ⭐ |

---

## 🎓 สำหรับ Presentation

### **Slides ที่แนะนำ:**

#### Slide 1: Problem Statement
- ปัญหา: ข้อมูลเยอะ, หาไม่เจอ, ไม่มี search ที่ดี
- Solution: RAG Chatbot with Multi-Agent System

#### Slide 2: System Architecture
- 9 Specialized Agents
- 3 Intent Classification Methods
- Hybrid Search (BM25 + Vector + PyThaiNLP)

#### Slide 3: Intent Classification Methods
```
┌─────────────┬──────────────┬──────────────┬─────────────┐
│ Method      │ Speed        │ Accuracy     │ Cost        │
├─────────────┼──────────────┼──────────────┼─────────────┤
│ Rule-Based  │ ⚡⚡⚡ Fast  │ ⭐⭐⭐ Good  │ 💰 Free     │
│ LLM-Based   │ ⚡ Slow      │ ⭐⭐⭐⭐⭐ Best │ 💰💰💰 $$$ │
│ Hybrid      │ ⚡⚡ Medium  │ ⭐⭐⭐⭐ V.Good│ 💰 Cheap   │
└─────────────┴──────────────┴──────────────┴─────────────┘
```

#### Slide 4: RAGAS Evaluation Results
- แสดงตารางเปรียบเทียบ
- แสดง comparison chart
- Highlight: Hybrid wins in most metrics

#### Slide 5: Recommendations
- **Production: Hybrid** (สมดุลดีที่สุด)
- **Demo: Rule-Based** (เร็ว ฟรี)
- **Research: LLM-Based** (แม่นยำที่สุด)

---

## 🛠️ Advanced Usage

### เพิ่มคำถามทดสอบ:

แก้ไข `test_questions.json`:
```json
{
  "id": 16,
  "question": "คำถามของคุณ",
  "expected_intent": "digital_services",
  "ground_truth": "คำตอบที่คาดหวัง",
  "category": "simple",
  "difficulty": "easy"
}
```

### ปรับ Threshold (Hybrid):

แก้ไข `main_app/main_unified_chatbot_hybrid.py`:
```python
HIGH_CONFIDENCE_THRESHOLD = 8.0  # ปรับค่านี้
# 6.0 = ใช้ Rule-Based บ่อยขึ้น (เร็ว, ประหยัด)
# 10.0 = ใช้ LLM บ่อยขึ้น (แม่นยำ, แพง)
```

### Export ผลลัพธ์:

```python
# ใน evaluate_chatbots.py
results = run_evaluation()

# Export to CSV
import pandas as pd
df = pd.DataFrame(results)
df.to_csv('results.csv')

# Export to Excel
df.to_excel('results.xlsx')
```

---

## 📚 Documentation

- **Quick Start:** `QUICKSTART.md` - เริ่มต้นใช้งานใน 3 ขั้นตอน
- **Detailed Guide:** `README_EVALUATION.md` - คู่มือ RAGAS แบบละเอียด
- **RAGAS Docs:** https://docs.ragas.io/

---

## ✅ Checklist

ก่อนรัน evaluation ตรวจสอบ:

- [ ] ติดตั้ง RAGAS แล้ว
- [ ] มี `.env` file พร้อม:
  - `ASTRA_DB_APPLICATION_TOKEN`
  - `ASTRA_DB_API_ENDPOINT`
  - `OPENROUTER_API_KEY`
- [ ] Chatbot ทุกตัวทำงานได้ (ทดสอบแยกก่อน)
- [ ] มี internet connection
- [ ] มี AstraDB collections พร้อมข้อมูล

---

## 🎯 TL;DR

```bash
# ติดตั้ง
pip install ragas datasets langchain-openai

# รัน evaluation
cd evaluation && python evaluate_chatbots.py

# วิเคราะห์ผล
python analyze_results.py

# เสร็จ! 🎉
```

---

## 💡 Tips

1. **ครั้งแรก:** ใช้ Quick test (option 1) เพื่อทดสอบว่าระบบทำงาน
2. **สำหรับ Demo:** ใช้ Standard test (option 2) ได้ผลลัพธ์ดี
3. **สำหรับวิจัย:** ใช้ Full test (option 3) ได้ข้อมูลครบถ้วน
4. **ประหยัดเวลา:** ใช้ `simple_test.py` สำหรับ quick check
5. **Cost Monitor:** ดู cost analysis ว่า Hybrid ประหยัดจริง

---

**Happy Evaluating!** 🚀

