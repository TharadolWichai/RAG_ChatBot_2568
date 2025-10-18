# 🚀 Quick Start - RAGAS Evaluation

## เริ่มต้นใช้งาน 3 ขั้นตอน

---

## ⚙️ Step 1: ติดตั้ง

```bash
# ติดตั้ง packages ที่จำเป็น
pip install -r evaluation/requirements_evaluation.txt
```

**หรือติดตั้งแบบย่อ:**
```bash
pip install ragas datasets langchain-openai
```

---

## 🧪 Step 2: รัน Evaluation

```bash
cd evaluation
python evaluate_chatbots.py
```

**เลือกขนาดการทดสอบ:**
- Option 1: Quick (3 คำถาม, ~2-3 นาที)
- Option 2: Standard (6 คำถาม, ~5-7 นาที) ← **แนะนำ**
- Option 3: Full (12 คำถาม, ~10-15 นาที)

**ผลลัพธ์ที่ได้:**
- ✅ RAGAS Scores สำหรับ chatbot แต่ละตัว
- ⚡ Response time เฉลี่ย
- 💾 ไฟล์ JSON: `evaluation_results_YYYYMMDD_HHMMSS.json`

---

## 📊 Step 3: วิเคราะห์ผลลัพธ์

```bash
python analyze_results.py
```

**จะได้:**
- 📈 Detailed metrics analysis
- 💰 Cost analysis
- 🏆 Winners by category
- 📊 Comparison chart (PNG)
- 📄 Markdown report

---

## 📋 ตัวอย่างผลลัพธ์

```
================================================================================
📊 COMPARISON SUMMARY
================================================================================

Metric                    | Rule-Based      | LLM-Based       | Hybrid         
--------------------------------------------------------------------------------
Faithfulness              |          0.8750 |          0.9200 |          0.9100
Answer Relevancy          |          0.7800 |          0.8900 |          0.8850
Context Precision         |          0.8200 |          0.8400 |          0.8500
Context Recall            |          0.8500 |          0.8600 |          0.8700
Context Relevancy         |          0.8100 |          0.8300 |          0.8400
--------------------------------------------------------------------------------
Avg Response Time (s)     |          2.30   |          5.80   |          3.20  
Errors                    |             0   |             0   |             0   
================================================================================

🏆 Best Performers:
   Faithfulness             : LLM-Based (0.9200)
   Answer Relevancy         : LLM-Based (0.8900)
   Context Precision        : Hybrid (0.8500)
   Context Recall           : Hybrid (0.8700)
   Context Relevancy        : Hybrid (0.8400)
   Fastest Response         : Rule-Based (2.30s)
```

---

## 💡 คำแนะนำ

### ✅ **สำหรับ Presentation:**
1. รัน Standard test (option 2)
2. วิเคราะห์ผลด้วย `analyze_results.py`
3. ใช้กราฟและ Markdown report

### ✅ **สำหรับ Production:**
- ดูที่ Hybrid Version
- ตรวจสอบ Cost Analysis
- ตัดสินใจจาก Overall Score

### ✅ **สำหรับ Debug/Improvement:**
- รัน Full test
- ดูที่คำถามที่ตอบผิด
- ปรับ threshold และ weights

---

## 🎯 คำถามที่ควรถาม

### **จากผลลัพธ์:**
1. Version ไหน Faithfulness สูงสุด? (ตอบตาม context ดีที่สุด)
2. Version ไหนเร็วที่สุด? (User experience ดีที่สุด)
3. Version ไหนประหยัดที่สุด? (Cost-effective)
4. Version ไหนเหมาะกับ Production? (สมดุลดีที่สุด)

### **คำตอบที่คาดหวัง:**
1. LLM-Based (0.92+)
2. Rule-Based (2-3s)
3. Rule-Based ($0)
4. **Hybrid** (สมดุลดี!) ⭐

---

## 🔧 Troubleshooting

### ❌ "ModuleNotFoundError: No module named 'ragas'"
```bash
pip install ragas datasets langchain-openai
```

### ❌ "No chatbots available"
- ตรวจสอบว่าอยู่ใน directory ถูกต้อง
- ตรวจสอบ `main_app/` มีไฟล์ chatbot ครบ
- ลองรัน chatbot แยกก่อน (เช่น `python main_app/main_unified_chatbot.py`)

### ❌ "OPENROUTER_API_KEY not found"
- เพิ่ม `OPENROUTER_API_KEY=your_key` ใน `.env`
- หรือข้าม LLM-Based และ Hybrid (ใช้แค่ Rule-Based)

---

## 📞 Need Help?

ถ้ามีปัญหาหรือคำถาม:
1. อ่าน `README_EVALUATION.md` สำหรับรายละเอียดเพิ่มเติม
2. ตรวจสอบ error messages
3. ลอง Quick test (option 1) ก่อน

---

**Happy Evaluating!** 🎉

