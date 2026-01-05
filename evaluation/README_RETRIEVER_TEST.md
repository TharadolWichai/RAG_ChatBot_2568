# 🔬 Retriever Evaluation - 10 Questions Test

## 📋 ภาพรวม

ไฟล์ `evaluate_retriever_10q.py` เป็น evaluation script สำหรับทดสอบประสิทธิภาพของ Retriever ด้วยคำถาม **10 ข้อ** จาก `test_forRetriver.json` โดยครอบคลุมทั้ง **10 intents** (1 คำถามต่อ 1 intent)

## 🎯 วัตถุประสงค์

- ทดสอบประสิทธิภาพ Retriever ของแต่ละ chatbot version
- เปรียบเทียบ 3 versions: **Rule-Based**, **LLM-Based**, และ **Hybrid**
- วิเคราะห์คุณภาพ contexts ที่ retrieve มา
- **เลือกได้** ว่าจะส่งกี่ contexts ให้ RAGAS: **5, 10, หรือทั้งหมด**
- ประเมินด้วย RAGAS metrics: Faithfulness, Context Precision, Context Recall
- วัด Answer Relevancy เป็น bonus metric (สำหรับดูเฉยๆ ไม่โฟกัส)

## 📊 คำถามทดสอบ (10 ข้อ)

| # | Intent | คำถาม | Category | Difficulty |
|---|--------|-------|----------|------------|
| 1 | allpeople | ข้อมูลทั้งหมดของอาจารย์พุธษดี | general | medium |
| 2 | contact | ช่องทางติดต่อมหาวิทยาลัย | simple | easy |
| 3 | links | จองห้องประชุม | simple | easy |
| 4 | scholarship | รายละเอียดของทุนการศึกษาทั้งหมด | general | hard |
| 5 | student_club | สมาชิกของสโมสรนักศึกษา | general | medium |
| 6 | students | คู่มือสหกิจศึกษา | simple | easy |
| 7 | researchgroup | กลุ่มวิจัยทั้งหมด | general | medium |
| 8 | bsc_entrance | การรับเข้าปริญญาตรี | general | hard |
| 9 | digital_services | บริการดิจิทัลทั้งหมด | general | hard |
| 10 | graduate | สาขาวิชาของปริญญาเอก | simple | medium |

### Distribution:
- **Categories:** Simple (4), General (6)
- **Difficulty:** Easy (3), Medium (4), Hard (3)
- **Intents:** ครอบคลุมทั้ง 10 intents

## 🚀 วิธีใช้งาน

### 1. เตรียมสภาพแวดล้อม

```bash
# ติดตั้ง dependencies
pip install ragas datasets langchain-openai langchain-huggingface

# ตรวจสอบ .env file
OPENAI_API_KEY=your_api_key_here
# หรือ
OPENROUTER_API_KEY=your_openrouter_key_here
```

### 2. รันการทดสอบ

```bash
cd evaluation
python evaluate_retriever_10q.py
```

### 3. เลือกจำนวน Contexts

เมื่อรันจะถามให้เลือกจำนวน contexts:
```
📚 Select number of contexts to send to RAGAS:
   1. Top 5 contexts
   2. Top 10 contexts
   3. All contexts (no limit)

Select option (1-3) [default: 3]:
```

**คำแนะนำการเลือก:**
- **Top 5 (ตัวเลือก 1):** ⚡ เร็ว, ใช้ API น้อย, เหมาะสำหรับ quick test
- **Top 10 (ตัวเลือก 2):** ⚖️ สมดุล, ครอบคลุมปานกลาง
- **All contexts (ตัวเลือก 3):** 🎯 ครบถ้วนที่สุด, ใช้เวลานานกว่า, เหมาะสำหรับ comprehensive evaluation

### 4. ผลลัพธ์ที่ได้

Script จะแสดงผลและบันทึก:

#### Console Output:
```
🔬 Retriever Evaluation - 10 Questions Test
================================================

📋 รายการคำถามที่จะทดสอบ:
   1. [allpeople          ] ข้อมูลทั้งหมดของอาจารย์พุธษดี
      Difficulty: medium | Category: general
   ...

🧪 Evaluating: Rule-Based
================================================
[1/10] Intent: allpeople | MEDIUM | general
❓ Question: ข้อมูลทั้งหมดของอาจารย์พุธษดี
─────────────────────────────────────────────
      📚 Retrieved 8 contexts → Using top 5  ← แสดงว่า retrieve 8 แต่ใช้ 5
   ⏱️  Response Time: 2.34s
   📏 Answer Length: 456 chars
   📚 Contexts Retrieved: 5
   📊 Contexts Stats:
      - Total length: 2345 chars
      - Avg length: 469 chars
      - Min/Max: 123/789 chars
...

📊 COMPARISON SUMMARY - 10 Questions Retriever Test
================================================
Metric                         | Rule-Based          | LLM-Based           | Hybrid
─────────────────────────────────────────────
📈 Main Metrics
Faithfulness                   |              0.8956 |              0.9012 |              0.9123
Context Precision              |              0.8234 |              0.8567 |              0.8789
Context Recall                 |              0.7890 |              0.8012 |              0.8345
─────────────────────────────────────────────
💡 Bonus Metric (for reference)
Answer Relevancy               |              0.0567 |              0.0612 |              0.0589
─────────────────────────────────────────────
⚡ Performance & Retriever Stats
Avg Response Time (s)          |               2.45 |               3.67 |               2.89
Avg Retrieved/Question         |                8.2 |                8.5 |                8.3  ← จำนวนที่ retrieve มา
Avg Used/Question              |                5.0 |                5.0 |                5.0  ← จำนวนที่ส่งให้ RAGAS
Avg Context Length (chars)     |               450 |               523 |               489
Total Retrieved                |               82 |               85 |               83
Total Used                     |               50 |               50 |               50
```

#### Saved JSON File:
```json
{
  "evaluation_info": {
    "test_file": "test_forRetriver.json",
    "total_questions": 10,
    "timestamp": "2025-01-20T...",
    "metrics": ["faithfulness", "context_precision", "context_recall"]
  },
  "results": {
    "Rule-Based": {
      "ragas_scores": { ... },
      "performance": { ... },
      "retriever_stats": { ... }
    },
    ...
  }
}
```

## 📈 Metrics ที่วัด

### RAGAS Metrics (Main):
1. **Faithfulness** (0.0-1.0) 🎯 **Main Focus**
   - ความถูกต้องของคำตอบเมื่อเทียบกับ contexts
   - คะแนนสูง = คำตอบไม่มีข้อมูลที่ไม่ได้มาจาก contexts

2. **Context Precision** (0.0-1.0) 🎯 **Main Focus**
   - ความแม่นยำของ contexts ที่ retrieve มา
   - คะแนนสูง = contexts ที่ retrieve มามีความเกี่ยวข้องสูง

3. **Context Recall** (0.0-1.0) 🎯 **Main Focus**
   - ความครอบคลุมของ contexts ที่ retrieve มา
   - คะแนนสูง = retrieve ได้ contexts ที่จำเป็นครบถ้วน

### RAGAS Bonus Metric:
4. **Answer Relevancy** (0.0-1.0) 💡 **For Reference Only**
   - ความสอดคล้องระหว่างคำตอบกับคำถาม
   - คะแนนสูง = คำตอบตรงประเด็นกับคำถาม
   - **หมายเหตุ:** ค่าอาจจะต่ำ (0.05-0.10) ซึ่งเป็นปกติ ไม่ต้องกังวล แค่ดูเฉยๆ

### Performance Metrics:
- **Response Time:** เวลาตอบกลับเฉลี่ย (วินาที)
- **Errors:** จำนวน errors ที่เกิดขึ้น

### Retriever Stats:
- **Avg Retrieved/Question:** จำนวน contexts เฉลี่ยที่ retriever ดึงมาได้ (ก่อนจำกัด)
- **Avg Used/Question:** จำนวน contexts เฉลี่ยที่ส่งให้ RAGAS (หลังจำกัด top 5/10/all)
- **Avg Context Length:** ความยาวเฉลี่ยของแต่ละ context (chars)
- **Total Retrieved:** จำนวน contexts ทั้งหมดที่ retriever ดึงมา
- **Total Used:** จำนวน contexts ทั้งหมดที่ส่งให้ RAGAS ประเมิน

## 🔍 จุดเด่นของการทดสอบนี้

### 1. **เลือกจำนวน Contexts ได้ + การันตีคุณภาพ**
- ⚡ **Top 5:** เร็ว, ประหยัด API calls, เหมาะสำหรับ quick iteration
- ⚖️ **Top 10:** สมดุลระหว่างความครอบคลุมและความเร็ว
- 🎯 **All contexts:** ครบถ้วนที่สุด, ดีที่สุดสำหรับ final evaluation
- เปรียบเทียบได้ว่าจำนวน contexts ส่งผลต่อ metrics อย่างไร

**✅ การันตี:** ทุก Retriever เรียง contexts ตาม **relevance score จากสูง→ต่ำ** แล้ว
- การเลือก Top 5/10 = ได้ contexts ที่ relevant ที่สุด
- Retrievers ที่เรียงด้วย combined_score: 8 ตัว
- Retrievers ที่เรียงด้วย priority: 2 ตัว (ResearchGroup, BSCEntrance)

### 2. **ครอบคลุมทุก Intent**
- ทดสอบทั้ง 10 intents (1 คำถาม/intent)
- ตรวจสอบว่า retriever ทำงานได้ดีกับทุก collection หรือไม่

### 3. **วิเคราะห์ Retriever โดยตรง**
- นับจำนวน contexts ที่ retrieve มา (ก่อนและหลังจำกัด)
- วัดความยาวและคุณภาพของ contexts
- เปรียบเทียบประสิทธิภาพระหว่าง 3 versions

### 4. **ทดสอบหลายระดับความยาก**
- Easy (3): ข้อมูลพื้นฐาน ตรงไปตรงมา
- Medium (4): ข้อมูลปานกลาง ต้องรวบรวม
- Hard (3): ข้อมูลซับซ้อน ต้องรวบรวมหลายแหล่ง

### 5. **ประหยัดเวลา**
- ทดสอบเพียง 10 คำถาม (แทน 30-50)
- เลือก contexts ได้ตามความเหมาะสม
- เหมาะสำหรับ iterative testing
- รวดเร็ว แต่ครอบคลุมทุก intent

## 📂 Output Files

ไฟล์ผลลัพธ์จะถูกบันทึกด้วยรูปแบบ:
```
retriever_eval_10q_[contexts]_YYYYMMDD_HHMMSS.json
```

ตัวอย่าง:
```
retriever_eval_10q_5ctx_20250120_143025.json     ← Top 5 contexts
retriever_eval_10q_10ctx_20250120_150530.json    ← Top 10 contexts
retriever_eval_10q_all_20250120_162045.json      ← All contexts
```

ชื่อไฟล์จะบอกว่าใช้ contexts แบบไหนในการทดสอบ

## 🔧 การปรับแต่ง

### เปลี่ยน Test Questions:
แก้ไขไฟล์ `test_forRetriver.json`:
```json
{
  "test_questions": [
    {
      "id": 1,
      "question": "คำถามของคุณ",
      "expected_intent": "intent_name",
      "ground_truth": "คำตอบที่คาดหวัง",
      "category": "simple",
      "difficulty": "easy"
    }
  ]
}
```

### เปิด/ปิด Chatbot Version:
แก้ไขใน `evaluate_retriever_10q.py`:
```python
# ปิด LLM-Based (ตัวอย่าง)
LLM_BASED_AVAILABLE = False
```

### เปลี่ยน RAGAS Metrics:
แก้ไข metrics ที่ต้องการใน evaluation:
```python
ragas_results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        context_precision,
        context_recall,
        # answer_relevancy,  # เพิ่มได้ถ้าต้องการ
    ],
    llm=llm,
    embeddings=embeddings
)
```

## 🆚 เปรียบเทียบกับ evaluate_chatbots.py

| Feature | evaluate_chatbots.py | evaluate_retriever_10q.py |
|---------|---------------------|--------------------------|
| จำนวนคำถาม | ปรับได้ (3-275+) | 10 ข้อ (fixed) |
| คำถามจาก | test_questions.json | test_forRetriver.json |
| เน้น | การเปรียบเทียบ chatbot | การวิเคราะห์ retriever |
| Intent coverage | บางส่วน | ทั้งหมด (10/10) |
| Retriever stats | ไม่มี | ✅ มี (detailed) |
| Use case | Full evaluation | Quick retriever test |

## 🔒 การยืนยันคุณภาพ Contexts

### ทำไมมั่นใจว่าได้ Contexts ที่ดีที่สุด?

ทุก Retriever ใน `main_app/` มีการเรียงลำดับ contexts ตาม relevance แล้ว:

```python
# ตัวอย่างจาก ScholarshipRetriever
candidate_docs.sort(
    key=lambda doc: doc.metadata.get("combined_score", 0), 
    reverse=True  # ← สูงสุดก่อน
)
```

**วิธีการคำนวณ Relevance:**
- **Combined Score (8 retrievers):** รวมคะแนนจาก BM25 (text search) + Vector (semantic search)
- **Priority (2 retrievers):** ตรวจสอบ keywords → จัดอันดับความสำคัญ

**ดังนั้น:**
- ✅ Context ที่ 1 = Relevant ที่สุด
- ✅ Context ที่ 2 = Relevant รองลงมา
- ✅ Context ที่ 5 = Top 5 ที่ดีที่สุด
- ✅ Context ที่ 10 = Top 10 ที่ดีที่สุด

เมื่อเลือก "Top 5" คุณจะได้ contexts ที่มี relevance score สูงสุด 5 อันดับแรก!

## 💡 Tips

### 1. **เลือก Contexts อย่างฉลาด**
```bash
# Quick iteration (เร็ว, ประหยัด)
Select option: 1  (Top 5)

# Testing ปกติ (สมดุล)
Select option: 2  (Top 10)

# Final evaluation (ครบถ้วน)
Select option: 3  (All)
```

### 2. **เปรียบเทียบผลกระทบของ Contexts**
```bash
# รันด้วย Top 5
python evaluate_retriever_10q.py  # เลือก 1

# รันด้วย All contexts
python evaluate_retriever_10q.py  # เลือก 3

# เปรียบเทียบว่า metrics เปลี่ยนแปลงอย่างไร
```

### 3. **ใช้สำหรับ Iterative Testing**
```bash
# ทดสอบหลายรอบเพื่อปรับ retriever
python evaluate_retriever_10q.py  # รอบ 1 (Top 5 - เร็ว)
# ปรับ embeddings หรือ retriever settings
python evaluate_retriever_10q.py  # รอบ 2 (Top 5 - เร็ว)
# ทดสอบครั้งสุดท้าย
python evaluate_retriever_10q.py  # รอบ 3 (All - ครบถ้วน)
```

### 4. **วิเคราะห์จำนวน Contexts**
ดูจำนวน contexts ที่ retrieve มา vs ที่ใช้:
- **Retrieved มาก แต่ใช้น้อย:** retriever ดึงมาเยอะ แต่ส่วนใหญ่ไม่เกี่ยวข้อง
- **Retrieved น้อย:** อาจจะตั้งค่า top_k ต่ำเกินไป
- **Retrieved = Used:** ดึงมาพอดี (ถ้าเลือก All contexts)

### 5. **วิเคราะห์ Context Precision vs Recall**
- **High Precision, Low Recall:** retrieve น้อย แต่แม่นยำ
- **Low Precision, High Recall:** retrieve เยอะ แต่มี noise
- **Both High:** ⭐ ดีที่สุด!

## 🐛 Troubleshooting

### Error: "No test questions loaded"
```bash
# ตรวจสอบว่าไฟล์ test_forRetriver.json อยู่ใน evaluation/
ls evaluation/test_forRetriver.json
```

### Error: "RAGAS not installed"
```bash
pip install ragas datasets langchain-openai langchain-huggingface
```

### Error: "No API key found"
```bash
# ตรวจสอบ .env
cat .env | grep API_KEY
```

## 📚 เอกสารเพิ่มเติม

- [RAGAS Documentation](https://docs.ragas.io/)
- [LangChain Documentation](https://python.langchain.com/)
- [evaluate_chatbots.py](./evaluate_chatbots.py) - Full evaluation script

## 🎯 Next Steps

หลังจากรันการทดสอบแล้ว:

1. **วิเคราะห์ผลลัพธ์:** ดู metrics ที่ต่ำกว่าคาดหวัง
2. **ตรวจสอบ Contexts:** ดูว่า retriever ดึงข้อมูลที่เกี่ยวข้องมาหรือไม่
3. **ปรับปรุง Retriever:** 
   - เปลี่ยน embeddings model
   - ปรับ similarity threshold
   - เพิ่ม/ลด top_k contexts
4. **ทดสอบซ้ำ:** รัน evaluation อีกครั้งเพื่อดูการปรับปรุง

---

**สร้างโดย:** ChatBot RAG CS KKU Team  
**วันที่อัพเดทล่าสุด:** 2025-01-20  
**Version:** 1.0.0

