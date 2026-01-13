# 🔬 Rule-Based Strict Mode Evaluation - ทดสอบเฉพาะ Rule-Based

## 📁 ไฟล์ที่เกี่ยวข้อง

- **`evaluate_rule_based_strict.py`**: สคริปต์ทดสอบ Rule-Based Strict Mode อย่างเดียว
- **`test_forRetriver_no_keywords.json`**: ชุดคำถาม 38 ข้อ (แก้ไข 8 ข้อให้ไม่มีคีย์เวิร์ด)

---

## 🎯 วัตถุประสงค์

ทดสอบ **Rule-Based Chatbot ใน Strict Mode** เท่านั้น เพื่อดูว่า:
- ✅ ตอบได้กี่ข้อเมื่อ**มีคีย์เวิร์ด** (30 ข้อ)
- ❌ ตอบไม่ได้กี่ข้อเมื่อ**ไม่มีคีย์เวิร์ด** (8 ข้อ)
- 📊 **Faithfulness, Context Precision, Context Recall** ลดลงแค่ไหน

---

## 🚀 วิธีใช้งาน

### **รันสคริปต์:**

```bash
cd evaluation
python evaluate_rule_based_strict.py
```

**ระบบจะ:**
1. ✅ โหลด `test_forRetriver_no_keywords.json` โดยอัตโนมัติ
2. ✅ เปิด **Strict Mode** (ไม่มี multi-agent fallback)
3. ✅ ทดสอบ Rule-Based chatbot อย่างเดียว
4. ✅ แสดงผลลัพธ์และบันทึกเป็น JSON

**ไม่ต้องเลือกอะไร** - รันแล้วรอผลเลย! 🎉

---

## 📊 ผลลัพธ์ที่คาดหวัง

### **Strict Mode (No Fallback):**

```
📊 Rule-Based Strict Mode Evaluation Results
==================================================================================
📈 RAGAS Scores:
   - Faithfulness:       0.7500-0.8000  ⚠️ (ลดลงจาก 1.0000)
   - Answer Relevancy:   0.1000-0.1500  ⚠️ (ต่ำ เพราะ error messages)
   - Context Precision:  0.5500-0.6500  ⚠️ (ลดลง เพราะบางคำถามไม่มี contexts)
   - Context Recall:     0.7500-0.8500  ⚠️ (ลดลง เพราะข้อมูลไม่ครบ)

⚡ Performance:
   - Avg Response Time:  3-5s  ⚡ (เร็ว เพราะไม่มี fallback)
   - Errors:             8 ❌  (คำถามที่ลบคีย์เวิร์ด)

❌ Error Analysis:
   - Total Errors:              8
   - Errors from Modified Q:    8 🔧 (คำถามที่ลบคีย์เวิร์ด)
   - Errors from Original Q:    0 ✅ (คำถามที่มีคีย์เวิร์ด)

   📋 Error Details:
      1. Q4: ช่องทางอิเล็กทรอนิกส์ติดต่อคุณธนพล 🔧 (No Keywords)
         Error: ❌ [Error] Cannot answer: No matching keyword found...
      2. Q6: วิธีการเข้าถึงหน่วยงานมหาวิทยาลัย 🔧 (No Keywords)
         Error: ❌ [Error] Cannot answer: No matching keyword found...
      3. Q9: ขอใช้ห้องประชุม 🔧 (No Keywords)
         Error: ❌ [Error] Cannot answer: No matching keyword found...
      ... (และอีก 5 ข้อ)
```

---

## 🔍 การวิเคราะห์

### **8 คำถามที่ตอบไม่ได้:**

| ID | คำถาม (ไม่มีคีย์เวิร์ด) | Intent | คีย์เวิร์ดที่ลบ |
|----|-------------------------|--------|----------------|
| 4 | "ช่องทางอิเล็กทรอนิกส์ติดต่อคุณธนพล" | allpeople | `อีเมล`, `อาจารย์` |
| 6 | "วิธีการเข้าถึงหน่วยงานมหาวิทยาลัย" | contact | `ติดต่อ`, `ช่องทาง` |
| 9 | "ขอใช้ห้องประชุม" | links | `จอง` |
| 12 | "เอกสารร้องขอแก้ไขคะแนน" | links | `แบบฟอร์ม` |
| 14 | "ข้อมูลการสนับสนุนทางการเงินทั้งหมด" | scholarship | `ทุน`, `ทุนการศึกษา` |
| 21 | "หัวหน้าองค์กรนิสิตคือใคร" | student_club | `ประธาน`, `สโมสร` |
| 22 | "หนังสือแนะนำการทำงานร่วมกับบริษัท" | students | `สหกิจ`, `คู่มือ` |
| 29 | "เข้าศึกษาต่อระดับปริญญาตรีได้อย่างไร" | bsc_entrance | `รับเข้า` |
| 36 | "สาขาที่เปิดสอนระดับดุษฎีบัณฑิต" | graduate | `ปริญญาเอก` |

### **30 คำถามที่ตอบได้:**
ทั้งหมดเป็นคำถามที่**มีคีย์เวิร์ด**ตรงกับ Rule-Based classifier → ตอบได้ปกติ ✅

---

## 📈 ผลกระทบต่อ RAGAS Metrics

### **Faithfulness ลดลง:**
```
Faithfulness = จำนวน claims ที่ถูก support โดย contexts
               ────────────────────────────────────────────
                      จำนวน claims ทั้งหมด

- 30 ข้อที่มีคีย์เวิร์ด:  Faithfulness = 1.0
- 8 ข้อที่ไม่มีคีย์เวิร์ด: Faithfulness = 0.0 (error message)
- เฉลี่ย: (30×1.0 + 8×0.0) / 38 = 0.789 ⚠️
```

### **Context Precision ลดลง:**
- คำถามที่ตอบไม่ได้ → ไม่มี contexts ที่ relevant
- → Context Precision = 0 สำหรับ 8 ข้อนั้น
- → เฉลี่ยลดลง ⚠️

### **Context Recall ลดลง:**
- Ground truth ต้องการข้อมูลบางอย่าง
- แต่ไม่มี contexts ถูก retrieve มา (เพราะตอบไม่ได้)
- → Context Recall = 0 สำหรับ 8 ข้อนั้น
- → เฉลี่ยลดลง ⚠️

---

## 💡 สิ่งที่เรียนรู้

### **1. Rule-Based Limitations ชัดเจน:**
- ❌ **ขึ้นอยู่กับคีย์เวิร์ดมาก** → ไม่มีคีย์เวิร์ด = ตอบไม่ได้ (21% error rate)
- ❌ **ไม่เข้าใจความหมาย** → แม้คำถามจะหมายความเหมือนกัน
- ⚠️ **Metrics ลดลงทุกตัว** → Faithfulness, Context Precision, Context Recall

### **2. Strict Mode vs Normal Mode:**

| Feature | Strict Mode (ทดสอบนี้) | Normal Mode |
|---------|----------------------|-------------|
| ไม่เจอคีย์เวิร์ด | ❌ ตอบไม่ได้เลย | ✅ ใช้ multi-agent search (ช้า) |
| Errors | 8 errors (21%) | 0 errors (แต่ช้ามาก) |
| Response Time | ⚡ 3-5s (เร็ว) | 🐌 15-30s (ช้า) |
| เหมาะสำหรับ | 🔬 Testing, Evaluation | 🏭 Production (ไม่แนะนำ) |

### **3. ทำไมต้องใช้ Hybrid:**
- ✅ ใช้ Rule-Based ก่อน (เร็ว, ฟรี) เมื่อมีคีย์เวิร์ด
- ✅ Fallback เป็น LLM (แม่นยำ) เมื่อไม่มีคีย์เวิร์ด
- ✅ **Best of Both Worlds** → เร็ว + แม่นยำ + ไม่มี errors

---

## 📦 Output Files

เมื่อรันเสร็จ จะได้ไฟล์:

```
evaluation/rule_based_strict_eval_YYYYMMDD_HHMMSS.json
```

**เนื้อหาในไฟล์:**
```json
{
  "evaluation_info": {
    "test_file": "test_forRetriver_no_keywords.json",
    "total_questions": 38,
    "modified_questions": 8,
    "chatbot": "Rule-Based (Strict Mode - No Fallback)",
    "metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
  },
  "results": {
    "ragas_scores": {
      "faithfulness": 0.7895,
      "answer_relevancy": 0.1234,
      "context_precision": 0.5789,
      "context_recall": 0.7895
    },
    "performance": {
      "avg_response_time": 4.52,
      "errors_count": 8
    },
    "error_analysis": {
      "total_errors": 8,
      "errors_from_modified": 8,
      "errors_from_original": 0
    }
  }
}
```

---

## 🎓 คำแนะนำ

### **ควรใช้สคริปต์นี้เมื่อ:**
- 🔬 ต้องการทดสอบ Rule-Based Strict Mode เท่านั้น
- 📊 ต้องการดูว่า Faithfulness ลดลงแค่ไหนเมื่อไม่มีคีย์เวิร์ด
- 🎯 ต้องการ**พิสูจน์ข้อจำกัด**ของ Rule-Based
- 📈 ต้องการข้อมูลเปรียบเทียบกับ LLM/Hybrid

### **ไม่ควรใช้เมื่อ:**
- 🏭 ต้องการทดสอบ Production mode (ใช้ `evaluate_retriever_10q.py`)
- 📊 ต้องการเปรียบเทียบทั้ง 3 versions (ใช้ `evaluate_retriever_10q.py`)
- ✅ ต้องการทดสอบกับคำถามที่มีคีย์เวิร์ด (ใช้ `test_forRetriver.json`)

---

## 🔗 Related Files

- [`test_forRetriver_no_keywords.json`](./test_forRetriver_no_keywords.json) - ชุดคำถามที่ลบคีย์เวิร์ด
- [`test_forRetriver.json`](./test_forRetriver.json) - ชุดคำถามปกติ (มีคีย์เวิร์ด)
- [`evaluate_retriever_10q.py`](./evaluate_retriever_10q.py) - เปรียบเทียบทั้ง 3 versions
- [`main_unified_chatbot.py`](../main_app/main_unified_chatbot.py) - Rule-Based chatbot
- [`README_NO_KEYWORDS_TEST.md`](./README_NO_KEYWORDS_TEST.md) - คู่มือ no keywords test

---

## ✅ Quick Start

```bash
# 1. ไปที่ folder evaluation
cd evaluation

# 2. รันสคริปต์
python evaluate_rule_based_strict.py

# 3. กด Enter เพื่อเริ่มทดสอบ

# 4. รอ 5-10 นาที (ขึ้นอยู่กับ API speed)

# 5. ดูผลลัพธ์และบันทึกเป็น JSON
```

**ง่ายมาก! ไม่ต้องเลือกอะไรเลย** 🎉

---

**Last Updated:** 2025-01-20  
**Purpose:** Testing Rule-Based Strict Mode ONLY with NO keyword matching (focused evaluation)

