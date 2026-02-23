# ⚡ Quick Start: Manual Evaluation (Hybrid Chatbot)

เริ่มต้นใช้งานภายใน 3 นาที!

---

## 🚀 เริ่มเลย (3 ขั้นตอน)

### 1️⃣ เตรียม Dataset

```bash
cd evaluation
```

ใช้ dataset ตัวอย่าง:
```bash
# ใช้เลย (10 คำถาม)
cat dataset_hybrid_example.json
```

หรือสร้างของคุณเอง:
```json
{
  "dataset": [
    {"id": 1, "question": "อาจารย์สมชาย", "ground_truth": "..."},
    {"id": 2, "question": "ติดต่อวิทยาลัย", "ground_truth": "..."}
  ]
}
```

---

### 2️⃣ รันการประเมินผล

```bash
python manual_evaluate_hybrid.py \
  --dataset dataset_hybrid_example.json \
  --output results_hybrid.json
```

**จะเห็น:**
```
🤖 กำลังเริ่มต้น UnifiedChatbotAutomated...
🔍 Discovering available collections...
✅ Unified Chatbot (Automated) initialized with 10 agents

####################################################
📝 คำถามที่ 1/10 (ID: 1)
####################################################
❓ อาจารย์สมชาย

============================================================
🔀 กำลังวิเคราะห์คำถามด้วย Hybrid Classification...
============================================================
   ✅ Rule-Based มั่นใจสูง (คะแนน: 9.00)

🎯 Hybrid Intent Classification Result:
   ประเภท: allpeople
   ความมั่นใจ: 9.00
   วิธีการ: rule_based

➡️  เลือก Agent: 👨‍🏫 อาจารย์และบุคลากร
📚 กำลังดึง contexts...
   ✅ ได้ 5 contexts

[... LLM processing ...]

✅ บันทึก ID 1 สำเร็จ!
   - Contexts: 5
   - Classification: allpeople (9.00)
   - Response time: 2.35s
   - Total in file: 1 questions

[... ทำต่อจนครบ 10 คำถาม ...]

🎉 การประเมินผลเสร็จสิ้น!
   เวลารวม: 18.45s
   เวลาเฉลี่ย: 1.85s per question
```

---

### 3️⃣ ตรวจสอบผลลัพธ์

```bash
python manual_evaluate_hybrid.py --summary results_hybrid.json
```

**จะแสดง:**
```
============================================================
📊 สรุปข้อมูลในไฟล์: results_hybrid.json
============================================================

📋 Metadata:
   Chatbot Type: UnifiedChatbotAutomated
   Classification: hybrid
   Total Questions: 10

📝 Results:
   🎯 Intent Distribution:
      allpeople: 2
      contact: 2
      scholarship: 1
      links: 1
      research: 1
      bsc_entrance: 1
      digital_services: 1
      students: 1

   🔧 Classification Method:
      rule_based: 9
      llm_fallback: 1

   📚 Contexts:
      Total: 45
      Average: 4.5 per question

   ⏱️  Response Time:
      Total: 18.45s
      Average: 1.85s per question
```

---

## ✅ เสร็จแล้ว!

ไฟล์ `results_hybrid.json` พร้อมใช้งาน มี:
- ✅ คำถาม 10 ข้อ
- ✅ คำตอบจากโมเดล
- ✅ Contexts ทั้งหมด
- ✅ Classification info

---

## 🎯 ขั้นตอนถัดไป

### เทียบกับโมเดลอื่น

รันโมเดลอื่นด้วย dataset เดียวกัน:

```bash
# Hybrid (เสร็จแล้ว)
python manual_evaluate_hybrid.py \
  --dataset dataset_hybrid_example.json \
  --output results_hybrid.json

# Rule-Based (ถ้ามี)
python manual_evaluate_rule.py \
  --dataset dataset_hybrid_example.json \
  --output results_rule.json

# LLM-Based (ถ้ามี)
python manual_evaluate_llm.py \
  --dataset dataset_hybrid_example.json \
  --output results_llm.json
```

### ประเมินผลด้วย RAGAS

```bash
# (ยังไม่ได้สร้าง - coming soon)
python evaluate_from_json.py \
  --dataset dataset_hybrid_example.json \
  --results results_hybrid.json \
  --output evaluation_results.json
```

---

## 💡 โหมดอื่นๆ

### Interactive Mode (ทดสอบเร็ว)

```bash
python manual_evaluate_hybrid.py --interactive
```

พิมพ์คำถามเพื่อทดสอบทันที:
```
❓ คำถามที่ 1 (หรือ 'exit'): อาจารย์สมชาย
[... ดูผลลัพธ์ ...]
💾 บันทึกลง JSON ไหม? (y/n): y
✅ บันทึกสำเร็จ
```

### Resume จากที่หยุด

ถ้ารันไปแล้ว 5 คำถาม แล้วหยุด:

```bash
python manual_evaluate_hybrid.py \
  --dataset dataset_hybrid_example.json \
  --output results_hybrid.json \
  --start-from 6
```

เริ่มต่อจาก ID 6

---

## 📚 เอกสารเพิ่มเติม

- [README_MANUAL_EVALUATION.md](README_MANUAL_EVALUATION.md) - คู่มือฉบับสมบูรณ์
- [json_response_logger.py](json_response_logger.py) - API Documentation

---

## 🎉 Happy Evaluating!

