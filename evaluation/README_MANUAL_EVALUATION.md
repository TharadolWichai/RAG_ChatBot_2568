# 📝 คู่มือการใช้งาน Manual Evaluation (Hybrid Chatbot)

ระบบบันทึกคำถาม คำตอบ และ contexts แบบ Manual สำหรับ UnifiedChatbotAutomated

---

## 🎯 ภาพรวม

ระบบนี้ช่วยให้คุณ:
- ✅ ถามคำถามกับ Hybrid Chatbot ทีละข้อ
- ✅ บันทึก answer, contexts, และ classification info ลง JSON
- ✅ เตรียมข้อมูลสำหรับประเมินผลด้วย RAGAS
- ✅ Resume ได้ถ้าหยุดกลางคัน

---

## 📋 ไฟล์ที่เกี่ยวข้อง

```
evaluation/
├── json_response_logger.py           # Helper functions สำหรับบันทึก JSON
├── manual_evaluate_hybrid.py         # สคริปต์หลัก
├── dataset_hybrid_example.json       # Dataset ตัวอย่าง
└── results_hybrid.json               # ไฟล์ผลลัพธ์ (สร้างอัตโนมัติ)
```

---

## 🚀 วิธีใช้งาน

### ขั้นตอนที่ 1: เตรียม Dataset

สร้างไฟล์ `dataset.json`:

```json
{
  "dataset": [
    {
      "id": 1,
      "question": "อาจารย์สมชาย",
      "ground_truth": "ควรตอบข้อมูลเกี่ยวกับอาจารย์..."
    },
    {
      "id": 2,
      "question": "ติดต่อวิทยาลัย",
      "ground_truth": "ควรตอบเบอร์โทร อีเมล..."
    }
  ]
}
```

**หรือใช้ตัวอย่าง:**
```bash
cp evaluation/dataset_hybrid_example.json evaluation/my_dataset.json
```

---

### ขั้นตอนที่ 2: รันการประเมินผล

#### แบบที่ 1: Dataset Mode (แนะนำ)

```bash
cd evaluation
python manual_evaluate_hybrid.py --dataset dataset_hybrid_example.json --output results_hybrid.json
```

**จะเกิดอะไรขึ้น:**
1. โหลด chatbot และ dataset
2. ถามคำถามทีละข้อ
3. บันทึกลง `results_hybrid.json` ทันที
4. แสดง log ละเอียดทุกขั้นตอน

#### แบบที่ 2: Interactive Mode (ทดสอบเร็ว)

```bash
python manual_evaluate_hybrid.py --interactive --output test_results.json
```

**ลักษณะการทำงาน:**
- พิมพ์คำถามเพื่อทดสอบ
- เลือกว่าจะบันทึกหรือไม่
- พิมพ์ `summary` เพื่อดูสรุป
- พิมพ์ `exit` เพื่อออก

---

### ขั้นตอนที่ 3: ตรวจสอบผลลัพธ์

#### แสดงสรุปข้อมูล:

```bash
python manual_evaluate_hybrid.py --summary results_hybrid.json
```

**จะแสดง:**
- จำนวนคำถามทั้งหมด
- การกระจายตัวของ intent
- Classification method ที่ใช้
- จำนวน contexts เฉลี่ย
- เวลาตอบเฉลี่ย

#### เปิดไฟล์โดยตรง:

```bash
cat results_hybrid.json
# หรือ
code results_hybrid.json  # VS Code
```

---

## 📊 โครงสร้างไฟล์ผลลัพธ์

`results_hybrid.json`:

```json
{
  "metadata": {
    "chatbot_type": "UnifiedChatbotAutomated",
    "classification_method": "hybrid",
    "created_at": "2024-01-21T14:30:00",
    "last_updated": "2024-01-21T14:35:00",
    "total_questions": 3
  },
  "results": [
    {
      "id": 1,
      "question": "อาจารย์สมชาย",
      "answer": "👨‍🏫 [อาจารย์และบุคลากร]\n\n...",
      "contexts": [
        "อาจารย์ สมชาย ใจดี...",
        "ผศ.ดร.สมชาย ศรีสุข..."
      ],
      "num_contexts": 2,
      "classification": {
        "intent": "allpeople",
        "confidence": 9.0,
        "method": "rule_based",
        "reason": "High confidence from keyword matching"
      },
      "response_time": 2.35,
      "timestamp": "2024-01-21T14:30:15"
    }
  ]
}
```

---

## 🔧 ตัวเลือกเพิ่มเติม

### Resume จากคำถามที่หยุดไว้

```bash
python manual_evaluate_hybrid.py \
  --dataset my_dataset.json \
  --output results_hybrid.json \
  --start-from 5
```

เริ่มจากคำถาม ID 5 (ข้าม 1-4)

### ใช้ไฟล์ output ต่างกัน

```bash
# รันโมเดลที่ 1
python manual_evaluate_hybrid.py --dataset dataset.json --output results_model1.json

# รันโมเดลที่ 2
python manual_evaluate_hybrid.py --dataset dataset.json --output results_model2.json
```

---

## 💡 Tips

### 1. ดู Log แบบละเอียด

Log จะแสดงทุกขั้นตอน:
- 🔀 Hybrid Classification
- 📚 Context Retrieval
- 🤖 LLM Answer Generation
- 💾 JSON Saving

### 2. จัดการ Error

ถ้าเจอ error กลางคัน:
- สามารถ continue หรือ skip ได้
- ข้อมูลที่บันทึกแล้วปลอดภัย
- Resume จาก ID ที่หยุดได้

### 3. ทดสอบก่อนรันจริง

```bash
# ทดสอบ 2-3 คำถามก่อน (interactive mode)
python manual_evaluate_hybrid.py --interactive --output test.json

# ตรวจสอบว่า format ถูกต้อง
python manual_evaluate_hybrid.py --summary test.json

# ถ้าโอเค ค่อยรัน dataset เต็ม
python manual_evaluate_hybrid.py --dataset full_dataset.json
```

---

## 📈 Workflow แนะนำ

### สำหรับการวัดผล (10 คำถาม):

```bash
# 1. เตรียม dataset
nano dataset_10q.json

# 2. รันการประเมินผล
python manual_evaluate_hybrid.py \
  --dataset dataset_10q.json \
  --output results_hybrid.json

# 3. ตรวจสอบผลลัพธ์
python manual_evaluate_hybrid.py --summary results_hybrid.json

# 4. ตรวจสอบว่ามี contexts ครบ
grep "num_contexts" results_hybrid.json

# 5. พร้อมนำไปวัดผล RAGAS!
```

---

## ⚠️ ข้อควรระวัง

### 1. Contexts อาจใหญ่มาก

- contexts ไม่จำกัดความยาว
- ไฟล์ JSON อาจใหญ่ (หลาย MB)
- ปกติสำหรับ 10 คำถาม: 1-5 MB

### 2. ระยะเวลาการทำงาน

- แต่ละคำถาม: 1-3 วินาที
- 10 คำถาม: ประมาณ 1-2 นาที
- ขึ้นกับความซับซ้อนของคำถาม

### 3. API Cost

- ใช้ OpenRouter API (GPT-4o-mini)
- แต่ละคำถาม: ~0.001-0.005 USD
- 10 คำถาม: ~0.01-0.05 USD

---

## 🐛 Troubleshooting

### ปัญหา: Chatbot ไม่สามารถเริ่มต้นได้

```bash
# ตรวจสอบว่ามี .env และ API key
cat .env | grep OPENAI_API_KEY

# ตรวจสอบว่า AstraDB เชื่อมต่อได้
python -c "from automated_data_ingestion.core.astradb_manager import AstraDBManager; mgr = AstraDBManager(); print(mgr.list_collections())"
```

### ปัญหา: ไม่มี contexts

- ตรวจสอบว่า collection มีข้อมูล
- ตรวจสอบว่า retriever ทำงาน
- ลอง interactive mode เพื่อ debug

### ปัญหา: JSON format ผิด

```bash
# ตรวจสอบ JSON format
python -m json.tool results_hybrid.json

# ถ้า error ให้ลบไฟล์และรันใหม่
rm results_hybrid.json
python manual_evaluate_hybrid.py --dataset dataset.json
```

---

## 📞 ติดต่อ

มีปัญหาหรือข้อสงสัย?
- ดู log ละเอียดใน terminal
- ตรวจสอบ traceback
- ลอง interactive mode เพื่อ debug

---

## 🎉 เสร็จแล้ว!

หลังจากรันเสร็จ คุณจะได้:
- ✅ `results_hybrid.json` - ไฟล์ผลลัพธ์สมบูรณ์
- ✅ คำถาม + คำตอบ + contexts + classification
- ✅ พร้อมนำไปวัดผล RAGAS

ขั้นตอนถัดไป:
→ ใช้ `evaluate_from_json.py` เพื่อวัดผลด้วย RAGAS

