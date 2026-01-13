# 🧪 No Keywords Test - ทดสอบ Rule-Based Strict Mode

## 📁 ไฟล์ที่เกี่ยวข้อง

- **`test_forRetriver_no_keywords.json`**: ชุดคำถามที่ลบคีย์เวิร์ด 8 ข้อ (21%)
- **`evaluate_retriever_10q.py`**: สคริปต์ evaluation ที่รองรับการเลือกไฟล์ทดสอบ

---

## 🎯 วัตถุประสงค์

ทดสอบ**ความแตกต่างระหว่าง 3 Chatbot Versions** เมื่อคำถามไม่มีคีย์เวิร์ดที่ตรงกับ Rule-Based classifier:

### **🔬 สมมติฐาน:**

| Chatbot Version | คาดการณ์ผลลัพธ์ (Strict Mode) | เหตุผล |
|-----------------|-------------------------------|--------|
| **Rule-Based** | ❌ **ตอบไม่ได้ 8 ข้อ** (21%) | ไม่มีคีย์เวิร์ด → `unknown` → Strict Mode ไม่ fallback |
| **LLM-Based** | ✅ **ตอบได้ทุกข้อ** (100%) | LLM เข้าใจความหมาย ไม่ต้องอาศัยคีย์เวิร์ด |
| **Hybrid** | ✅ **ตอบได้ทุกข้อ** (100%) | Rule-Based ล้มเหลว → fallback เป็น LLM |

---

## 📝 รายการคำถามที่แก้ไข (8 ข้อ จาก 38 ข้อ)

| ID | คำถามเดิม (มีคีย์เวิร์ด) | คำถามใหม่ (ไม่มีคีย์เวิร์ด) | คีย์เวิร์ดที่ลบ | Intent |
|----|---------------------------|------------------------------|----------------|--------|
| 4 | "อีเมลของอาจารย์ธนพล" | "ช่องทางอิเล็กทรอนิกส์ติดต่อคุณธนพล" | `อีเมล`, `อาจารย์` | allpeople |
| 6 | "ช่องทางติดต่อมหาวิทยาลัย" | "วิธีการเข้าถึงหน่วยงานมหาวิทยาลัย" | `ติดต่อ`, `ช่องทาง` | contact |
| 9 | "จองห้องประชุม" | "ขอใช้ห้องประชุม" | `จอง` | links |
| 12 | "แบบฟอร์มขอเปลี่ยนแปลงเกรด" | "เอกสารร้องขอแก้ไขคะแนน" | `แบบฟอร์ม`, `ฟอร์ม` | links |
| 14 | "รายละเอียดของทุนการศึกษาทั้งหมด" | "ข้อมูลการสนับสนุนทางการเงินทั้งหมด" | `ทุน`, `ทุนการศึกษา` | scholarship |
| 21 | "ใครเป็นประธานของสโมสรนักศึกษา" | "หัวหน้าองค์กรนิสิตคือใคร" | `ประธาน`, `สโมสร` | student_club |
| 22 | "คู่มือสหกิจศึกษา" | "หนังสือแนะนำการทำงานร่วมกับบริษัท" | `สหกิจ`, `คู่มือ` | students |
| 29 | "การรับเข้าปริญญาตรี" | "เข้าศึกษาต่อระดับปริญญาตรีได้อย่างไร" | `รับเข้า`, `การรับเข้า` | bsc_entrance |
| 36 | "สาขาวิชาของปริญญาเอก" | "สาขาที่เปิดสอนระดับดุษฎีบัณฑิต" | `ปริญญาเอก`, `ป.เอก` | graduate |

---

## 🚀 วิธีใช้งาน

### **1. รันสคริปต์ evaluation**

```bash
cd evaluation
python evaluate_retriever_10q.py
```

### **2. เลือก Test Dataset**

```
📝 Select Test Dataset:
   1. test_forRetriver.json (มีคีย์เวิร์ด - ทดสอบปกติ) ⭐
   2. test_forRetriver_no_keywords.json (ไม่มีคีย์เวิร์ด 8 ข้อ - ทดสอบ Strict Mode)

Select option (1-2) [default: 1]: 2  ← เลือก 2
```

### **3. เลือกจำนวน Contexts**

```
📚 Select number of contexts to send to RAGAS:
   1. Top 5 contexts
   2. Top 10 contexts
   3. All contexts (no limit)

Select option (1-3) [default: 3]: 1  ← เลือกตามต้องการ
```

### **4. เลือก Rule-Based Mode**

```
🔒 Select Rule-Based Mode:
   1. Normal Mode (with multi-agent fallback) - Production Mode
   2. Strict Mode (no fallback) - Evaluation Mode ⭐

Select option (1-2) [default: 2]: 2  ← เลือก Strict Mode เพื่อดูความแตกต่าง
```

---

## 📊 ผลลัพธ์ที่คาดหวัง

### **Strict Mode (เลือก option 2):**

```
📊 COMPARISON SUMMARY
==================================================================================
Metric                   | Rule-Based      | LLM-Based       | Hybrid
----------------------------------------------------------------------------------
📈 Main Metrics
Faithfulness             |      0.8xxx     |      1.0000     |      1.0000
Context Precision        |      0.6xxx     |      0.7xxx     |      0.7xxx
Context Recall           |      0.7xxx     |      1.0000     |      1.0000
----------------------------------------------------------------------------------
⚡ Performance & Retriever Stats
Avg Response Time (s)    |         xxx     |         xxx     |         xxx
Errors                   |           8     |           0     |           0   ← Rule-Based ตอบไม่ได้ 8 ข้อ
==================================================================================

🏆 Best Performers:
   Faithfulness              : LLM-Based (1.0000)    ← ชนะ
   Context Precision         : Hybrid (0.7xxx)
   Context Recall            : LLM-Based (1.0000)    ← ชนะ
   Fastest Response          : Rule-Based (xxx)
```

### **Normal Mode (เลือก option 1):**
Rule-Based จะใช้ multi-agent search fallback → ตอบได้ทุกข้อ (แต่ช้ามาก)

---

## 🔍 คีย์เวิร์ดที่ลบออกจาก `main_unified_chatbot.py`

### **Intent Classifier Keywords (ตัวอย่าง):**

```python
intent_patterns = {
    "allpeople": {
        "keywords": ["อาจารย์", "อ.", "ดร.", "ผู้ช่วย", "บุคลากร", ...],
        ...
    },
    "contact": {
        "keywords": ["ติดต่อ", "โทร", "อีเมล", "เบอร์", "ที่อยู่", ...],
        ...
    },
    "links": {
        "keywords": ["ลิงก์", "ระบบ", "จอง", "แบบฟอร์ม", "ดาวน์โหลด", ...],
        ...
    },
    "scholarship": {
        "keywords": ["ทุน", "ทุนการศึกษา", "ทุนวิจัย", "scholarship", ...],
        ...
    },
    # ... และอื่นๆ
}
```

---

## 💡 สิ่งที่เรียนรู้จากการทดสอบนี้

### **1. Rule-Based Limitations:**
- ❌ **ขึ้นอยู่กับคีย์เวิร์ดมาก** → ถ้าไม่มีคีย์เวิร์ดที่กำหนด = ตอบไม่ได้ (Strict Mode)
- ⚠️ **Fallback ช่วยได้ แต่ช้า** → multi-agent search ต้องค้นหาทุก retriever (Normal Mode)

### **2. LLM-Based Advantages:**
- ✅ **เข้าใจความหมาย** → ไม่ต้องอาศัยคีย์เวิร์ด
- ✅ **Flexible** → ตอบได้แม้คำถามจะแตกต่างจากที่กำหนด
- ⚠️ **แต่มีค่าใช้จ่าย API**

### **3. Hybrid Best of Both Worlds:**
- ✅ **ประหยัด** → ใช้ Rule-Based ก่อน (ฟรี)
- ✅ **Fallback เป็น LLM** → เมื่อ Rule-Based ไม่แน่ใจ
- ✅ **Balance ระหว่างความเร็วและความแม่นยำ**

---

## 📌 หมายเหตุ

- ไฟล์นี้ใช้สำหรับ**ทดสอบ Strict Mode เท่านั้น**
- สำหรับการใช้งานจริง (Production) แนะนำ **Hybrid Mode** เพราะ balance ดีที่สุด
- คำถาม 30 ข้อที่เหลือ (79%) ยังคงมีคีย์เวิร์ดเหมือนเดิม เพื่อให้ Rule-Based ทำงานได้บ้าง

---

## 🎓 ข้อควรรู้

### **Strict Mode vs Normal Mode:**

| Feature | Strict Mode | Normal Mode |
|---------|-------------|-------------|
| Rule-Based ไม่เจอคีย์เวิร์ด | ❌ ตอบว่า "Cannot answer" | ✅ ใช้ multi-agent search fallback |
| เวลาตอบคำถาม | ⚡ เร็ว (ไม่ fallback) | 🐌 ช้า (ต้องค้นหาทุก retriever) |
| เหมาะสำหรับ | 🔬 Evaluation, Testing | 🏭 Production, Real use |
| แสดงข้อจำกัดของ Rule-Based | ✅ ชัดเจน | ❌ ซ่อนอยู่ (fallback ช่วย) |

---

## 📈 File Output

เมื่อรันเสร็จ จะได้ไฟล์ JSON:

```
evaluation/retriever_eval_10q_5ctx_strict_YYYYMMDD_HHMMSS.json
                            ↑     ↑      ↑
                      contexts  mode   timestamp
```

- `_5ctx` = ใช้ top 5 contexts
- `_strict` = Strict Mode (ไม่ fallback)
- `_normal` = Normal Mode (มี fallback)

---

## ✅ Checklist สำหรับการทดสอบ

- [ ] เลือก Test Dataset: `test_forRetriver_no_keywords.json`
- [ ] เลือก Contexts: Top 5 หรือ Top 10
- [ ] **เลือก Strict Mode** (option 2) เพื่อดูความแตกต่างชัดเจน
- [ ] บันทึกผลลัพธ์เป็น JSON
- [ ] เปรียบเทียบผลลัพธ์ระหว่าง Rule-Based, LLM, Hybrid
- [ ] สังเกต Errors ของ Rule-Based (ควรมี 8 errors ใน Strict Mode)

---

## 🔗 Related Files

- [`test_forRetriver.json`](./test_forRetriver.json) - ชุดคำถามปกติ (มีคีย์เวิร์ด)
- [`evaluate_retriever_10q.py`](./evaluate_retriever_10q.py) - สคริปต์ evaluation
- [`main_unified_chatbot.py`](../main_app/main_unified_chatbot.py) - Rule-Based classifier
- [`README_RETRIEVER_TEST.md`](./README_RETRIEVER_TEST.md) - คู่มือการใช้งาน evaluation

---

**Last Updated:** 2025-01-20  
**Purpose:** Testing Rule-Based Strict Mode with NO keyword matching (21% modified questions)

