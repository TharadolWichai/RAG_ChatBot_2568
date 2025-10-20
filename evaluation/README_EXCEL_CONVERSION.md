# 📊 Excel to Test Questions Converter

## 📖 Overview

เครื่องมือสำหรับแปลงไฟล์ Excel ที่มีคำถามและคำตอบ (ground truth) เป็นไฟล์ `test_questions.json` สำหรับใช้กับ RAGAS evaluation

## 🚀 Quick Start

### 1. ติดตั้ง Dependencies

```bash
pip install pandas openpyxl
```

### 2. แปลงไฟล์ Excel

```bash
# วิธีที่ 1: แบบง่าย (สร้างไฟล์ใหม่)
python evaluation/excel_to_test_questions_complete.py "evaluate 50 qustion.xlsx"

# วิธีที่ 2: แทนที่ไฟล์เดิมทันที
python evaluation/excel_to_test_questions_complete.py "evaluate 50 qustion.xlsx" --replace

# วิธีที่ 3: กำหนดชื่อไฟล์ output
python evaluation/excel_to_test_questions_complete.py "myfile.xlsx" -o my_questions.json
```

## 📋 รูปแบบไฟล์ Excel

### ที่ต้องมี (Required)

ไฟล์ Excel ต้องมี **อย่างน้อย 1 column** สำหรับคำถาม:

| คำถาม (Question) | คำตอบจากเว็บ (Ground_truth) |
|-----------------|---------------------------|
| อาจารย์พุธษดี | ควรตอบข้อมูลเกี่ยวกับอาจารย์พุธษดี... |
| ติดต่อวิทยาลัย | เบอร์โทร: 043-202-555 อีเมล: ... |
| ทุนการศึกษา | มีทุนทั้งหมด 4 ประเภท... |

### ชื่อ Column ที่รองรับ

**สำหรับคำถาม:**
- `question`, `คำถาม`, `Question`, `คำถาม/question`

**สำหรับ ground truth:**
- `ground_truth`, `answer`, `คำตอบ`, `Ground Truth`, `คำตอบที่ถูกต้อง`, `เฉลย`

## 🤖 Auto Classification

Script จะจำแนกคำถามอัตโนมัติโดยใช้ keyword matching:

### Intent Types

| Intent | ตัวอย่างคำถาม |
|--------|-------------|
| `allpeople` | "อาจารย์พุธษดี", "ติดต่ออาจารย์" |
| `bsc_entrance` | "รับเข้าศึกษา", "รอบ Portfolio", "คะแนน TGAT" |
| `scholarship` | "ทุนการศึกษา", "ทุน ASEAN", "คุณสมบัติทุน" |
| `research` | "กลุ่มวิจัย AIDA", "งานวิจัย" |
| `digital_services` | "Web Hosting", "Virtual Machine" |
| `students` | "ลิงก์นักศึกษา", "ระบบโครงงาน" |
| `student_club` | "สโมสรนักศึกษา", "ประธานสโมสร" |
| `links` | "ลิงก์จองห้อง", "เว็บไซต์" |
| `contact` | "ติดต่อวิทยาลัย", "เบอร์โทร" |
| `news` | "ข่าว", "ประกาศ", "กิจกรรม" |
| `unknown` | คำถามที่ไม่ตรงกับ pattern ข้างต้น |

### Category Types

| Category | คำอธิบาย | ตัวอย่าง |
|----------|---------|---------|
| `simple` | คำถามสั้น ตรงไปตรงมา | "อาจารย์พุธษดี", "ติดต่อวิทยาลัย" |
| `complex` | คำถามยาว มีหลายส่วน | "อธิบายขั้นตอนการ...", "เปรียบเทียบ..." |
| `ambiguous` | คำถามคลุมเครือ | "มีบริการอะไรบ้าง", "ขอข้อมูลเกี่ยวกับ..." |
| `general` | คำถามทั่วไป |  |

### Difficulty Levels

| Difficulty | คำอธิบาย |
|-----------|---------|
| `easy` | คำถามง่าย เช่น ลิงก์, เบอร์, อีเมล |
| `medium` | คำถามปานกลาง (default) |
| `hard` | คำถามยาก เช่น วิเคราะห์, เปรียบเทียบ |

## 📄 Output Format

```json
{
  "test_questions": [
    {
      "id": 1,
      "question": "วิทยาลัยการคอมพิวเตอร์ มข. มีรอบการรับเข้าศึกษาระดับปริญญาตรีรอบไหนบ้าง",
      "expected_intent": "bsc_entrance",
      "ground_truth": "รอบที่ 1 Portfolio, รอบที่ 2 โควตาภาคตะวันออกเฉียงเหนือ, และรอบที่ 3 Admission",
      "category": "general",
      "difficulty": "medium"
    }
  ],
  "metadata": {
    "total_questions": 52,
    "categories": {
      "simple": 37,
      "general": 10,
      "complex": 5
    },
    "difficulty": {
      "easy": 2,
      "medium": 49,
      "hard": 1
    },
    "intents_coverage": {
      "unknown": 24,
      "bsc_entrance": 8,
      "scholarship": 5
    }
  }
}
```

## 🔧 Available Scripts

### 1. `excel_to_test_questions_complete.py` (แนะนำ) ⭐

**One-stop solution** - รวมทุกขั้นตอนในไฟล์เดียว

```bash
python evaluation/excel_to_test_questions_complete.py "your_file.xlsx"
```

**Features:**
- ✅ อ่าน Excel
- ✅ แปลงเป็น JSON
- ✅ Auto-classify intent
- ✅ สร้าง metadata
- ✅ รองรับ Windows encoding

### 2. `convert_excel_to_json.py`

แปลง Excel เป็น JSON (ยังไม่ classify)

```bash
python evaluation/convert_excel_to_json.py "your_file.xlsx"
```

### 3. `auto_classify_intent.py`

Classify intent สำหรับ JSON ที่มีอยู่แล้ว

```bash
python evaluation/auto_classify_intent.py "your_questions.json"
```

## 📊 ใช้กับ RAGAS Evaluation

### ขั้นตอนที่ 1: แปลง Excel

```bash
python evaluation/excel_to_test_questions_complete.py "evaluate 50 qustion.xlsx" --replace
```

หรือ

```bash
python evaluation/excel_to_test_questions_complete.py "evaluate 50 qustion.xlsx"
mv test_questions_new.json test_questions.json
```

### ขั้นตอนที่ 2: รัน Evaluation

```bash
cd evaluation
python evaluate_chatbots.py
```

เลือกจำนวนคำถามและจำนวน contexts ตามต้องการ

## 🎯 Tips & Best Practices

### 1. การเตรียมไฟล์ Excel

✅ **ควรทำ:**
- ใช้ชื่อ column ที่ชัดเจน (เช่น "คำถาม", "คำตอบ")
- ลบ row ว่างออก (หรือปล่อยไว้ script จะข้ามให้)
- เขียน ground truth ที่ครอบคลุมและชัดเจน

❌ **ไม่ควรทำ:**
- ใช้ชื่อ column แปลกๆ ที่ไม่เกี่ยวกับคำถาม/คำตอบ
- มี merged cells
- มีหลาย sheet (จะอ่านแค่ sheet แรก)

### 2. การปรับแต่ง Intent Classification

ถ้า auto-classification ไม่ถูกต้อง มี 2 วิธี:

**วิธีที่ 1: แก้ไข JSON โดยตรง**
```bash
notepad test_questions_new.json
# แก้ไข expected_intent ด้วยมือ
```

**วิธีที่ 2: เพิ่ม keywords ใน script**
- เปิดไฟล์ `excel_to_test_questions_complete.py`
- แก้ไข `INTENT_KEYWORDS` dictionary
- เพิ่ม regex patterns ใหม่

ตัวอย่าง:
```python
INTENT_KEYWORDS = {
    "allpeople": [
        r'อาจารย์',
        r'ผศ\.',
        r'บุคลากร',
        # เพิ่มของคุณที่นี่
        r'преподаватель',  # ตัวอย่าง
    ]
}
```

### 3. การตรวจสอบผลลัพธ์

หลังจากแปลงแล้ว ควรตรวจสอบ:

1. **จำนวนคำถาม**
   ```bash
   # Windows
   type test_questions_new.json | find /c "\"id\""
   
   # Linux/Mac
   grep -c '"id"' test_questions_new.json
   ```

2. **Intent distribution**
   - ดูใน terminal output หลังรัน script
   - ตรวจสอบว่า `unknown` ไม่เกิน 30-40%

3. **Ground truth quality**
   - เปิดไฟล์ดู 5-10 คำถามแรก
   - ตรวจสอบว่า ground truth ตอบคำถามได้จริง

## 🐛 Troubleshooting

### ปัญหา: `UnicodeEncodeError`

**สาเหตุ:** Windows encoding issue

**วิธีแก้:**
- Script มี encoding fix อยู่แล้ว
- ถ้ายังมีปัญหา ให้รันผ่าน PowerShell แทน CMD

### ปัญหา: `ModuleNotFoundError: No module named 'pandas'`

**วิธีแก้:**
```bash
pip install pandas openpyxl
```

### ปัญหา: `openpyxl` not found

**วิธีแก้:**
```bash
pip install openpyxl
```

### ปัญหา: คอลัมน์ไม่ถูกต้อง

**วิธีแก้:**
- เปิด Excel file
- เปลี่ยนชื่อ column เป็น "คำถาม" และ "คำตอบ"
- หรือแก้ไข `column_mapping` logic ใน script

### ปัญหา: Intent classification ผิด

**วิธีแก้:**
1. แก้ไข JSON file ด้วยมือ
2. หรือปรับ `INTENT_KEYWORDS` ใน script
3. รัน classification ใหม่:
   ```bash
   python evaluation/auto_classify_intent.py test_questions_new.json
   ```

## 📚 Example Workflow

```bash
# 1. แปลง Excel
python evaluation/excel_to_test_questions_complete.py "evaluate 50 qustion.xlsx"

# 2. ตรวจสอบผลลัพธ์
type test_questions_new.json | more  # Windows
# หรือ
cat test_questions_new.json | less   # Linux/Mac

# 3. แก้ไขถ้าจำเป็น
notepad test_questions_new.json

# 4. ใช้กับ evaluation
mv test_questions_new.json test_questions.json
python evaluate_chatbots.py
```

## 🎓 Advanced Usage

### Custom Intent Keywords

สร้างไฟล์ `custom_keywords.json`:

```json
{
  "my_custom_intent": [
    "keyword1",
    "keyword2",
    "regex_pattern.*here"
  ]
}
```

จากนั้นแก้ไข script ให้โหลดจากไฟล์นี้

### Batch Conversion

สร้าง batch script สำหรับแปลงหลายไฟล์:

```bash
# convert_all.bat (Windows)
for %%f in (*.xlsx) do (
    python evaluation/excel_to_test_questions_complete.py "%%f" -o "%%~nf.json"
)
```

```bash
# convert_all.sh (Linux/Mac)
for file in *.xlsx; do
    python evaluation/excel_to_test_questions_complete.py "$file" -o "${file%.xlsx}.json"
done
```

## 📞 Support

หากมีปัญหาหรือข้อสงสัย:

1. ตรวจสอบ [README_EVALUATION.md](./README_EVALUATION.md) สำหรับข้อมูล RAGAS
2. ตรวจสอบ [TROUBLESHOOTING_API.md](./TROUBLESHOOTING_API.md) สำหรับปัญหา API
3. ดู [CONTEXT_PRECISION_FIX.md](./CONTEXT_PRECISION_FIX.md) และ [REDUCE_CONTEXTS_FIX.md](./REDUCE_CONTEXTS_FIX.md)

## 📝 License

MIT License - ใช้ได้อย่างอิสระ

---

**Last Updated:** October 19, 2025

