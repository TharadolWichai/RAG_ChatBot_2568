# Quick Start Guide

## 🚀 เริ่มใช้งานใน 5 นาที

### ขั้นตอนที่ 1: ติดตั้ง Dependencies

```bash
pip install streamlit langchain langchain-openai langchain-community sentence-transformers astrapy beautifulsoup4 requests selenium webdriver-manager python-dotenv
```

หรือใช้ไฟล์ requirements:

```bash
cd automated_data_ingestion
pip install -r requirements.txt
```

### ขั้นตอนที่ 2: ตั้งค่า Environment Variables

ตรวจสอบว่าไฟล์ `.env` ใน root directory มีการตั้งค่าดังนี้:

```env
ASTRA_DB_APPLICATION_TOKEN=your_token_here
ASTRA_DB_API_ENDPOINT=your_endpoint_here
ASTRA_DB_KEYSPACE=default_keyspace
OPENAI_API_KEY=your_openai_key_here  # Optional แต่แนะนำให้ใช้
```

### ขั้นตอนที่ 3: รัน Dashboard

```bash
streamlit run automated_data_ingestion/dashboard/app.py
```

Dashboard จะเปิดใน browser อัตโนมัติ

### ขั้นตอนที่ 4: สร้าง Job แรก

1. **กรอกข้อมูลพื้นฐาน**:
   - ชื่อ Job: `Test Students Page`
   - URL: `https://computing.kku.ac.th/students`
   - Collection Name: `test_students_embedding`
   - Extraction Prompt: 
     ```
     ดึงข้อมูลลิงก์ทั้งหมดที่มีอยู่ในหน้าเว็บ
     สำหรับแต่ละลิงก์ให้เก็บชื่อลิงก์และ URL
     ```

2. **กด "Create & Run Job"**

3. **รอผลลัพธ์** - ระบบจะแสดงสถานะและจำนวน documents ที่เพิ่มเข้าไป

## 📝 ตัวอย่าง Extraction Prompts

### สำหรับหน้าเว็บที่มีลิงก์
```
ดึงข้อมูลลิงก์ทั้งหมดที่มีอยู่ในหน้าเว็บ
สำหรับแต่ละลิงก์ให้เก็บ:
- ชื่อลิงก์ (text)
- URL
- คำสำคัญที่เกี่ยวข้อง
```

### สำหรับหน้าเว็บที่มีเนื้อหา
```
ดึงเนื้อหาหลักจากหน้าเว็บ
รวมถึงหัวข้อ ย่อหน้า และรายการ
ลบ navigation และ footer ออก
```

### สำหรับ API Response
```
ดึงข้อมูลจาก JSON response
สำหรับแต่ละ item ให้เก็บ title, description, และ url
```

## 🎯 Tips

1. **ใช้ Selenium**: เปิดใช้งานถ้าหน้าเว็บใช้ JavaScript
2. **Wait Time**: เพิ่มถ้าหน้าเว็บโหลดช้า
3. **Metadata Category**: ใช้เพื่อแยกข้อมูลใน collection เดียวกัน
4. **Hash Keys**: ใช้เพื่อป้องกันข้อมูลซ้ำ (เช่น `url`, `title`)

## ❓ Troubleshooting

### Dashboard ไม่เปิด
- ตรวจสอบว่า streamlit ติดตั้งแล้ว: `pip install streamlit`
- ตรวจสอบว่าไม่มี port 8501 ถูกใช้งานอยู่

### Connection Error
- ตรวจสอบ `.env` file ว่ามี ASTRA_DB credentials ครบ
- ตรวจสอบ network connection

### Selenium Error
- ติดตั้ง ChromeDriver หรือใช้ webdriver-manager
- `pip install webdriver-manager`

## 📚 ต่อไป

อ่าน [README.md](README.md) เพื่อดูรายละเอียดเพิ่มเติม

