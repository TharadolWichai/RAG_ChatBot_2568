# Batch Processing Guide

## ภาพรวม

Batch Processing คือฟีเจอร์ที่ช่วยให้คุณสามารถดึงข้อมูลจาก List API และสร้าง jobs อัตโนมัติสำหรับแต่ละรายการ

### Use Case ตัวอย่าง

- **กลุ่มวิจัย (Research Groups)**: 
  - List API: `https://api.computing.kku.ac.th/api/v1/page/getPageMappingBySlug/research`
  - Detail API Pattern: `https://api.computing.kku.ac.th/api/v1/page/getPageMappingBySlug/{slug}`
  - Detail URL Pattern: `https://computing.kku.ac.th/{slug}`

## วิธีใช้งาน

### ขั้นตอนที่ 1: เปิด Dashboard

รัน Streamlit dashboard:
```bash
streamlit run automated_data_ingestion/dashboard/app.py
```

### ขั้นตอนที่ 2: สร้าง Batch Job

1. เปิด tab **"📝 Create New Job"**

2. กรอกข้อมูลพื้นฐาน:
   - **ชื่อ Job**: ตัวอย่าง "Research Groups Batch"
   - **Collection Name**: ตัวอย่าง "researchgroup_embeddings"
   - **Extraction Prompt**: Prompt สำหรับ extract ข้อมูลจากแต่ละรายการ
     ```
     ดึงข้อมูลรายละเอียดของกลุ่มวิจัยทั้งหมด รวมถึงชื่อ, คำอธิบาย, งานวิจัย, และข้อมูลอื่นๆ
     ```

3. เปิด **"🔄 Batch Processing"** expander

4. ติ๊ก **"เปิดโหมด Batch Processing"**

5. กรอกข้อมูล Batch:
   - **List API URL**: 
     ```
     https://api.computing.kku.ac.th/api/v1/page/getPageMappingBySlug/research
     ```
   - **Detail URL Pattern** (optional):
     ```
     https://computing.kku.ac.th/{slug}
     ```
   - **Detail API Pattern** (optional):
     ```
     https://api.computing.kku.ac.th/api/v1/page/getPageMappingBySlug/{slug}
     ```

6. คลิก **"🚀 Create & Run Job"**

### ขั้นตอนที่ 3: ระบบจะทำงานอัตโนมัติ

1. **ดึง List จาก API**: ระบบจะดึงข้อมูลจาก List API
2. **Extract Slugs/URLs**: ใช้ LLM หรือ rule-based method เพื่อ extract slugs/URLs
3. **สร้าง Jobs**: สร้าง job แยกสำหรับแต่ละ slug/URL
4. **รัน Jobs**: รัน jobs ทั้งหมดทีละตัว
5. **สรุปผล**: แสดงสรุปผลการทำงาน

## ตัวอย่างการใช้งาน

### ตัวอย่าง 1: กลุ่มวิจัย (Research Groups)

**List API**: 
```
https://api.computing.kku.ac.th/api/v1/page/getPageMappingBySlug/research
```

**Detail API Pattern**:
```
https://api.computing.kku.ac.th/api/v1/page/getPageMappingBySlug/{slug}
```

**Detail URL Pattern**:
```
https://computing.kku.ac.th/{slug}
```

**Extraction Prompt**:
```
ดึงข้อมูลรายละเอียดของกลุ่มวิจัย รวมถึงชื่อ, คำอธิบาย, งานวิจัย, สมาชิก, และข้อมูลอื่นๆ
```

### ตัวอย่าง 2: ข่าว (News Articles)

**List API**: 
```
https://api.computing.kku.ac.th/api/v1/article/getArticles
```

**Detail API Pattern**:
```
https://api.computing.kku.ac.th/api/v1/article/getArticleBySlug/{slug}
```

**Detail URL Pattern**:
```
https://computing.kku.ac.th/news/{slug}
```

**Extraction Prompt**:
```
ดึงข้อมูลข่าวทั้งหมด รวมถึงหัวข้อ, เนื้อหา, วันที่เผยแพร่, และข้อมูลอื่นๆ
```

## การทำงานของระบบ

### 1. List Processing

ระบบจะ:
- ดึง JSON จาก List API
- ใช้ LLM หรือ rule-based method เพื่อ extract slugs/URLs
- รองรับการ extract จาก fields ต่างๆ เช่น:
  - `slug`
  - `url`
  - `href`
  - และอื่นๆ

### 2. Job Creation

สำหรับแต่ละ slug/URL ที่ extract ได้:
- สร้าง job config ใหม่
- ใช้ patterns เพื่อสร้าง URL และ API URL
- ใช้ extraction prompt เดียวกัน

### 3. Job Execution

รัน jobs ทีละตัว:
- ดึงข้อมูลจาก URL/API
- Extract ข้อมูลตาม prompt
- บันทึกลง AstraDB

### 4. Results Summary

แสดงสรุป:
- จำนวน items ทั้งหมด
- จำนวนที่สำเร็จ
- จำนวนที่ล้มเหลว
- จำนวน documents ที่ process, insert, skip

## Tips & Best Practices

1. **List API**: ตรวจสอบว่า API ส่งกลับ JSON ที่มี slugs/URLs
2. **Patterns**: ใช้ `{slug}` เป็น placeholder ใน patterns
3. **Extraction Prompt**: เขียน prompt ให้ชัดเจนว่าต้องการข้อมูลอะไร
4. **Collection Name**: ใช้ collection เดียวกันสำหรับทุก items ใน batch
5. **Hash Keys**: กำหนด hash_keys เพื่อป้องกันข้อมูลซ้ำ

## Troubleshooting

### ไม่พบ items ใน List API
- ตรวจสอบว่า API URL ถูกต้อง
- ตรวจสอบโครงสร้าง JSON จาก API
- ลองใช้ rule-based extraction แทน LLM

### Jobs ล้มเหลว
- ตรวจสอบ Detail URL/API Patterns
- ตรวจสอบว่ามีข้อมูลใน detail URLs
- ดู error messages ใน console

### ข้อมูลซ้ำ
- กำหนด `hash_keys` ใน Advanced Options
- ตรวจสอบ `metadata_filter`

## ข้อจำกัด

- Batch processing จะรัน jobs ทีละตัว (sequential)
- ถ้ามี items มาก อาจใช้เวลานาน
- ต้องมี OpenAI API key สำหรับ LLM extraction

## Support

ถ้ามีปัญหาหรือคำถาม:
1. ตรวจสอบ logs ใน console
2. ดู error messages ใน dashboard
3. ตรวจสอบ API responses

