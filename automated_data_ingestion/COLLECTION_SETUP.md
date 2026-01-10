# Collection Setup Guide

## 📋 วิธีสร้าง Collection ใน AstraDB

เนื่องจากบาง Plan ของ AstraDB ไม่อนุญาตให้สร้าง Collection ผ่านโค้ด จึงต้องสร้างผ่าน AstraDB UI

### ขั้นตอนการสร้าง Collection

1. **เข้าสู่ AstraDB Console**
   - ไปที่ https://astra.datastax.com
   - Login เข้าสู่ระบบ

2. **เลือก Database**
   - คลิกที่ Database ที่ต้องการใช้งาน

3. **สร้าง Collection**
   - คลิกที่แท็บ "Collections" หรือ "CQL Console"
   - คลิกปุ่ม "Create Collection" หรือ "Add Collection"

4. **ตั้งค่า Collection**
   - **Collection Name**: ตั้งชื่อตามที่ต้องการ (เช่น `allpeople_embedding`)
   - **Vector Dimension**: `384` (สำหรับ sentence-transformers/all-MiniLM-L6-v2)
   - **Similarity Metric**: `cosine`

5. **ยืนยันการสร้าง**
   - คลิก "Create" หรือ "Confirm"
   - รอให้ระบบสร้าง collection เสร็จ

### ตัวอย่าง Collection Names ที่แนะนำ

- `allpeople_embedding` - สำหรับข้อมูลบุคลากร
- `students_embedding` - สำหรับข้อมูลนักศึกษา
- `news_embedding` - สำหรับข้อมูลข่าว
- `scholarship_embedding` - สำหรับข้อมูลทุนการศึกษา
- `services_embedding` - สำหรับข้อมูลบริการ

### ⚠️ หมายเหตุ

- แต่ละ Database (Serverless) สามารถมี Collection ได้ประมาณ 10 collections
- ถ้าเห็น error "Collection Limit Reached" ต้องลบ collection เก่าก่อน
- หลังจากสร้าง collection แล้ว ระบบ automated data ingestion จะใช้งานได้ทันที

### 🔧 Troubleshooting

**Q: เห็น error "Collection does not exist"**
A: ตรวจสอบว่าสร้าง collection ใน database และ keyspace ที่ถูกต้องแล้วหรือยัง

**Q: ไม่สามารถสร้าง collection ผ่าน UI**
A: ตรวจสอบว่า:
- Database plan รองรับการสร้าง collection หรือไม่
- มีสิทธิ์ในการสร้าง collection หรือไม่
- ยังไม่เกินจำนวน collection limit

**Q: ต้องการให้ระบบสร้าง collection อัตโนมัติ**
A: ปัจจุบันระบบตั้งค่าให้ไม่สร้างอัตโนมัติ เพื่อหลีกเลี่ยง permission issues
   สามารถสร้าง collection ผ่าน UI แล้วใช้งานได้เลย

