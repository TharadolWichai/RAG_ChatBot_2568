# Incremental Indexing System

## 📋 ภาพรวม

ระบบ **Incremental Indexing** ช่วยให้การดึงข้อมูลและอัพเดทข้อมูลใน AstraDB มีประสิทธิภาพมากขึ้น โดยจะ:

- ✅ **ดึงเฉพาะข้อมูลใหม่** - ไม่ต้องดึงข้อมูลซ้ำทั้งหมด
- ✅ **ตรวจสอบความซ้ำซ้อน** - ใช้ content hash ในการตรวจสอบ
- ✅ **ประหยัดเวลาและทรัพยากร** - ลดการ insert ข้อมูลซ้ำ
- ✅ **รักษาข้อมูลเดิม** - ไม่ลบข้อมูลที่มีอยู่แล้ว (ถ้าไม่ต้องการ)

---

## 🔧 วิธีการทำงาน

### 1. Content Hash
ระบบจะสร้าง **SHA-256 hash** จากเนื้อหาและ metadata สำคัญของแต่ละ document:

```python
content_hash = SHA256(content + metadata_keys)
```

### 2. การตรวจสอบความซ้ำซ้อน
- ดึง content_hash ของเอกสารทั้งหมดที่มีอยู่ใน collection
- เปรียบเทียบกับ hash ของเอกสารใหม่
- ถ้า hash ตรงกัน = เอกสารซ้ำ → **Skip**
- ถ้า hash ไม่ตรงกัน = เอกสารใหม่ → **Insert**

### 3. Metadata Filter
ใช้ filter เพื่อดึงเฉพาะเอกสารประเภทเดียวกัน:

```python
metadata_filter = {"type": "news"}  # ดึงเฉพาะข่าว
```

### 4. Hash Keys
กำหนด metadata keys ที่ต้องการรวมในการสร้าง hash:

```python
hash_keys = ["article_id", "slug", "published_at"]
```

---

## 📦 การใช้งาน

### วิธีที่ 1: ใช้ Helper Function

```python
from incremental_utils import enable_incremental_mode

# เตรียม documents
documents = [...]  # List of LangChain Document objects

# เรียกใช้ incremental indexing
stats = enable_incremental_mode(
    collection=collection,              # AstraDB collection
    embedding_model=embedding,          # Embedding model
    new_documents=documents,            # Documents ใหม่
    metadata_filter={"type": "news"},   # Filter เอกสารเดิม
    hash_keys=["article_id", "slug"],   # Keys สำหรับ hash
    delete_missing=False                # ลบเอกสารเก่าหรือไม่
)

# ดูผลลัพธ์
print(f"Inserted: {stats['inserted']}")
print(f"Skipped: {stats['skipped']}")
print(f"Updated: {stats['updated']}")
print(f"Deleted: {stats['deleted']}")
```

### วิธีที่ 2: ใช้ Class โดยตรง

```python
from incremental_utils import IncrementalIndexManager

# สร้าง manager
manager = IncrementalIndexManager(collection, embedding_model)

# ประมวลผล
manager.process_incremental_update(
    new_documents=documents,
    metadata_filter={"type": "news"},
    hash_keys=["article_id", "slug"],
    delete_missing=False
)

# ดูสถิติ
stats = manager.get_stats()
```

---

## 🎯 ตัวอย่างการใช้งานในแต่ละไฟล์

### 1. News Data (`topic_news_data.py`, `ืnews_data.py`)

```python
stats = enable_incremental_mode(
    collection=collection,
    embedding_model=embedding,
    new_documents=docs,
    metadata_filter={"type": "news"},
    hash_keys=["article_id", "slug", "published_at"],
    delete_missing=False
)
```

**Hash Keys:**
- `article_id` - ID ของข่าว
- `slug` - URL slug
- `published_at` - วันที่เผยแพร่

---

### 2. Scholarship Data (`scholarship_data.py`)

```python
stats = enable_incremental_mode(
    collection=collection,
    embedding_model=embedding,
    new_documents=all_documents,
    metadata_filter={"type": "scholarship_main"},
    hash_keys=["scholarship_id", "scholarship_type", "section"],
    delete_missing=False
)
```

**Hash Keys:**
- `scholarship_id` - ID ของทุน
- `scholarship_type` - ประเภททุน
- `section` - ส่วนของข้อมูล (overview, detail)

---

### 3. Links & Services (`links_data.py`, `students_data.py`)

```python
stats = enable_incremental_mode(
    collection=collection,
    embedding_model=embedding,
    new_documents=docs,
    metadata_filter={"category": "links"},
    hash_keys=["link_text", "url"],
    delete_missing=False
)
```

**Hash Keys:**
- `link_text` - ชื่อลิงก์
- `url` - URL ของลิงก์

---

### 4. Graduate Programs (`graduate_data.py`)

```python
stats = enable_incremental_mode(
    collection=collection,
    embedding_model=embedding_model,
    new_documents=all_documents,
    metadata_filter={"type": "graduate"},
    hash_keys=["category", "program_name"],
    delete_missing=False
)
```

**Hash Keys:**
- `category` - หมวดหมู่ (master, phd, graduate)
- `program_name` - ชื่อหลักสูตร

---

### 5. Contact Info (`contact_data.py`)

```python
stats = enable_incremental_mode(
    collection=collection,
    embedding_model=embedding_model,
    new_documents=docs,
    metadata_filter={"type": "contact_info"},
    hash_keys=["source", "type"],
    delete_missing=False
)
```

**Hash Keys:**
- `source` - แหล่งที่มา (URL)
- `type` - ประเภทข้อมูล

---

### 6. Digital Services (`digital_services_data.py`)

```python
stats = enable_incremental_mode(
    collection=collection,
    embedding_model=embedding,
    new_documents=all_documents,
    metadata_filter={"type": "digital_service"},
    hash_keys=["service_name", "category"],
    delete_missing=False
)
```

**Hash Keys:**
- `service_name` - ชื่อบริการ
- `category` - หมวดหมู่บริการ

---

### 7. Student Clubs (`student_club_data.py`)

```python
stats = enable_incremental_mode(
    collection=collection,
    embedding_model=embedding,
    new_documents=docs_for_incremental,
    metadata_filter={"type": "student_club"},
    hash_keys=["club_name", "club_id"],
    delete_missing=False
)
```

**Hash Keys:**
- `club_name` - ชื่อชมรม
- `club_id` - ID ของชมรม

---

### 8. All People (`allpeople_data.py`)

```python
stats = enable_incremental_mode(
    collection=collection,
    embedding_model=embedding,
    new_documents=docs,
    metadata_filter={"type": "allpeople"},
    hash_keys=["name", "position"],
    delete_missing=False
)
```

**Hash Keys:**
- `name` - ชื่อบุคลากร
- `position` - ตำแหน่ง

---

## 📊 ผลลัพธ์ที่ได้

เมื่อรันเสร็จจะได้สรุปผลดังนี้:

```
============================================================
📊 สรุปผลการ Incremental Indexing
============================================================
✅ Insert ใหม่:             15 รายการ
🔄 Update:                   0 รายการ
⏭️  Skip (มีอยู่แล้ว):      85 รายการ
🗑️  Delete (หายไป):         0 รายการ
============================================================
```

---

## ⚙️ Parameters

### `metadata_filter` (Dict)
- Filter สำหรับดึงเอกสารเดิมที่ต้องการเปรียบเทียบ
- ตัวอย่าง: `{"type": "news"}`, `{"category": "links"}`

### `hash_keys` (List[str])
- Metadata keys ที่ต้องการรวมในการสร้าง hash
- ควรเลือก keys ที่ทำให้แต่ละเอกสารมี hash ไม่ซ้ำกัน
- ตัวอย่าง: `["article_id", "slug"]`, `["name", "position"]`

### `delete_missing` (bool)
- `False` (default): ไม่ลบเอกสารเก่าที่ไม่มีในข้อมูลใหม่
- `True`: ลบเอกสารเก่าที่ไม่มีในข้อมูลใหม่ (ระวังใช้!)

---

## 🔍 การตรวจสอบผลลัพธ์

### ตรวจสอบจำนวนเอกสารใน Collection

```python
count = collection.count_documents({})
print(f"Total documents: {count}")
```

### ตรวจสอบเอกสารที่มี content_hash

```python
# Query เอกสารที่มี content_hash
docs_with_hash = collection.find(
    {"metadata.content_hash": {"$exists": True}},
    limit=10
)

for doc in docs_with_hash:
    print(f"Hash: {doc['metadata']['content_hash'][:16]}...")
    print(f"Content: {doc['content'][:100]}...")
```

---

## 🚀 ข้อดีของ Incremental Indexing

1. **ประหยัดเวลา** - ไม่ต้องดึงข้อมูลซ้ำทั้งหมด
2. **ประหยัด API calls** - ลดการเรียก embedding API
3. **ประหยัด storage** - ไม่มีข้อมูลซ้ำซ้อน
4. **รักษาข้อมูลเดิม** - ไม่ต้องลบและ insert ใหม่ทั้งหมด
5. **ตรวจสอบได้** - มี stats แสดงผลการดำเนินการ

---

## ⚠️ ข้อควรระวัง

1. **Hash Keys ต้องเลือกให้เหมาะสม**
   - ถ้าเลือก keys ที่ไม่ unique อาจทำให้เอกสารต่างกันมี hash เดียวกัน
   - แนะนำใช้ ID หรือ slug ร่วมกับ keys อื่นๆ

2. **Metadata Filter ต้องตรงกับข้อมูล**
   - ถ้า filter ไม่ตรง อาจดึงเอกสารผิดประเภทมาเปรียบเทียบ

3. **Delete Missing ใช้ด้วยความระมัดระวัง**
   - ถ้าตั้งเป็น `True` และข้อมูลใหม่ไม่ครบ อาจลบข้อมูลเก่าที่ยังต้องการ

4. **Content Hash อยู่ใน Metadata**
   - ทุก document จะมี `content_hash` ใน metadata
   - ไม่ควรลบหรือแก้ไข field นี้

---

## 🛠️ Troubleshooting

### ปัญหา: ข้อมูลซ้ำยังถูก insert

**สาเหตุ:**
- Hash keys ไม่ครอบคลุมข้อมูลที่เปลี่ยนแปลง
- Metadata filter ไม่ถูกต้อง

**แก้ไข:**
- เพิ่ม hash keys ให้ครอบคลุมมากขึ้น
- ตรวจสอบ metadata filter

### ปัญหา: ข้อมูลใหม่ไม่ถูก insert

**สาเหตุ:**
- Content เหมือนกับข้อมูลเดิมทุกอย่าง
- Hash keys ไม่รวม field ที่เปลี่ยนแปลง

**แก้ไข:**
- ตรวจสอบว่าเนื้อหาเปลี่ยนแปลงจริงหรือไม่
- ปรับ hash keys ให้เหมาะสม

### ปัญหา: Error "content_hash not found"

**สาเหตุ:**
- เอกสารเก่าไม่มี content_hash (ถูก insert ก่อนใช้ incremental indexing)

**แก้ไข:**
- รัน data ingestion ใหม่ทั้งหมดครั้งเดียว
- หรือเพิ่ม content_hash ให้กับเอกสารเก่าด้วย script

---

## 📝 สรุป

ระบบ Incremental Indexing ช่วยให้การจัดการข้อมูลมีประสิทธิภาพมากขึ้น โดยเฉพาะเมื่อต้องอัพเดทข้อมูลบ่อยๆ 

**การใช้งานพื้นฐาน:**
1. Import `enable_incremental_mode`
2. เตรียม documents
3. กำหนด `metadata_filter` และ `hash_keys`
4. เรียกใช้ function
5. ตรวจสอบ stats

**ไฟล์ที่เกี่ยวข้อง:**
- `incremental_utils.py` - Core utilities
- `*_data.py` - Data ingestion scripts (ทั้งหมดใช้ incremental indexing แล้ว)

---

## 📚 เอกสารเพิ่มเติม

- [AstraDB Documentation](https://docs.datastax.com/en/astra/home/astra.html)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [SHA-256 Hashing](https://en.wikipedia.org/wiki/SHA-2)

---

**อัพเดทล่าสุด:** 31 ธันวาคม 2025

