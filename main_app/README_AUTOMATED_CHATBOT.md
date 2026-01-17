# Unified Chatbot with Automated Data Ingestion Integration

## 📋 ภาพรวม

`main_unified_chatbot_automated.py` เป็น Unified Chatbot ที่ทำงานร่วมกับ **Automated Data Ingestion System** แบบ dynamic โดย:

- ✅ **ดึง Collections แบบ Dynamic** - ไม่ต้อง hard-code collection names
- ✅ **Hybrid Intent Classification** - ใช้ Rule-Based + LLM (เหมือน main_unified_chatbot_hybrid.py)
- ✅ **Auto-Discovery** - ค้นหา collections ที่มีอยู่ใน AstraDB อัตโนมัติ
- ✅ **Dynamic Retriever Creation** - สร้าง retriever และ QA chain แบบ dynamic
- ✅ **รองรับการเพิ่ม Collections ใหม่** - เพิ่ม collection ใหม่ผ่าน Dashboard ไม่ต้องแก้โค้ด

## 🎯 ความแตกต่างจาก main_unified_chatbot_hybrid.py

| Feature | main_unified_chatbot_hybrid.py | main_unified_chatbot_automated.py |
|---------|-------------------------------|-----------------------------------|
| **Collection Source** | Hard-coded ในโค้ด | Dynamic จาก AstraDB |
| **Retriever Creation** | Static (ต้อง import แต่ละ agent) | Dynamic (สร้างจาก collection name) |
| **Adding New Collection** | ต้องแก้โค้ด | เพิ่มผ่าน Dashboard แล้วรันใหม่ |
| **Integration** | แยกจาก automated_data_ingestion | รวมกับ automated_data_ingestion |

## 🚀 การใช้งาน

### 1. ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
pip install -r automated_data_ingestion/requirements.txt
```

### 2. ตั้งค่า Environment Variables

สร้างไฟล์ `.env`:

```env
ASTRA_DB_APPLICATION_TOKEN=your_token_here
ASTRA_DB_API_ENDPOINT=your_endpoint_here
ASTRA_DB_KEYSPACE=default_keyspace
OPENAI_API_KEY=your_openai_key_here  # Optional - สำหรับ LLM fallback
OPENAI_BASE_URL=https://openrouter.ai/api/v1  # Optional
```

### 3. สร้าง Collections ผ่าน Automated Data Ingestion

ใช้ Dashboard เพื่อสร้าง collections:

```bash
streamlit run automated_data_ingestion/dashboard/app.py
```

หรือใช้ Python code:

```python
from automated_data_ingestion.core.orchestrator import DataIngestionOrchestrator
from automated_data_ingestion.models.job_config import ScrapingJobConfig

orchestrator = DataIngestionOrchestrator()
orchestrator.initialize_astradb()

config = ScrapingJobConfig(
    job_id="job-001",
    name="Scrape AllPeople",
    url="https://computing.kku.ac.th/allpeople",
    collection_name="allpeople_embedding",
    extraction_prompt="ดึงข้อมูลอาจารย์ทั้งหมด..."
)

result = orchestrator.execute_job(config)
```

### 4. รัน Unified Chatbot

```bash
python main_app/main_unified_chatbot_automated.py
```

## 🔧 การทำงาน

### 1. Auto-Discovery Collections

เมื่อเริ่มต้น chatbot จะ:
- เชื่อมต่อกับ AstraDB
- ค้นหา collections ที่มีอยู่ทั้งหมด
- สร้าง mapping ระหว่าง intent กับ collection names

```python
# Auto-detect collections
chatbot = UnifiedChatbotAutomated()

# หรือระบุ mapping เอง
collection_mapping = {
    "allpeople": "allpeople_embedding",
    "contact": "services_embedding",
    "links": "services_embedding",
    "scholarship": "scholarship_embedding",
    ...
}
chatbot = UnifiedChatbotAutomated(collection_mapping=collection_mapping)
```

### 2. Dynamic Retriever Creation

สำหรับแต่ละ collection:
- สร้าง `DynamicAstraDBRetriever` จาก collection
- สร้าง QA chain จาก retriever
- เก็บไว้ใน `chatbot_map`

### 3. Hybrid Intent Classification

เหมือนกับ `main_unified_chatbot_hybrid.py`:
- **Rule-Based**: ใช้ keywords และ patterns
- **LLM Fallback**: ใช้เมื่อ rule-based ไม่มั่นใจ

### 4. Query Routing

1. วิเคราะห์ query ด้วย Hybrid Classification
2. Map intent ไปยัง collection
3. ใช้ QA chain ของ collection นั้น
4. ถ้าไม่แน่ใจ → Multi-agent search

## 📝 Collection Mapping

### Default Mapping Patterns

ระบบจะพยายาม map collections ตาม patterns ต่อไปนี้:

```python
default_patterns = {
    "allpeople": ["allpeople", "faculty", "staff"],
    "contact": ["contact", "services"],
    "links": ["links", "services"],
    "scholarship": ["scholarship"],
    "student_club": ["student_club", "club"],
    "students": ["students", "student"],
    "research": ["research", "researchgroup"],
    "bsc_entrance": ["bsc", "entrance", "admission"],
    "digital_services": ["digital", "services"],
    "graduate": ["graduate"]
}
```

### Custom Mapping

ถ้าต้องการระบุ mapping เอง:

```python
collection_mapping = {
    "allpeople": "allpeople_embedding",
    "contact": "contact_embedding",
    "links": "links_embedding",
    "scholarship": "scholarship_embedding",
    "student_club": "student_club_embedding",
    "students": "students_embedding",
    "research": "researchgroup_embedding",
    "bsc_entrance": "bsc_entrance_embedding",
    "digital_services": "digital_services_embedding",
    "graduate": "graduate_embedding"
}

chatbot = UnifiedChatbotAutomated(collection_mapping=collection_mapping)
```

## 🔄 Workflow

```
1. User สร้าง collection ผ่าน Automated Data Ingestion Dashboard
   ↓
2. Collection ถูกเก็บใน AstraDB
   ↓
3. รัน Unified Chatbot → Auto-discover collections
   ↓
4. สร้าง retrievers และ QA chains แบบ dynamic
   ↓
5. User ถามคำถาม → Hybrid Classification
   ↓
6. Route ไปยัง collection ที่เหมาะสม
   ↓
7. ค้นหาและตอบคำถาม
```

## 💡 ตัวอย่างการใช้งาน

### ตัวอย่าง 1: Auto-Discovery

```python
from main_app.main_unified_chatbot_automated import UnifiedChatbotAutomated

# Auto-detect collections
chatbot = UnifiedChatbotAutomated()

# ถามคำถาม
answer = chatbot.answer("อาจารย์สมชาย")
print(answer)
```

### ตัวอย่าง 2: Custom Mapping

```python
from main_app.main_unified_chatbot_automated import UnifiedChatbotAutomated

# ระบุ mapping เอง
collection_mapping = {
    "allpeople": "allpeople_embedding",
    "contact": "services_embedding",
    "links": "services_embedding"
}

chatbot = UnifiedChatbotAutomated(collection_mapping=collection_mapping)
answer = chatbot.answer("ติดต่อวิทยาลัย")
```

### ตัวอย่าง 3: เพิ่ม Collection ใหม่

1. สร้าง collection ใหม่ผ่าน Dashboard:
   - Collection Name: `news_embedding`
   - URL: `https://computing.kku.ac.th/news`
   - Extraction Prompt: `ดึงข้อมูลข่าวทั้งหมด...`

2. เพิ่ม intent pattern ใน `HybridIntentClassifier`:
   ```python
   "news": {
       "keywords": ["ข่าว", "news", "ประกาศ", "announcement"],
       "patterns": [r'.*ข่าว.*', r'.*news.*']
   }
   ```

3. เพิ่ม collection mapping:
   ```python
   collection_mapping["news"] = "news_embedding"
   ```

4. รัน chatbot ใหม่ → ระบบจะใช้ collection ใหม่อัตโนมัติ!

## 🎨 Features

### 1. Dynamic Retriever Factory

```python
retriever = create_retriever_from_collection(
    collection_name="allpeople_embedding",
    astradb_manager=astradb_manager
)
```

### 2. Dynamic QA Chain Creation

```python
qa_function = create_qa_chain(
    retriever=retriever,
    collection_name="allpeople_embedding"
)
```

### 3. Hybrid Search

`DynamicAstraDBRetriever` รองรับ:
- **Vector Search**: ใช้ embeddings
- **Text Search**: ใช้ regex matching
- **Hybrid**: รวมผลลัพธ์จากทั้งสองวิธี

## ⚠️ ข้อควรระวัง

1. **Collection Names**: ควรใช้ชื่อที่สอดคล้องกับ intent patterns
2. **Vector Dimension**: ต้องเป็น 384 (สำหรับ all-MiniLM-L6-v2)
3. **Collection Existence**: ต้องสร้าง collection ก่อนรัน chatbot
4. **Metadata**: ควรมี metadata ที่เหมาะสมสำหรับ filtering

## 🔧 Troubleshooting

### Collection ไม่พบ

```
⚠️ Failed to create retriever for allpeople_embedding: Collection not found
```

**Solution**: สร้าง collection ผ่าน Dashboard หรือ Python code ก่อน

### Intent ไม่ match

```
❓ ไม่แน่ใจประเภทคำถาม - จะค้นหาจากทุก Agent
```

**Solution**: 
- เพิ่ม keywords/patterns ใน `intent_patterns`
- หรือใช้ LLM fallback (ต้องมี OPENAI_API_KEY)

### Retriever ไม่ทำงาน

```
❌ Error from อาจารย์และบุคลากร: ...
```

**Solution**: 
- ตรวจสอบว่า collection มีข้อมูล
- ตรวจสอบว่า vector search เปิดใช้งาน
- ตรวจสอบ embedding model

## 📚 Related Files

- `main_unified_chatbot_hybrid.py` - Original hybrid chatbot (hard-coded)
- `automated_data_ingestion/core/astradb_manager.py` - AstraDB manager
- `automated_data_ingestion/core/orchestrator.py` - Data ingestion orchestrator
- `automated_data_ingestion/dashboard/app.py` - Dashboard for creating collections

## 🚀 Future Enhancements

- [ ] Auto-reload collections (ไม่ต้อง restart)
- [ ] Collection health check
- [ ] Performance monitoring
- [ ] Caching mechanisms
- [ ] Multi-keyspace support
- [ ] Collection versioning

## 📄 License

Same as parent project

