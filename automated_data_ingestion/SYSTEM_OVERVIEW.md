# System Overview - Automated Data Ingestion

## 🎯 เป้าหมายของระบบ

สร้างระบบที่ **automate** การดึงข้อมูลจากเว็บไซต์หรือ API โดย:
- ✅ **ไม่ต้องแก้ไขโค้ด** เมื่อต้องการดึงข้อมูลหน้าเว็บใหม่
- ✅ **ใช้ Dashboard** สำหรับกรอกข้อมูลและตั้งค่า
- ✅ **Prompt-based extraction** ระบุสิ่งที่ต้องการผ่าน prompt
- ✅ **รองรับทั้งเว็บไซต์และ API**

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                   │
│  - Create Jobs                                          │
│  - View History                                         │
│  - Manage Collections                                   │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              DataIngestionOrchestrator                   │
│  - Coordinate all components                            │
│  - Error handling                                       │
│  - Job management                                       │
└──────┬──────────────┬──────────────┬────────────────────┘
       │              │              │
       ▼              ▼              ▼
┌─────────────┐ ┌──────────────┐ ┌──────────────────┐
│  WebScraper │ │ LLMExtractor │ │ AstraDBManager   │
│             │ │              │ │                  │
│ - Selenium  │ │ - OpenAI LLM │ │ - Connection     │
│ - Requests  │ │ - Rule-based │ │ - Collections    │
│ - API calls │ │   fallback   │ │ - Incremental    │
└─────────────┘ └──────────────┘ └──────────────────┘
```

## 📦 Components

### 1. Dashboard (`dashboard/app.py`)
- **UI** สำหรับสร้างและจัดการ jobs
- **Form** สำหรับกรอก URL, collection name, และ prompt
- **History** ดูประวัติการทำงาน
- **Collections** จัดการ collections ใน AstraDB

### 2. WebScraper (`core/scraper.py`)
- **Selenium** สำหรับ JavaScript rendering
- **Requests** สำหรับ static pages
- **API** สำหรับ REST endpoints
- Auto-detect ว่าจะใช้วิธีไหน

### 3. LLMExtractor (`core/llm_extractor.py`)
- **OpenAI LLM** สำหรับ extract ตาม prompt
- **Rule-based fallback** เมื่อไม่มี OpenAI API
- รองรับทั้ง HTML และ JSON

### 4. AstraDBManager (`core/astradb_manager.py`)
- จัดการการเชื่อมต่อ AstraDB
- สร้าง/เข้าถึง collections
- Incremental indexing (ไม่เพิ่มข้อมูลซ้ำ)

### 5. Orchestrator (`core/orchestrator.py`)
- Coordinate กระบวนการทั้งหมด
- Scrape → Extract → Store
- Error handling และ cleanup

## 🔄 Workflow

```
1. User creates job via Dashboard
   ↓
2. Orchestrator receives job config
   ↓
3. WebScraper fetches data (HTML/JSON)
   ↓
4. LLMExtractor extracts data based on prompt
   ↓
5. Documents are created with metadata
   ↓
6. AstraDBManager stores in collection
   ↓
7. Results returned to Dashboard
```

## 📊 Data Flow

```
URL/API → Scraper → Content (HTML/JSON)
                              ↓
                    LLMExtractor + Prompt
                              ↓
                      Documents (LangChain)
                              ↓
                    Text Splitting (chunks)
                              ↓
                    Embeddings (HuggingFace)
                              ↓
                    AstraDB (Vector Store)
```

## 🔑 Key Features

### 1. Prompt-Based Extraction
ผู้ใช้ระบุสิ่งที่ต้องการผ่าน prompt แทนการเขียนโค้ด:
```
"ดึงข้อมูลลิงก์ทั้งหมดที่มีอยู่ในหน้าเว็บ"
```

### 2. Auto-Detection
- ตรวจสอบว่า URL เป็น API หรือ webpage
- เลือกวิธี scraping ที่เหมาะสม (Selenium vs Requests)
- Fallback mechanisms เมื่อเกิด error

### 3. Incremental Indexing
- ไม่เพิ่มข้อมูลซ้ำ (ใช้ content hash)
- อัพเดทข้อมูลที่มีการเปลี่ยนแปลง
- ลบข้อมูลเก่าที่ไม่มีแล้ว (optional)

### 4. Flexible Configuration
- Metadata filters
- Hash keys สำหรับ duplicate detection
- Chunking settings
- Custom headers

## 🔄 Differences from Original System

| Feature | Original System | New System |
|---------|----------------|------------|
| **Configuration** | Hard-coded in Python | Dashboard UI |
| **Adding new source** | Create new Python file | Fill form in Dashboard |
| **Extraction logic** | Coded in Python | Prompt-based |
| **Flexibility** | Low (needs code changes) | High (no code changes) |
| **User-friendly** | Requires Python knowledge | No coding required |

## 🚀 Usage Scenarios

### Scenario 1: New Website
**เดิม**: ต้องสร้างไฟล์ Python ใหม่, เขียน scraping logic  
**ใหม่**: กรอก URL และ prompt ใน Dashboard

### Scenario 2: Change Extraction
**เดิม**: แก้ไขโค้ด Python  
**ใหม่**: แก้ไข prompt ใน Dashboard

### Scenario 3: Multiple Sources
**เดิม**: ต้อง maintain หลายไฟล์ Python  
**ใหม่**: สร้างหลาย jobs ใน Dashboard

## 📝 Future Enhancements

- [ ] Scheduled jobs (cron-like)
- [ ] Email notifications
- [ ] More LLM providers (Claude, Gemini)
- [ ] Visual extraction preview
- [ ] Batch job processing
- [ ] Export/Import job configurations

## 🔗 Related Files

- `README.md` - เอกสารการใช้งาน
- `QUICKSTART.md` - คู่มือเริ่มต้นใช้งาน
- `example_usage.py` - ตัวอย่างการใช้งานผ่าน Python

