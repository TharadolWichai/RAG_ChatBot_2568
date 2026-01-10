# Automated Data Ingestion System

ระบบอัตโนมัติสำหรับการดึงข้อมูลจากเว็บไซต์หรือ API และเก็บใน AstraDB โดยไม่ต้องแก้ไขโค้ด

## 📋 ภาพรวม

ระบบนี้ช่วยให้คุณสามารถ:
- ✅ กรอก URL หรือ API endpoint ผ่าน Dashboard
- ✅ ตั้งชื่อ collection ใน AstraDB
- ✅ ใช้ Prompt เพื่อระบุว่าต้องการดึงข้อมูลส่วนไหน
- ✅ ไม่ต้องแก้ไขโค้ดเมื่อต้องการดึงข้อมูลหน้าเว็บใหม่

## 🚀 Quick Start

### 1. ติดตั้ง Dependencies

```bash
pip install streamlit langchain langchain-openai langchain-community sentence-transformers astrapy beautifulsoup4 requests selenium webdriver-manager
```

หรือใช้ไฟล์ requirements:
```bash
pip install -r requirements.txt
```

### 2. ตั้งค่า Environment Variables

สร้างไฟล์ `.env` ใน root directory (ถ้ายังไม่มี):

```env
ASTRA_DB_APPLICATION_TOKEN=your_token_here
ASTRA_DB_API_ENDPOINT=your_endpoint_here
ASTRA_DB_KEYSPACE=default_keyspace
OPENAI_API_KEY=your_openai_key_here  # Optional - สำหรับ LLM-based extraction
```

### 3. รัน Dashboard

```bash
streamlit run automated_data_ingestion/dashboard/app.py
```

Dashboard จะเปิดใน browser ที่ `http://localhost:8501`

## 📖 การใช้งาน

### สร้าง Job ใหม่

1. เปิด Dashboard และไปที่ tab "📝 Create New Job"
2. กรอกข้อมูล:
   - **ชื่อ Job**: ชื่อที่ระบุเพื่อจำง่าย
   - **URL หรือ API Endpoint**: ลิงก์ของเว็บไซต์หรือ API ที่ต้องการดึงข้อมูล
   - **Collection Name**: ชื่อ collection ใน AstraDB ที่จะเก็บข้อมูล
   - **Extraction Prompt**: คำอธิบายว่าต้องการดึงข้อมูลส่วนไหน
     - ตัวอย่าง: "ดึงข้อมูลลิงก์ทั้งหมดที่มีอยู่ในหน้า แยกชื่อลิงก์และ URL"
   - **คำอธิบาย**: (Optional) คำอธิบายเพิ่มเติมเกี่ยวกับ job

3. ตั้งค่าขั้นสูง (Optional):
   - **Metadata Category**: หมวดหมู่สำหรับ metadata
   - **Hash Keys**: Keys ที่ใช้สำหรับสร้าง unique hash (คั่นด้วย comma)
   - **Chunk Size/Overlap**: การตั้งค่าสำหรับ text splitting

4. กด "🚀 Create & Run Job" เพื่อเริ่มทำงาน

### ดูประวัติ Jobs

ไปที่ tab "📊 Job History" เพื่อดูประวัติการทำงานของ jobs ทั้งหมด

### จัดการ Collections

ไปที่ tab "📦 Collections" เพื่อดูและจัดการ collections ใน AstraDB

## 🏗️ โครงสร้างโปรเจค

```
automated_data_ingestion/
├── core/
│   ├── scraper.py          # Web scraping engine (รองรับ Selenium และ requests)
│   ├── llm_extractor.py    # LLM-based extraction
│   ├── astradb_manager.py  # AstraDB connection และ management
│   └── orchestrator.py     # Main orchestrator ที่รวมทุกอย่าง
├── dashboard/
│   └── app.py              # Streamlit dashboard
├── models/
│   └── job_config.py       # Data models สำหรับ job configuration
├── utils/
│   └── config.py           # Configuration management
└── README.md
```

## 🔧 Core Components

### 1. WebScraper (`core/scraper.py`)
- รองรับการ scrape เว็บไซต์ด้วย Selenium (สำหรับ JavaScript) และ requests
- รองรับการเรียก API endpoints
- Auto-detect ว่าจะใช้ Selenium หรือ requests

### 2. LLMExtractor (`core/llm_extractor.py`)
- ใช้ OpenAI LLM เพื่อ extract ข้อมูลตาม prompt ที่ระบุ
- Fallback ไปใช้ rule-based extraction หากไม่มี OpenAI API key
- รองรับทั้ง HTML และ JSON

### 3. AstraDBManager (`core/astradb_manager.py`)
- จัดการการเชื่อมต่อกับ AstraDB
- สร้างหรือเข้าถึง collections
- รองรับ incremental indexing

### 4. DataIngestionOrchestrator (`core/orchestrator.py`)
- Orchestrate กระบวนการทั้งหมด: scrape → extract → store
- จัดการ error handling และ cleanup

## 📝 ตัวอย่าง Extraction Prompts

### สำหรับหน้าเว็บที่มีลิงก์
```
ดึงข้อมูลลิงก์ทั้งหมดที่มีอยู่ในหน้าเว็บ
สำหรับแต่ละลิงก์ให้เก็บ:
- ชื่อลิงก์ (text)
- URL
- คำสำคัญที่เกี่ยวข้อง
```

### สำหรับหน้าเว็บที่มีตารางข้อมูล
```
ดึงข้อมูลจากตารางทั้งหมด
สำหรับแต่ละแถวให้เก็บ:
- ชื่อ (column 1)
- รายละเอียด (column 2)
- ลิงก์ (ถ้ามี)
```

### สำหรับ API Response
```
ดึงข้อมูลจาก JSON response
สำหรับแต่ละ item ให้เก็บ:
- title
- description
- url
- published_date
```

## ⚙️ Advanced Configuration

### Metadata Filter

ใช้ metadata filter เพื่อแยกข้อมูลใน collection เดียวกัน:

```python
metadata_filter = {"category": "students"}
```

### Hash Keys

ระบุ hash keys เพื่อป้องกันข้อมูลซ้ำ:

```python
hash_keys = ["url", "title"]  # สร้าง hash จาก url และ title
```

### Chunking

ตั้งค่า chunking สำหรับเอกสารขนาดใหญ่:

```python
chunk_size = 500      # ขนาด chunk
chunk_overlap = 50    # overlap ระหว่าง chunks
```

## 🐛 Troubleshooting

### Selenium ไม่ทำงาน
- ตรวจสอบว่า ChromeDriver อยู่ใน `drivers/` folder
- หรือติดตั้ง `webdriver-manager`: `pip install webdriver-manager`

### OpenAI API ไม่ทำงาน
- ตรวจสอบว่า `OPENAI_API_KEY` ถูกตั้งค่าใน `.env`
- ระบบจะ fallback ไปใช้ rule-based extraction อัตโนมัติ

### AstraDB Connection Error
- ตรวจสอบว่า `ASTRA_DB_APPLICATION_TOKEN` และ `ASTRA_DB_API_ENDPOINT` ถูกตั้งค่า
- ตรวจสอบ network connection

## 🔄 Differences from Original System

### ระบบเดิม
- ต้องสร้างไฟล์ Python ใหม่สำหรับแต่ละแหล่งข้อมูล
- ต้องแก้ไขโค้ดเมื่อต้องการเปลี่ยนการดึงข้อมูล
- Hard-coded URLs และ extraction logic

### ระบบใหม่
- ✅ ไม่ต้องแก้ไขโค้ด - ใช้ Dashboard
- ✅ Prompt-based extraction - ระบุสิ่งที่ต้องการผ่าน prompt
- ✅ รองรับทั้งเว็บไซต์และ API
- ✅ จัดการ configuration ผ่าน UI
- ✅ ดูประวัติและจัดการ jobs ได้ง่าย

## 📚 API Usage (สำหรับ Developers)

คุณสามารถใช้ระบบนี้ผ่าน Python code โดยตรง:

```python
from automated_data_ingestion.core.orchestrator import DataIngestionOrchestrator
from automated_data_ingestion.models.job_config import ScrapingJobConfig

# Initialize
orchestrator = DataIngestionOrchestrator()
orchestrator.initialize_astradb()

# Create job config
config = ScrapingJobConfig(
    job_id="job-001",
    name="Scrape Students Page",
    url="https://computing.kku.ac.th/students",
    collection_name="students_embedding",
    extraction_prompt="ดึงข้อมูลลิงก์ทั้งหมดที่มีอยู่ในหน้า"
)

# Execute
result = orchestrator.execute_job(config)
print(f"Status: {result.status}")
print(f"Documents inserted: {result.documents_inserted}")
```

## 📄 License

Same as parent project

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

