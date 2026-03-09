# 🔬 ระบบ Automated Data Ingestion - วิเคราะห์ลึก (In-Depth Analysis)

## 📊 ภาพรวมทั้งระบบ

ระบบนี้ทำงานแบบ Pipeline ที่มี 3 ขั้นตอนหลัก:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INPUT (Dashboard)                       │
│  URL/API + Prompt + Collection Name → ScrapingJobConfig            │
└────────────────────────────┬────────────────────────────────────────┘
                             │
          ┌──────────────────▼──────────────────┐
          │   DataIngestionOrchestrator          │
          │   (ประสานงานทั้งระบบ)                │
          └──┬──────────────┬────────────┬───────┘
             │              │            │
    ┌────────▼──────┐ ┌─────▼────┐ ┌──▼─────────────┐
    │   WebScraper  │ │LLMExtract │ │AstraDBManager │
    │               │ │           │ │                │
    │1. Fetch HTML  │ │2. Parse & │ │3. Embed &     │
    │2. Fetch JSON  │ │   Extract │ │   Store       │
    │3. Auto-detect │ │   Data    │ │                │
    └───────────────┘ └───────────┘ └────────────────┘
            │                │              │
            └────────────────┴──────────────┘
                             │
                   ┌─────────▼─────────┐
                   │ AstraDB (Vector DB)│
                   │                    │
                   │ 💾 Collections     │
                   │ 🔍 Embeddings      │
                   │ 📊 Metadata        │
                   └────────────────────┘
```

---

## 🔄 ขั้นตอนที่ 1: WebScraper - ดึงข้อมูล (Data Fetching)

### ตัวอักษรที่ทำงาน

```python
# core/scraper.py

class WebScraper:
    def __init__(self, use_selenium: bool = True, wait_time: int = 3):
        # use_selenium: ใช้ Selenium เพื่อ render JavaScript
        # wait_time: รอให้ JavaScript โหลด (วินาที)
        self.use_selenium = use_selenium
        self.wait_time = wait_time
        self.driver = None
```

### กระบวนการ

#### **1.1 ดึง HTML จากเว็บ (Web Scraping)**

```
URL Input (เช่น: https://computing.kku.ac.th/students)
    │
    ├─ ตรวจสอบ: ต้องใช้ Selenium หรือ requests?
    │
    ├─ [ถ้า JavaScript ต้องการ render] → Selenium
    │   │
    │   ├─ Setup Chrome driver (headless mode)
    │   │   - ปิด GUI (--headless)
    │   │   - ปิด GPU acceleration (--disable-gpu)
    │   │   - Ignore SSL errors
    │   │
    │   ├─ Navigate ไปยัง URL
    │   │
    │   ├─ รอ JavaScript loading
    │   │   └─ time.sleep(wait_time) // default 3 วินาที
    │   │
    │   └─ Extract page_source ที่ render แล้ว
    │
    └─ [ถ้า Static HTML] → Requests
        │
        ├─ ตั้ง User-Agent headers
        ├─ GET request ไปยัง URL
        ├─ ตรวจสอบ status code (200)
        └─ Parse HTML response

ผลลัพธ์: BeautifulSoup Object (parsed HTML)
```

**ตัวอย่างโค้ด:**

```python
def scrape_with_selenium(self, url: str) -> BeautifulSoup:
    """Selenium จำลองการใช้ browser"""
    # 1. Setup driver
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # ไม่แสดง browser
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--ignore-certificate-errors')
    
    # 2. Navigate
    self.driver.get(url)  # ไปยัง URL
    
    # 3. รอ JavaScript
    time.sleep(self.wait_time)  # รอให้ JavaScript โหลด
    
    # 4. Extract
    soup = BeautifulSoup(self.driver.page_source, 'html.parser')
    return soup  # HTML ที่ render แล้ว

def scrape_with_requests(self, url: str) -> BeautifulSoup:
    """Requests สำหรับ static HTML"""
    # 1. ตั้ง headers
    headers = self._get_default_headers()
    
    # 2. GET request
    response = requests.get(url, headers=headers, timeout=30)
    
    # 3. ตรวจสอบ status
    if response.status_code != 200:
        raise Exception(f"HTTP {response.status_code}")
    
    # 4. Parse
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup
```

#### **1.2 ดึงข้อมูลจาก API (API Fetching)**

```
API Input (เช่น: https://api.computing.kku.ac.th/api/v1/students)
    │
    ├─ GET/POST request ไปยัง API
    │
    ├─ ตรวจสอบ status code
    │
    ├─ Parse JSON response
    │   │
    │   ├─ ตรวจสอบ structure
    │   │   └─ ถ้ามี 'data' key → extract จาก data
    │   │   └─ ถ้ามี 'items' → extract จาก items
    │   │   └─ ถ้า direct → ใช้ทั้งหมด
    │   │
    │   └─ Return as JSON dict/list
    │
    └─ Convert to JSON string

ผลลัพธ์: JSON string พร้อมใช้
```

**ตัวอย่างโค้ด:**

```python
def fetch_api(self, url: str, method: str = "GET", params=None) -> Any:
    """ดึงข้อมูล JSON จาก API"""
    headers = self._get_default_headers()
    
    # Request
    if method == "GET":
        response = requests.get(url, headers=headers, params=params, timeout=30)
    elif method == "POST":
        response = requests.post(url, headers=headers, json=params, timeout=30)
    
    # Check status
    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code}")
    
    # Parse JSON
    data = response.json()
    return data
```

#### **1.3 Auto-Detection Logic**

```python
def scrape(self, url: str) -> BeautifulSoup:
    """ตัดสินใจเอง: ใช้ Selenium หรือ Requests?"""
    
    if self.use_selenium:
        try:
            # พยายามใช้ Selenium ก่อน
            return self.scrape_with_selenium(url)
        except Exception as e:
            # ถ้า fail → ใช้ requests แทน
            print(f"Selenium failed: {e}")
            return self.scrape_with_requests(url)
    else:
        # ตั้งค่าให้ใช้ requests เลย
        return self.scrape_with_requests(url)
```

---

## 🤖 ขั้นตอนที่ 2: LLMExtractor - แยกข้อมูล (Data Extraction)

### ตัวอักษรที่ทำงาน

```python
# core/llm_extractor.py

class LLMExtractor:
    def __init__(self, use_openai: bool = True, model_name: str = "gpt-4o-mini"):
        # use_openai: ใช้ OpenAI LLM
        # model_name: ชื่อ model (gpt-4o-mini, gpt-4, เป็นต้น)
        self.use_openai = use_openai
        self.llm = ChatOpenAI(model_name=model_name)  # Initialize OpenAI
```

### กระบวนการ

```
HTML/JSON Content (from WebScraper)
    │
    ├─ Prepare Content (1️⃣ เตรียมข้อมูล)
    │   │
    │   ├─ ถ้า HTML → Clean & Convert to Text
    │   │   └─ ลบ <script>, <style>, comments
    │   │   └─ ลบ HTML tags ทั้งหมด
    │   │   └─ เก็บเฉพาะ text content
    │   │
    │   ├─ ถ้า JSON → Extract Content
    │   │   └─ ลบ metadata fields (domain, timestamp, etc.)
    │   │   └─ ดึง data.items หรือ data.data
    │   │   └─ Format ให้อ่านง่าย
    │   │
    │   └─ ถ้า Combined → รวมทั้งสอง
    │       └─ HTML + JSON side by side
    │
    ├─ Build Prompt (2️⃣ สร้าง prompt) 
    │   │
    │   ├─ System Prompt (ให้ LLM รู้ว่าต้องทำอะไร)
    │   │   └─ "Extract ข้อมูลตามที่ user ระบุ"
    │   │   └─ "ถ้าเป็น detail page → รวม document เดียว"
    │   │   └─ "ถ้าเป็น list → แยก document"
    │   │   └─ "Return JSON array เท่านั้น"
    │   │
    │   └─ User Prompt (บอกว่าต้องการอะไร)
    │       └─ Content preview
    │       └─ Extraction prompt from user
    │       └─ Instructions
    │
    ├─ Call LLM (3️⃣ เรียก LLM)
    │   │
    │   ├─ Send system + user prompt ไป OpenAI
    │   │
    │   └─ Get response (JSON string)
    │       └─ [
    │           {"content": "เนื้อหา", "metadata": {...}},
    │           {"content": "...", "metadata": {...}}
    │           ]
    │
    └─ Parse Response (4️⃣ parse ผลลัพธ์)
        │
        ├─ Parse JSON response
        │
        ├─ ตรวจสอบ structure
        │
        └─ Convert to LangChain Document objects
            └─ document.page_content = extracted content
            └─ document.metadata = extracted metadata
```

### ตัวอย่างโค้ด

**A. Clean HTML Content**

```python
def _clean_html_to_text(self, html_content: str) -> str:
    """ทำความสะอาด HTML → plain text"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # ลบ script, style, noscript, iframe
    for tag in soup(['script', 'style', 'noscript', 'iframe', 'meta', 'link', 'head']):
        tag.decompose()  # ลบ tag นี้ออกจาก tree
    
    # ลบ comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    
    # แปลงเป็น plain text
    text = soup.get_text(separator='\n', strip=True)
    
    # ทำความสะอาด whitespace
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    cleaned_text = '\n'.join(lines)
    
    return cleaned_text
```

**B. Extract JSON Content**

```python
def _extract_json_content(self, json_data, max_depth=6, current_depth=0) -> str:
    """ดึง content จาก JSON (ลบ structure)"""
    
    if current_depth >= max_depth:
        return ""  # หลีกเลี่ยง infinite recursion
    
    # ถ้า dict
    if isinstance(json_data, dict):
        content_parts = []
        
        for key, value in json_data.items():
            # ข้าม metadata fields
            if key.lower() in ['domain', 'timestamp', 'status', 'code']:
                continue
            
            # ถ้า 'data' key → recursive
            if key.lower() == 'data' and isinstance(value, (dict, list)):
                nested = self._extract_json_content(value, max_depth, current_depth)
                if nested:
                    content_parts.append(nested)
                continue
            
            # ถ้า value เป็น string/number → เก็บ
            if isinstance(value, (str, int, float, bool)):
                if value:  # ข้าม empty values
                    content_parts.append(f"{key}: {value}")
            
            # ถ้า value เป็น dict/list → recursive
            elif isinstance(value, (dict, list)):
                nested = self._extract_json_content(value, max_depth, current_depth + 1)
                if nested:
                    content_parts.append(f"{key}:\n{nested}")
        
        return '\n'.join(content_parts)
    
    # ถ้า list
    elif isinstance(json_data, list):
        content_parts = []
        
        for idx, item in enumerate(json_data):
            if isinstance(item, (str, int, float, bool)):
                if item:
                    content_parts.append(str(item))
            elif isinstance(item, (dict, list)):
                nested = self._extract_json_content(item, max_depth, current_depth + 1)
                if nested:
                    content_parts.append(f"--- Item {idx + 1} ---\n{nested}")
        
        return '\n'.join(content_parts)
    
    else:
        return str(json_data) if json_data else ""
```

**C. Build LLM Prompt**

```python
def prepare_llm_content(self, content: str, prompt: str, content_type: str = "html"):
    """เตรียมข้อมูล + prompt สำหรับ LLM"""
    
    # ... Clean & Extract (as shown above) ...
    
    # System Prompt (ให้ LLM รู้ว่าต้องทำอะไร)
    system_prompt = """คุณเป็นผู้ช่วยในการ extract ข้อมูลจาก HTML/JSON

ภารกิจ:
1. อ่านโครงสร้างข้อมูล
2. วิเคราะห์ prompt
3. ค้นหาข้อมูลที่ตรงกัน
4. วิเคราะห์: detail page หรือ list?
   - Detail: รวม document เดียว
   - List: แยก document
5. Return JSON array เท่านั้น

Format: [{"content": "เนื้อหา", "metadata": {...}}, ...]
"""
    
    # User Prompt (บอก user อะไรที่ต้องการ)
    user_prompt = f"""Content:
{content_preview}

Prompt:
{prompt}

Instructions:
- Extract ข้อมูลตาม prompt
- ถ้า detail page → รวม document
- ถ้า list → แยก document
- Return JSON array
"""
    
    return content_preview, system_prompt, user_prompt
```

**D. Call LLM & Parse Response**

```python
def extract_with_llm(self, content: str, prompt: str, content_type: str = "html"):
    """เรียก LLM extract ข้อมูล"""
    
    # เตรียมข้อมูล
    content_preview, system_prompt, user_prompt = self.prepare_llm_content(
        content, prompt, content_type
    )
    
    # เรียก LLM (OpenAI)
    from langchain.schema import HumanMessage, SystemMessage
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = self.llm.invoke(messages)  # เรียก LLM
    response_text = response.content  # รับ response string
    
    # Parse JSON response
    import json
    try:
        # ลอง parse JSON
        data = json.loads(response_text)
    except json.JSONDecodeError:
        # ถ้า fail → ลองดึง JSON จาก markdown code block
        import re
        json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
        else:
            raise Exception(f"Invalid JSON response: {response_text}")
    
    # Convert to LangChain Documents
    documents = []
    
    if isinstance(data, list):
        for item in data:
            doc = Document(
                page_content=item.get("content", ""),
                metadata=item.get("metadata", {})
            )
            documents.append(doc)
    
    return documents
```

---

## 💾 ขั้นตอนที่ 3: AstraDBManager - เก็บข้อมูล (Data Storage)

### ตัวอักษรที่ทำงาน

```python
# core/astradb_manager.py

class AstraDBManager:
    def __init__(self, token: str, endpoint: str, keyspace: str):
        # token: AstraDB application token
        # endpoint: AstraDB API endpoint
        # keyspace: database keyspace
        self.client = DataAPIClient(token=token)
        self.database = self.client.get_database_by_api_endpoint(endpoint)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )  # ใช้ HuggingFace embedding
```

### กระบวนการ

```
Documents from LLMExtractor
    │
    ├─ Split Documents (1️⃣ แบ่งข้อมูล)
    │   │
    │   ├─ Check ขนาด content
    │   │   └─ ถ้า > chunk_size → split
    │   │   └─ ถ้า ≤ chunk_size → ใช้ทั้งหมด
    │   │
    │   ├─ CharacterTextSplitter
    │   │   └─ chunk_size: 500 (default)
    │   │   └─ chunk_overlap: 50 (เพื่อ context continuity)
    │   │   └─ separator: '\n'
    │   │
    │   └─ ผลลัพธ์: List of chunks
    │
    ├─ Generate Embeddings (2️⃣ สร้าง embeddings)
    │   │
    │   ├─ สำหรับแต่ละ chunk
    │   │   │
    │   │   ├─ embedding_model.embed_query(chunk_text)
    │   │   │   └─ Convert text → vector (384 dimensions)
    │   │   │   └─ โดยใช้ all-MiniLM-L6-v2 model
    │   │   │
    │   │   └─ Vector = [0.123, -0.456, 0.789, ..., 0.012]
    │   │
    │   └─ ผลลัพธ์: Vector embeddings
    │
    ├─ Prepare Documents (3️⃣ เตรียมเอกสาร)
    │   │
    │   ├─ สำหรับแต่ละ chunk
    │   │   │
    │   │   ├─ _id: unique ID (UUID)
    │   │   ├─ content: text content
    │   │   ├─ $vector: embedding vector
    │   │   └─ metadata: {'job_id': '...', 'source': '...', ...}
    │   │
    │   └─ Document dict สำหรับ AstraDB
    │
    ├─ Incremental Indexing (4️⃣ เพิ่มข้อมูล Smart)
    │   │
    │   ├─ ตรวจสอบ hash
    │   │   └─ hash_keys: ['url', 'title'] (ระบุ key สำหรับ hash)
    │   │   └─ สร้าง hash จาก hash_keys
    │   │   └─ เช็คว่ามี document แบบนี้ใน DB แล้วหรือไม่
    │   │
    │   ├─ ถ้า hash มีอยู่แล้ว (duplicate)
    │   │   └─ Skip (ข้าม)
    │   │
    │   ├─ ถ้า hash ใหม่
    │   │   └─ Insert (เพิ่มเข้า DB)
    │   │
    │   └─ ผลลัพธ์: Stats {inserted, skipped, updated, deleted}
    │
    └─ Return Results
        └─ Inserted count
        └─ Skipped count
        └─ Updated count
        └─ Deleted count
```

### ตัวอย่างโค้ด

**A. Split Documents**

```python
def insert_documents(self, collection_name: str, documents: List[Document],
                    chunk_size: int = 500, chunk_overlap: int = 50):
    """เก็บ documents ลง AstraDB"""
    
    collection = self.get_or_create_collection(collection_name)
    
    # 1. Split ถ้า content ใหญ่เกิน
    if len(documents) > 0 and len(documents[0].page_content) > chunk_size:
        print(f"Splitting documents (chunk_size={chunk_size}, overlap={chunk_overlap})")
        
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n"
        )
        documents = splitter.split_documents(documents)
```

**B. Generate Embeddings**

```python
def _basic_insert(self, collection, documents: List[Document]):
    """เก็บ documents (basic mode)"""
    
    inserted_count = 0
    batch_size = 20
    documents_to_insert = []
    
    for i, doc in enumerate(documents):
        # 1. Generate embedding สำหรับ content นี้
        vector = self.embedding_model.embed_query(doc.page_content)
        # vector มีขนาด 384 dimensions (สำหรับ all-MiniLM-L6-v2)
        
        # 2. Prepare document
        doc_dict = {
            "_id": str(uuid.uuid4()),           # unique ID
            "content": doc.page_content,         # text content
            "$vector": vector,                   # embedding vector
            "metadata": doc.metadata             # metadata
        }
        documents_to_insert.append(doc_dict)
        
        # 3. Insert in batches
        if len(documents_to_insert) >= batch_size or i == len(documents) - 1:
            result = collection.insert_many(documents_to_insert)
            inserted_count += len(result.inserted_ids)
            documents_to_insert = []
    
    return {
        "inserted": inserted_count,
        "skipped": 0,
        "updated": 0,
        "deleted": 0
    }
```

**C. Vector Search (ถูกใช้ใน Chatbot)**

```python
# เมื่อ user ถามคำถาม → Chatbot จะ search เช่นนี้

question = "ติดต่ออาจารย์สมชาย"

# 1. Embed คำถาม
question_vector = embedding_model.embed_query(question)

# 2. Search ใน AstraDB
results = collection.find(
    {},
    sort={"$vector": question_vector},  # ใช้ vector similarity
    limit=5  # ได้ 5 ผลลัพธ์ที่คล้ายที่สุด
)

# 3. ได้ documents ที่เกี่ยวข้อง
for result in results:
    print(result['content'])
    # → "อาจารย์สมชาย..."
```

---

## 🎯 ขั้นตอนที่ 4: Orchestrator - ประสานงาน (Coordination)

### ตัวอักษรที่ทำงาน

```python
# core/orchestrator.py

class DataIngestionOrchestrator:
    """จัดการกระบวนการ Scrape → Extract → Store"""
    
    def execute_job(self, config: ScrapingJobConfig) -> ScrapingJobResult:
        """ประสานงานทั้งระบบ"""
```

### ขั้นตอนโดยละเอียด

```
config = {
    job_id: "uuid",
    name: "Scrape Students",
    url: "https://computing.kku.ac.th/students",
    api_url: None,
    collection_name: "students_embedding",
    extraction_prompt: "ดึงข้อมูลนักศึกษาทั้งหมด",
    use_selenium: True,
    wait_time: 3,
    chunk_size: 500,
    chunk_overlap: 50,
    hash_keys: ["id"],
    metadata_filter: {"category": "students"}
}

┌─ execute_job(config)
│
├─ 1️⃣  Initialize (เตรียมส่วนประกอบ)
│  │
│  ├─ Initialize AstraDBManager
│  ├─ Initialize WebScraper
│  │   └─ use_selenium=True, wait_time=3
│  └─ Initialize LLMExtractor
│      └─ use_openai=True, model="gpt-4o-mini"
│
├─ 2️⃣  Fetch Data (ดึงข้อมูล)
│  │
│  ├─ Fetch HTML from URL
│  │  └─ scraper.scrape(config.url)
│  │  └─ ได้: BeautifulSoup object
│  │
│  ├─ Fetch JSON from API (ถ้ามี)
│  │  └─ scraper.fetch_api(config.api_url)
│  │  └─ ได้: JSON dict
│  │
│  └─ Combine HTML + JSON (ถ้ามีทั้งคู่)
│     └─ content = HTML + JSON
│     └─ content_type = "combined"
│
├─ 3️⃣  Prepare for LLM (เตรียม prompt)
│  │
│  ├─ Clean HTML → plain text
│  ├─ Extract JSON content → readable format
│  ├─ Prepare preview สำหรับ dashboard
│  └─ Build system + user prompt
│
├─ 4️⃣  Extract Data (แยกข้อมูล)
│  │
│  ├─ Call LLM.extract()
│  │  └─ Send content + prompt ไป OpenAI
│  │  └─ Get back JSON array
│  │
│  ├─ Parse response → List of Documents
│  │
│  ├─ Detect detail page vs list
│  │  └─ ถ้า detail → merge documents เป็น 1
│  │  └─ ถ้า list → keep แยกกัน
│  │
│  └─ Add metadata
│     └─ job_id, job_name, source_url
│     └─ merge with metadata_filter
│
├─ 5️⃣  Auto-detect Hash Keys (ถ้าไม่มี)
│  │
│  ├─ ดูว่า metadata มี field อะไร
│  │  └─ slug? name? email? url?
│  │
│  └─ Set hash_keys อัตโนมัติ
│     └─ สำหรับ incremental indexing
│
├─ 6️⃣  Store in AstraDB (เก็บลง DB)
│  │
│  ├─ astradb_manager.insert_documents()
│  │  │
│  │  ├─ Get or create collection
│  │  ├─ Split documents (ถ้าต้อง)
│  │  ├─ Generate embeddings
│  │  ├─ Incremental indexing (ตรวจ duplicate)
│  │  │  └─ hash existing documents ด้วย hash_keys
│  │  │  └─ compare กับ documents ใหม่
│  │  │  └─ skip duplicates
│  │  │  └─ insert new ones
│  │  │
│  │  └─ Return stats
│  │
│  └─ Get stats (inserted, skipped, updated, deleted)
│
├─ 7️⃣  Return Result
│  │
│  └─ ScrapingJobResult {
│      job_id,
│      status: "success/error",
│      documents_processed: 45,
│      documents_inserted: 45,
│      documents_skipped: 0,
│      documents_updated: 0,
│      execution_time: 12.5s,
│      error_message: None,
│      llm_content_preview: "...",
│      llm_prompt_preview: "..."
│     }
│
└─ Save to history.json (บันทึก)
   └─ {
       "job_config": config,
       "result": result,
       "timestamp": "2026-01-20T10:30:00"
      }
```

### ตัวอย่างโค้ด

```python
def execute_job(self, config: ScrapingJobConfig) -> ScrapingJobResult:
    """Execute scraping job"""
    
    result = ScrapingJobResult(job_id=config.job_id, status="running")
    start_time = time.time()
    
    try:
        print(f"🚀 Starting job: {config.name}")
        
        # 1. Initialize
        if not self.astradb_manager:
            self.initialize_astradb()
        
        self.scraper = WebScraper(
            use_selenium=config.use_selenium,
            wait_time=config.wait_time
        )
        
        self.extractor = LLMExtractor(use_openai=True)
        
        # 2. Fetch data
        print("📥 Step 1: Fetching data...")
        
        html_content = None
        json_content = None
        
        # Fetch HTML
        if config.url and config.url != "batch_mode":
            print(f"   🌐 Fetching HTML: {config.url}")
            soup = self.scraper.scrape(config.url)
            html_content = str(soup)
            print(f"   ✅ Fetched HTML ({len(html_content)} chars)")
        
        # Fetch API
        if config.api_url:
            print(f"   🔌 Fetching API: {config.api_url}")
            api_data = self.scraper.fetch_api(config.api_url)
            import json
            json_content = json.dumps(api_data, ensure_ascii=False, indent=2)
            print(f"   ✅ Fetched JSON ({len(json_content)} chars)")
        
        # Combine
        if html_content and json_content:
            content = f"=== HTML ===\n{html_content}\n\n=== JSON ===\n{json_content}"
            content_type = "combined"
        elif json_content:
            content = json_content
            content_type = "json"
        else:
            content = html_content
            content_type = "html"
        
        # 3. Prepare for LLM
        print("\n📤 Preparing LLM content...")
        llm_preview, system_prompt, user_prompt = self.extractor.prepare_llm_content(
            content, config.extraction_prompt, content_type
        )
        result.llm_content_preview = llm_preview
        result.llm_prompt_preview = user_prompt
        
        # 4. Extract
        print("\n🔍 Step 2: Extracting data...")
        documents = self.extractor.extract(
            content=content,
            prompt=config.extraction_prompt,
            content_type=content_type
        )
        print(f"   📊 Extracted {len(documents)} documents")
        
        # 5. Detect detail page
        is_detail_page = (
            config.batch_mode or
            "รายละเอียด" in config.extraction_prompt.lower() or
            "/detail/" in config.url
        )
        
        if is_detail_page and len(documents) > 1:
            print(f"   🔗 Merging {len(documents)} documents...")
            merged_content = "\n\n".join([doc.page_content for doc in documents])
            documents = [Document(page_content=merged_content, metadata=documents[0].metadata)]
        
        # 6. Add metadata
        for doc in documents:
            doc.metadata.update({
                "job_id": config.job_id,
                "job_name": config.name,
                "source_url": config.url,
                "collection": config.collection_name
            })
            if config.metadata_filter:
                doc.metadata.update(config.metadata_filter)
        
        result.documents_processed = len(documents)
        
        # 7. Store in AstraDB
        print("\n💾 Step 3: Storing in AstraDB...")
        stats = self.astradb_manager.insert_documents(
            collection_name=config.collection_name,
            documents=documents,
            hash_keys=config.hash_keys,
            metadata_filter=config.metadata_filter,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        
        result.documents_inserted = stats.get("inserted", 0)
        result.documents_skipped = stats.get("skipped", 0)
        result.documents_updated = stats.get("updated", 0)
        result.status = "success"
        
        # 8. Return
        result.execution_time = time.time() - start_time
        return result
        
    except Exception as e:
        result.status = "error"
        result.error_message = str(e)
        result.execution_time = time.time() - start_time
        return result
```

---

## 🔗 Data Flow ของข้อมูล (Complete Picture)

```
User Input (Dashboard)
├─ URL: "https://api.computing.kku.ac.th/api/v1/students"
├─ Prompt: "ดึงข้อมูลนักศึกษา ชื่อ เมล เบอร์โทร"
├─ Collection: "students_embedding"
└─ config object → Orchestrator

Orchestrator.execute_job(config)
│
├─ 1. SCRAPE PHASE
│   │
│   ├─ WebScraper.fetch_api(url)
│   │  └─ GET request
│   │  └─ Parse JSON
│   │  └─ Return: {status: 200, data: {items: [...]}}
│   │
│   └─ Content: JSON string (5000 chars)
│
├─ 2. EXTRACT PHASE
│   │
│   ├─ LLMExtractor._extract_json_content(json_data)
│   │  └─ Recursive walk through JSON
│   │  └─ Skip: status, code, timestamp
│   │  └─ Keep: id, name, email, tel
│   │  └─ Return: "id: 001, name: สมชาย, email: somchai@kku.ac.th, tel: 0812345678"
│   │
│   ├─ LLMExtractor.prepare_llm_content()
│   │  └─ Clean content
│   │  └─ Build system + user prompt
│   │  └─ System: "Extract ตาม prompt, ถ้าเป็น list → แยก document"
│   │  └─ User: "Content: ..., Prompt: ดึงข้อมูลนักศึกษา"
│   │
│   ├─ LLMExtractor.extract_with_llm()
│   │  └─ Call OpenAI API
│   │  └─ Input: system + user prompt
│   │  └─ Output: JSON response
│   │     [
│   │       {
│   │         "content": "student_id: 001\nname: สมชาย\nemail: somchai@kku.ac.th\ntel: 0812345678",
│   │         "metadata": {
│   │           "id": "001",
│   │           "name": "สมชาย",
│   │           "email": "somchai@kku.ac.th",
│   │           "tel": "0812345678"
│   │         }
│   │       },
│   │       {...next student...}
│   │     ]
│   │
│   ├─ Parse & Convert to Documents
│   │  └─ Document 1:
│   │     page_content: "student_id: 001\nname: สมชาย\n..."
│   │     metadata: {id: "001", name: "สมชาย", ...}
│   │
│   └─ Documents: [doc1, doc2, doc3, ..., doc45]
│
├─ 3. STORAGE PHASE
│   │
│   ├─ AstraDBManager.insert_documents()
│   │  │
│   │  ├─ CharacterTextSplitter
│   │  │  └─ Split large documents (> 500 chars)
│   │  │  └─ Keep overlap 50 chars
│   │  │  └─ Result: [chunk1, chunk2, ..., chunk47]
│   │  │
│   │  ├─ HuggingFaceEmbeddings
│   │  │  └─ For each chunk:
│   │  │  │   embedding_model.embed_query(chunk_text)
│   │  │  │   └─ "student_id: 001, name: สมชาย, ..."
│   │  │  │   └─ → [0.123, -0.456, 0.789, ..., 0.012] (384 dims)
│   │  │  │
│   │  │  └─ Embeddings: [vec1, vec2, ..., vec47]
│   │  │
│   │  └─ Incremental Indexing
│   │     └─ For each new document:
│   │        ├─ Generate hash (from hash_keys)
│   │        │  └─ hash = hash(["001"])  // hash_keys = ["id"]
│   │        │  └─ hash = "a1b2c3d4..."
│   │        │
│   │        ├─ Search existing documents
│   │        │  └─ Query: {"hash": "a1b2c3d4"}
│   │        │  └─ Found: None
│   │        │
│   │        ├─ Insert to collection
│   │        │  └─ collection.insert({
│   │        │      _id: "uuid-123",
│   │        │      content: "student_id: 001...",
│   │        │      $vector: [0.123, -0.456, ...],
│   │        │      metadata: {id: "001", hash: "a1b2c3d4", ...}
│   │     })
│   │        │  └─ Status: ✅ Inserted
│   │        │
│   │        └─ Repeat for other 46 documents
│   │
│   └─ Stats: {inserted: 45, skipped: 0, updated: 0, deleted: 0}
│
└─ 4. RETURN RESULT
   │
   └─ ScrapingJobResult {
      job_id: "uuid-456",
      status: "success",
      documents_processed: 45,
      documents_inserted: 45,
      documents_skipped: 0,
      execution_time: 15.3s,
      ...
   }

💾 SAVED IN ASTRADB
│
├─ Collection: students_embedding
├─ Documents: 47 chunks (from 45 students)
├─ Embeddings: 47 vectors (384 dims each)
└─ Ready for Chatbot to search
```

---

## 🔍 Chatbot Search Flow

เมื่อ user ถาม: "ติดต่อสมชาย"

```
Question: "ติดต่อสมชาย"
    │
    ├─ 1. Intent Classification
    │   └─ Rule-based + LLM
    │   └─ Intent: "contact/people"
    │   └─ Collection: "students_embedding"
    │
    ├─ 2. Embed Question
    │   └─ embedding_model.embed_query("ติดต่อสมชาย")
    │   └─ qvec = [0.234, -0.567, ...] (384 dims)
    │
    ├─ 3. Vector Search
    │   └─ collection.find(
    │      {},
    │      sort={"$vector": qvec},  // Sort by vector similarity
    │      limit=3
    │   )
    │   │
    │   └─ AstraDB finds closest vectors:
    │      ├─ Doc1: similarity=0.95, "student_id: 001, name: สมชาย, tel: 0812345678"
    │      ├─ Doc2: similarity=0.91, "student_id: 002, name: สายธรรม, tel: 0898765432"
    │      └─ Doc3: similarity=0.87, "student_id: 045, name: สมปัญญา, tel: 0811111111"
    │
    ├─ 4. RAG (Retrieval + Generation)
    │   │
    │   ├─ Retrieval: ได้ 3 documents (from vector search)
    │   │
    │   ├─ Generate: ให้ LLM สร้าง answer
    │   │  └─ LLM.invoke({
    │   │      "question": "ติดต่อสมชาย",
    │   │      "context": "student_id: 001, name: สมชาย, tel: 0812345678..."
    │   │   })
    │   │
    │   └─ Answer: "สมชาย สามารถติดต่อได้ที่เบอร์โทร 0812345678"
    │
    └─ Display to user
```

---

## 📊 สรุป 3 ขั้นตอนหลัก

| ขั้นตอน | Component | Input | Process | Output |
|--------|-----------|-------|---------|--------|
| **1️⃣ Scrape** | WebScraper | URL/API | Fetch HTML/JSON using Selenium/Requests | HTML/JSON string |
| **2️⃣ Extract** | LLMExtractor | HTML/JSON + Prompt | Clean content + Call OpenAI LLM | Documents (with metadata) |
| **3️⃣ Store** | AstraDBManager | Documents | Split + Embed + Incremental insert | Vector DB (AstraDB) |

---

นี่คือการไหลเต็มระบบ! 🚀

