import os
import re
import time
import uuid

from astrapy import DataAPIClient
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

load_dotenv()

# -------------------------------
# AstraDB config
# -------------------------------
ASTRA_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE", "default_keyspace")
COLLECTION_NAME = "graduate_embedding"  # Collection สำหรับข้อมูลบัณฑิตศึกษา

if not ASTRA_TOKEN or not ASTRA_ENDPOINT:
    raise ValueError("Missing AstraDB credentials in .env")

# -------------------------------
# Graduate Program URLs
# -------------------------------
GRADUATE_URLS = {
    "overview": {
        "url": "https://computing.kku.ac.th/graduate",
        "name": "หลักสูตรบัณฑิตศึกษา",
        "type": "graduate_overview",
        "category": "graduate"
    },
    "master": {
        "url": "https://computing.kku.ac.th/master-cp",
        "name": "หลักสูตรปริญญาโท (Master)",
        "type": "master_program",
        "category": "master"
    },
    "phd": {
        "url": "https://computing.kku.ac.th/phd-cp",
        "name": "หลักสูตรปริญญาเอก (Ph.D.)",
        "type": "phd_program",
        "category": "phd"
    }
}

def clean_text(text):
    """ทำความสะอาดข้อความ"""
    if not text:
        return ""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that might cause issues
    text = text.strip()
    return text

def setup_driver():
    """ตั้งค่า Selenium WebDriver"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    
    # ใช้ relative path ในโปรเจค
    driver_path = os.path.join(os.path.dirname(__file__), "..", "drivers", "chromedriver.exe")
    service = Service(driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def extract_main_content(soup, url_info):
    """ดึงเนื้อหาหลักจากหน้าเว็บ"""
    documents = []
    
    # ดึง title หลัก
    main_title = ""
    title_tag = soup.find(['h1', 'h2'], class_=re.compile(r'title|heading', re.I))
    if title_tag:
        main_title = clean_text(title_tag.get_text())
    
    # ดึงเนื้อหาทั้งหมดจาก main content area
    content_areas = soup.find_all(['div', 'section', 'article'], 
                                    class_=re.compile(r'content|main|body|detail', re.I))
    
    if not content_areas:
        # ถ้าไม่พบ content area ให้ดึงจาก body
        content_areas = [soup.find('body')]
    
    for area in content_areas:
        if not area:
            continue
            
        # ดึงข้อความจากแต่ละส่วน
        sections = area.find_all(['section', 'div'], recursive=False)
        
        if not sections:
            # ถ้าไม่มี sections ให้ดึงเนื้อหาทั้งหมด
            text_content = clean_text(area.get_text())
            if text_content and len(text_content) > 50:
                metadata = {
                    "source": url_info["url"],
                    "type": url_info["type"],
                    "category": url_info["category"],
                    "program_name": url_info["name"],
                    "section": "main_content",
                    "title": main_title or url_info["name"]
                }
                documents.append(Document(page_content=text_content, metadata=metadata))
        else:
            # ประมวลผลแต่ละ section
            for i, section in enumerate(sections):
                # ดึง heading ของ section
                section_title = ""
                heading = section.find(['h1', 'h2', 'h3', 'h4'])
                if heading:
                    section_title = clean_text(heading.get_text())
                
                # ดึงเนื้อหาของ section
                section_text = clean_text(section.get_text())
                
                if section_text and len(section_text) > 30:
                    metadata = {
                        "source": url_info["url"],
                        "type": url_info["type"],
                        "category": url_info["category"],
                        "program_name": url_info["name"],
                        "section": f"section_{i+1}",
                        "section_title": section_title or f"Section {i+1}",
                        "title": main_title or url_info["name"]
                    }
                    documents.append(Document(page_content=section_text, metadata=metadata))
    
    return documents

def extract_structured_info(soup, url_info):
    """ดึงข้อมูลที่มีโครงสร้าง เช่น ตาราง, รายการ"""
    documents = []
    
    # ดึงข้อมูลจากตาราง
    tables = soup.find_all('table')
    for i, table in enumerate(tables):
        rows = table.find_all('tr')
        table_data = []
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            row_data = [clean_text(cell.get_text()) for cell in cells]
            if any(row_data):  # ถ้ามีข้อมูลในแถว
                table_data.append(' | '.join(row_data))
        
        if table_data:
            table_text = '\n'.join(table_data)
            metadata = {
                "source": url_info["url"],
                "type": f"{url_info['type']}_table",
                "category": url_info["category"],
                "program_name": url_info["name"],
                "section": f"table_{i+1}",
                "title": f"Table {i+1}"
            }
            documents.append(Document(page_content=table_text, metadata=metadata))
    
    # ดึงข้อมูลจากรายการ (lists)
    lists = soup.find_all(['ul', 'ol'])
    for i, list_elem in enumerate(lists):
        items = list_elem.find_all('li')
        list_data = [clean_text(item.get_text()) for item in items]
        
        if list_data and len(list_data) > 0:
            # กรองรายการที่มีข้อมูลที่มีความหมาย
            meaningful_items = [item for item in list_data if len(item) > 10]
            
            if meaningful_items:
                list_text = '\n'.join(f"• {item}" for item in meaningful_items)
                metadata = {
                    "source": url_info["url"],
                    "type": f"{url_info['type']}_list",
                    "category": url_info["category"],
                    "program_name": url_info["name"],
                    "section": f"list_{i+1}",
                    "title": f"List {i+1}"
                }
                documents.append(Document(page_content=list_text, metadata=metadata))
    
    return documents

def scrape_graduate_page(url_info):
    """สแครปข้อมูลจากหน้าเว็บหลักสูตรบัณฑิตศึกษา"""
    print(f"\n🌐 Scraping: {url_info['name']}")
    print(f"   URL: {url_info['url']}")
    
    driver = setup_driver()
    documents = []
    
    try:
        driver.get(url_info['url'])
        time.sleep(5)  # รอให้ JavaScript โหลด
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # ดึงเนื้อหาหลัก
        main_docs = extract_main_content(soup, url_info)
        documents.extend(main_docs)
        
        # ดึงข้อมูลที่มีโครงสร้าง
        structured_docs = extract_structured_info(soup, url_info)
        documents.extend(structured_docs)
        
        print(f"✅ Scraped {len(documents)} documents from {url_info['name']}")
        
    except Exception as e:
        print(f"❌ Error scraping {url_info['url']}: {e}")
    finally:
        driver.quit()
    
    return documents

def main():
    print("🚀 Starting Graduate Programs Data Ingestion to AstraDB...")
    print(f"🔑 Using endpoint: {ASTRA_ENDPOINT}")
    print(f"🏠 Using keyspace: {ASTRA_KEYSPACE}")
    
    # Connect to AstraDB
    try:
        client = DataAPIClient(token=ASTRA_TOKEN)
        db = client.get_database_by_api_endpoint(ASTRA_ENDPOINT)
        print("✅ Connected to AstraDB successfully!")
    except Exception as e:
        print(f"❌ Failed to connect to AstraDB: {e}")
        return False
    
    # Get or create collection
    try:
        existing_collections = list(db.list_collection_names())
        print(f"📂 Existing collections: {existing_collections}")
        
        if COLLECTION_NAME in existing_collections:
            collection = db.get_collection(COLLECTION_NAME)
            print(f"📂 Using existing collection: {COLLECTION_NAME}")
            
            # Clear existing data
            try:
                delete_result = collection.delete_many({})
                print(f"🗑️ Cleared existing data: {delete_result.deleted_count} documents")
            except Exception as e:
                print(f"⚠️ Could not clear existing data: {e}")
        else:
            print(f"❌ Collection {COLLECTION_NAME} not found!")
            print("Please create the collection via AstraDB UI with vector support:")
            print(f"  - Collection Name: {COLLECTION_NAME}")
            print("  - Vector Dimension: 384")
            print("  - Vector Metric: cosine")
            return False
            
    except Exception as e:
        print(f"❌ Failed to access collection: {e}")
        return False
    
    # Scrape all graduate program pages
    all_documents = []
    
    for key, url_info in GRADUATE_URLS.items():
        docs = scrape_graduate_page(url_info)
        all_documents.extend(docs)
    
    if len(all_documents) == 0:
        print("🚨 No graduate program data found for vector creation")
        return False
    
    print(f"\n📝 Total scraped documents: {len(all_documents)}")
    
    # แสดงตัวอย่างข้อมูล
    print("\n📋 Sample of scraped data:")
    for i, doc in enumerate(all_documents[:3]):
        print(f"\n--- Document {i+1} ---")
        print(f"Category: {doc.metadata.get('category')}")
        print(f"Type: {doc.metadata.get('type')}")
        print(f"Program: {doc.metadata.get('program_name')}")
        print(f"Content: {doc.page_content[:150]}...")
    
    # Split documents into chunks
    splitter = CharacterTextSplitter(
        chunk_size=500,  # Smaller chunks to avoid AstraDB 8000 byte limit
        chunk_overlap=50,
        separator="\n"
    )
    chunks = splitter.split_documents(all_documents)
    print(f"\n📄 Created {len(chunks)} chunks from {len(all_documents)} documents")
    
    # Initialize embeddings
    print("🧠 Initializing embeddings model...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Insert chunks into AstraDB
    print("💾 Inserting graduate program data into AstraDB...")
    
    batch_size = 50
    documents_to_insert = []
    skipped_count = 0
    
    for i, chunk in enumerate(chunks):
        # Check content size (AstraDB has 8000 byte limit)
        content_size = len(chunk.page_content.encode('utf-8'))
        if content_size > 7500:  # Leave some margin
            print(f"⚠️ Skipping chunk {i+1} - too large ({content_size} bytes)")
            skipped_count += 1
            continue
        
        # Generate embedding
        vector = embedding_model.embed_query(chunk.page_content)
        
        # Prepare document for insertion
        doc = {
            "_id": str(uuid.uuid4()),
            "content": chunk.page_content,
            "$vector": vector,
            "metadata": chunk.metadata
        }
        documents_to_insert.append(doc)
        
        # Insert in batches
        if len(documents_to_insert) >= batch_size or i == len(chunks) - 1:
            try:
                result = collection.insert_many(documents_to_insert)
                print(f"📊 Inserted {len(result.inserted_ids)} documents (chunk {i+1}/{len(chunks)})")
                documents_to_insert = []
            except Exception as e:
                print(f"❌ Failed to insert batch: {e}")
    
    if skipped_count > 0:
        print(f"⚠️ Skipped {skipped_count} chunks due to size limitations")
    
    # Verify insertion
    try:
        count = collection.count_documents({})
        print(f"\n🔍 Total graduate program documents in collection: {count}")
    except Exception as e:
        print(f"⚠️ Could not verify document count: {e}")
    
    # Show summary by category
    print("\n📋 Summary by program:")
    print("-" * 60)
    master_count = sum(1 for doc in all_documents if doc.metadata.get('category') == 'master')
    phd_count = sum(1 for doc in all_documents if doc.metadata.get('category') == 'phd')
    overview_count = sum(1 for doc in all_documents if doc.metadata.get('category') == 'graduate')
    
    print(f"• หลักสูตรบัณฑิตศึกษา (Overview): {overview_count} documents")
    print(f"• ปริญญาโท (Master): {master_count} documents")
    print(f"• ปริญญาเอก (Ph.D.): {phd_count} documents")
    print("-" * 60)
    
    print("🎉 Graduate programs data ingestion completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)

