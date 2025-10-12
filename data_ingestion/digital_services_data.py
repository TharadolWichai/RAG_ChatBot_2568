# digital_services_data.py - Digital Services Data Ingestion via Web Scraping
# ดึงข้อมูลบริการดิจิตอลสำหรับนักศึกษาและบุคลากรผ่าน Web Scraping

import os
import uuid
import re
import time
import urllib3
from typing import List, Dict, Any

from astrapy import DataAPIClient
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Selenium imports
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from bs4 import BeautifulSoup
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("⚠️ Selenium not available")

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

# -------------------------------
# Digital Services Web URLs
# -------------------------------
DIGITAL_SERVICES = {
    "web_hosting": {
        "name": "บริการ Web Hosting",
        "name_en": "Digital Services - Web Hosting",
        "url": "https://computing.kku.ac.th/digitalsv-webhosting",
        "category": "hosting",
        "keywords": ["web hosting", "เว็บโฮสติ้ง", "โฮสติ้ง", "เว็บไซต์", "website"]
    },
    "virtual_machine": {
        "name": "บริการ Virtual Machine",
        "name_en": "Virtual Machine Service",
        "url": "https://computing.kku.ac.th/virtual-machine",
        "category": "infrastructure",
        "keywords": ["virtual machine", "vm", "เครื่องเสมือน", "virtual", "คลาวด์"]
    },
    "apple_store": {
        "name": "บริการ Apple Store for Education",
        "name_en": "Apple Store for Education",
        "url": "https://computing.kku.ac.th/apple-store",
        "category": "software",
        "keywords": ["apple", "app store", "แอปเปิล", "แอพสโตร์", "ios"]
    },
    "google_play": {
        "name": "บริการ Google Play for Education",
        "name_en": "Google Play for Education",
        "url": "https://computing.kku.ac.th/google-play",
        "category": "software",
        "keywords": ["google play", "กูเกิลเพลย์", "แอนดรอยด์", "android", "app"]
    },
    "grammarly": {
        "name": "บริการ Grammarly Premium",
        "name_en": "Grammarly Premium Service",
        "url": "https://computing.kku.ac.th/grammarly",
        "category": "software",
        "keywords": ["grammarly", "แกรมมารลี่", "ตรวจสอบภาษา", "grammar", "writing"]
    },
    "chatgpt_plus": {
        "name": "บริการ ChatGPT Plus",
        "name_en": "ChatGPT Plus Service",
        "url": "https://computing.kku.ac.th/chatgpt-plus",
        "category": "ai",
        "keywords": ["chatgpt", "แชทจีพีที", "ai", "ปัญญาประดิษฐ์", "gpt"]
    },
    "server_ds_ai": {
        "name": "บริการ Server สำหรับ Data Science และ AI",
        "name_en": "Data Science and AI Server",
        "url": "https://computing.kku.ac.th/server-ds-ai",
        "category": "infrastructure",
        "keywords": ["data science", "ai server", "เซิร์ฟเวอร์", "machine learning", "ข้อมูล"]
    },
    "kku_snapdrop": {
        "name": "บริการ KKU Snapdrop",
        "name_en": "KKU Snapdrop Service",
        "url": "https://computing.kku.ac.th/kku-snapdrop",
        "category": "utility",
        "keywords": ["snapdrop", "แชร์ไฟล์", "file sharing", "transfer", "ส่งไฟล์"]
    }
}

# -------------------------------
# AstraDB Config
# -------------------------------
ASTRA_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE", "default_keyspace")
COLLECTION_NAME = "digital_services_embedding"

if not ASTRA_TOKEN or not ASTRA_ENDPOINT:
    raise ValueError("❌ Missing AstraDB credentials in .env")

# -------------------------------
# Helper Functions
# -------------------------------
def clean_html_text(text: str) -> str:
    """ทำความสะอาดข้อความจาก HTML tags"""
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Replace &nbsp; with space
    text = text.replace('&nbsp;', ' ')
    # Replace multiple newlines with single newline
    text = re.sub(r'\n+', '\n', text)
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def scrape_service_with_selenium(service_info: Dict) -> List[Document]:
    """ใช้ Selenium สแครปหน้าบริการดิจิตอล และแยกตามโครงสร้าง (มี 4 ส่วน)"""
    
    if not SELENIUM_AVAILABLE:
        print("   ❌ Selenium not available")
        return []
    
    url = service_info['url']
    
    # ตั้งค่า Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--ignore-certificate-errors')
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    
    # ใช้ chromedriver จากโฟลเดอร์ drivers
    driver_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'drivers', 'chromedriver.exe')
    
    driver = None
    documents = []
    
    try:
        # เริ่ม Chrome driver
        service = Service(driver_path)
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        print(f"   🌐 กำลังโหลดหน้าเว็บ: {url}")
        driver.get(url)
        
        # รอให้ JavaScript โหลดเสร็จ
        print("   ⏳ รอ JavaScript rendering...")
        time.sleep(5)  # รอ 5 วินาที
        
        # Parse HTML ด้วย BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # 1. สร้าง Document หลัก (Overview)
        main_content_parts = [
            f"ชื่อบริการ: {service_info['name']}",
            f"ชื่อภาษาอังกฤษ: {service_info['name_en']}",
            f"หมวดหมู่: {service_info['category']}",
            f"URL: {url}",
            f"คำสำคัญ: {', '.join(service_info['keywords'])}"
        ]
        
        # ดึง title จาก h1 แรก
        h1_title = soup.find('h1')
        if h1_title:
            title_text = clean_html_text(h1_title.get_text())
            if title_text:
                main_content_parts.append(f"รายละเอียดสั้นๆ: {title_text}")
        
        main_content = "\n".join(main_content_parts)
        
        main_metadata = {
            "service_name": service_info['name'],
            "service_name_en": service_info['name_en'],
            "url": url,
            "category": service_info['category'],
            "keywords": service_info['keywords'],
            "type": "digital_service",
            "section": "overview"
        }
        
        documents.append(Document(page_content=main_content, metadata=main_metadata))
        print(f"   ✅ สร้าง overview document")
        
        # 2. ค้นหาส่วนต่างๆ ตามโครงสร้างในรูป
        # หาส่วนที่มี <p><span><strong>ส่วนที่ X</strong></span></p>
        sections_found = []
        
        # หา <strong> ทั้งหมดที่มีคำว่า "ส่วนที่"
        strong_tags = soup.find_all('strong')
        
        for strong_tag in strong_tags:
            section_text = strong_tag.get_text(strip=True)
            
            # เช็คว่าเป็นหัวข้อส่วน
            if 'ส่วนที่' in section_text or 'ข้อตกลง' in section_text or 'เทคโนโลยี' in section_text or 'คำแนะ' in section_text or 'ติดต่อ' in section_text:
                
                section_title = clean_html_text(section_text)
                print(f"   📍 พบส่วน: {section_title}")
                
                # หาเนื้อหาของส่วนนี้
                section_content_parts = [section_title]
                
                # หา parent element แล้วเอาเนื้อหาข้างล่าง
                parent = strong_tag.find_parent(['p', 'h1', 'h2', 'h3', 'div'])
                
                if parent:
                    # หา siblings ถัดไป (ol, ul, p)
                    next_elements = parent.find_next_siblings()
                    
                    for next_elem in next_elements:
                        # ถ้าเจอ strong tag ใหม่ที่เป็นหัวข้อส่วนถัดไป ให้หยุด
                        if next_elem.find('strong') and ('ส่วนที่' in next_elem.get_text() or 'ข้อตกลง' in next_elem.get_text() or 'เทคโนโลยี' in next_elem.get_text()):
                            break
                        
                        # ดึงเนื้อหา
                        if next_elem.name in ['ol', 'ul']:
                            # ถ้าเป็น list ให้ดึง li ทั้งหมด
                            for li in next_elem.find_all('li'):
                                li_text = clean_html_text(li.get_text())
                                if li_text and len(li_text) > 5:
                                    section_content_parts.append(f"• {li_text}")
                        
                        elif next_elem.name == 'p':
                            p_text = clean_html_text(next_elem.get_text())
                            if p_text and len(p_text) > 5:
                                section_content_parts.append(p_text)
                        
                        # จำกัดความยาวของแต่ละส่วน (ไม่เกิน 20 items)
                        if len(section_content_parts) > 25:
                            break
                
                # สร้าง document สำหรับส่วนนี้
                if len(section_content_parts) > 1:  # มีเนื้อหาเพิ่มจาก title
                    section_content = "\n".join(section_content_parts)
                    
                    section_metadata = {
                        "service_name": service_info['name'],
                        "service_name_en": service_info['name_en'],
                        "url": url,
                        "category": service_info['category'],
                        "keywords": service_info['keywords'],
                        "type": "digital_service",
                        "section": section_title.lower().replace(' ', '_')
                    }
                    
                    documents.append(Document(page_content=section_content, metadata=section_metadata))
                    print(f"   ✅ สร้าง document สำหรับส่วน: {section_title}")
                    sections_found.append(section_title)
        
        print(f"   📝 รวม {len(documents)} documents ({len(sections_found)} sections)")
        
        return documents
        
    except Exception as e:
        print(f"   ❌ Error scraping: {e}")
        import traceback
        traceback.print_exc()
        return []
        
    finally:
        # ปิด browser
        if driver:
            driver.quit()
            print("   🔚 ปิด Chrome driver แล้ว")

# -------------------------------
# Main Function
# -------------------------------
def main():
    print("🚀 Starting Digital Services Data Ingestion via Web Scraping...")
    print(f"🔑 Using endpoint: {ASTRA_ENDPOINT}")
    print(f"🏠 Using keyspace: {ASTRA_KEYSPACE}")
    print()
    
    # Initialize AstraDB client
    try:
        client = DataAPIClient(token=ASTRA_TOKEN)
        database = client.get_database_by_api_endpoint(ASTRA_ENDPOINT)
        print("✅ Connected to AstraDB successfully!")
    except Exception as e:
        print(f"❌ Failed to connect to AstraDB: {e}")
        return False
    
    # Get or create collection
    try:
        existing_collections = list(database.list_collection_names())
        print(f"📂 Existing collections: {existing_collections}")
        
        if COLLECTION_NAME in existing_collections:
            collection = database.get_collection(COLLECTION_NAME)
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
    
    # Scrape all services
    all_documents = []
    services_processed = []
    
    print(f"\n{'='*60}")
    print(f"📊 Processing {len(DIGITAL_SERVICES)} services via Web Scraping...")
    print(f"{'='*60}\n")
    
    for service_key, service_info in DIGITAL_SERVICES.items():
        print(f"\n{'='*60}")
        print(f"🔸 Processing: {service_info['name']}")
        print(f"{'='*60}")
        
        # Scrape service page
        documents = scrape_service_with_selenium(service_info)
        
        if documents:
            all_documents.extend(documents)
            services_processed.append(service_info)
            print(f"✅ Processed: {service_info['name']} ({len(documents)} documents)")
        else:
            print(f"⚠️ No documents created for {service_info['name']}")
    
    if not all_documents:
        print("\n❌ No documents created from any service!")
        return False
    
    print(f"\n📝 Total documents from all services: {len(all_documents)}")
    
    # Split documents (smaller chunks for AstraDB limit)
    print("\n📄 Splitting documents into chunks...")
    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separator="\n"
    )
    chunks = splitter.split_documents(all_documents)
    print(f"📄 Created {len(chunks)} chunks from {len(all_documents)} documents")
    
    # Initialize embeddings
    print("\n🧠 Initializing embeddings model...")
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Generate embeddings and insert to AstraDB
    print("💾 Inserting service data into AstraDB...")
    
    documents_to_insert = []
    skipped_count = 0
    
    for i, chunk in enumerate(chunks):
        # Check content size (AstraDB has 8000 byte limit)
        content_size = len(chunk.page_content.encode('utf-8'))
        
        if content_size > 7500:
            print(f"⚠️ Skipping chunk {i+1} - too large ({content_size} bytes)")
            skipped_count += 1
            continue
        
        # Generate embedding
        vector = embedding.embed_query(chunk.page_content)
        
        # Prepare document for insertion
        doc = {
            "_id": str(uuid.uuid4()),
            "content": chunk.page_content,
            "$vector": vector,
            "metadata": chunk.metadata
        }
        documents_to_insert.append(doc)
        
        if (i + 1) % 10 == 0:
            print(f"📊 Processed {i+1}/{len(chunks)} chunks...")
    
    if skipped_count > 0:
        print(f"⚠️ Skipped {skipped_count} chunks due to size limitations")
    
    print(f"\n💾 Inserting {len(documents_to_insert)} documents into AstraDB...")
    
    # Insert all documents
    try:
        result = collection.insert_many(documents_to_insert)
        print(f"✅ Successfully inserted {len(result.inserted_ids)} service documents into AstraDB!")
    except Exception as e:
        print(f"❌ Failed to insert documents: {e}")
        return False
    
    # Verify insertion
    try:
        count = collection.count_documents({})
        print(f"🔍 Total documents in collection: {count}")
    except Exception as e:
        print(f"⚠️ Could not verify document count: {e}")
    
    # Show summary
    print("\n" + "="*60)
    print("📋 Summary of digital services processed:")
    print("="*60)
    for service_info in services_processed:
        print(f"✅ {service_info['name']}")
        print(f"   Category: {service_info['category']}")
        print(f"   URL: {service_info['url']}")
        print("-" * 60)
    
    print(f"\n🎉 Digital Services data ingestion completed successfully!")
    print(f"📊 Processed: {len(services_processed)}/{len(DIGITAL_SERVICES)} services")
    print(f"📄 Total chunks inserted: {len(documents_to_insert)}")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
