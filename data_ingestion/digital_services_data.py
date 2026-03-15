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
from incremental_utils import enable_incremental_mode

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
# -------------------------------
# Auto-discover Services from Main Page
# -------------------------------
def auto_discover_services() -> Dict:
    """
    Auto-discover บริการทั้งหมดจากหน้า digital-services หลัก
    ไม่ต้อง hardcode รายการบริการ - ระบบจะหาเองอัตโนมัติ
    """
    if not SELENIUM_AVAILABLE:
        print("⚠️ Selenium not available, using fallback services")
        return FALLBACK_SERVICES
    
    main_url = "https://computing.kku.ac.th/digital-services"
    
    # ตั้งค่า Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--ignore-certificate-errors')
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    
    driver = None
    discovered_services = {}
    
    try:
        # Setup ChromeDriver
        try:
            from selenium.webdriver.chrome.service import Service as ChromeService
            from webdriver_manager.chrome import ChromeDriverManager
            service = ChromeService(ChromeDriverManager().install())
        except ImportError:
            driver_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'drivers', 'chromedriver.exe')
            if os.path.exists(driver_path):
                service = Service(driver_path)
            else:
                raise FileNotFoundError("ChromeDriver not found")
        
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        print(f"🔍 Auto-discovering services from: {main_url}")
        driver.get(main_url)
        time.sleep(5)  # รอให้หน้าโหลดเสร็จ
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # หา links ที่เป็นบริการจริงๆ จากหน้า digital-services
        # Strategy: หาเฉพาะ cards/boxes ที่เป็นบริการ (มักจะมี structure พิเศษ)
        
        # หาจาก service cards หรือ service items (ปรับตาม structure ของเว็บ)
        service_cards = soup.find_all(['div', 'a'], class_=lambda x: x and any(
            keyword in str(x).lower() for keyword in ['service', 'card', 'item', 'box']
        ))
        
        # ถ้าหาแบบ class ไม่เจอ ให้หาจาก links ทั้งหมด แต่กรองเฉพาะที่มี pattern ของ digital services
        if not service_cards:
            service_cards = soup.find_all('a', href=True)
        
        for card in service_cards:
            # ดึง link จาก card (อาจจะเป็น <a> เองหรือมี <a> ข้างใน)
            if card.name == 'a':
                link = card
            else:
                link = card.find('a', href=True)
                if not link:
                    continue
            
            href = link.get('href', '')
            text = link.get_text().strip()
            
            # Filter 1: ต้องมี href และ text ที่มีความหมาย
            if not href or not text or len(text) < 3:
                continue
            
            # Filter 2: สร้าง full URL
            if href.startswith('/'):
                full_url = f"https://computing.kku.ac.th{href}"
            elif href.startswith('http'):
                full_url = href
            else:
                continue
            
            # Filter 3: ต้องเป็น URL ใน computing.kku.ac.th และไม่ใช่หน้าหลัก
            if 'computing.kku.ac.th' not in full_url or full_url == main_url:
                continue
            
            # Filter 4: Skip URLs ที่ไม่ใช่บริการ (เพิ่มรายการให้เข้มงวด)
            skip_patterns = [
                # Navigation & Structure
                '/home', '/about', '/contact', '/vision', '/mission', '/history',
                '/structure', '/institution', '/board', '/facilities',
                # Academic
                '/academics', '/course', '/admission', '/entrance', '/graduate',
                '/scholarship', '/bsc-', '/msc-', '/phd-',
                # People & Community
                '/people', '/students', '/staffs', '/alumni', '/club',
                '/international-student', '/for-staffs',
                # Other
                '/news', '/research', '/publication', '/project',
                '/event', '/gallery', '/download', '/document',
                '/en/', '/th/', '#', 'javascript:', 'mailto:', 'tel:',
                '/login', '/register', 'facebook.com', 'twitter.com', 'youtube.com',
                # Specific non-services
                '/mikrotik', '/content/', '/cp-'
            ]
            
            if any(pattern in full_url.lower() for pattern in skip_patterns):
                continue
            
            # Filter 5: ต้องมี pattern ที่บ่งบอกว่าเป็น digital service
            # หรืออย่างน้อยต้องไม่อยู่ใน skip list ด้านบน
            service_indicators = [
                'digital', 'service', 'hosting', 'virtual', 'vm', 'cloud',
                'apple', 'google', 'grammarly', 'chatgpt', 'gpt', 'ai',
                'server', 'nas', 'storage', 'gpu', 'h100', 'snapdrop',
                'backup', 'database', 'api', 'app', 'software'
            ]
            
            # ต้องมีคำที่บ่งบอกว่าเป็นบริการอย่างน้อย 1 คำ
            has_service_indicator = any(
                indicator in full_url.lower() or indicator in text.lower()
                for indicator in service_indicators
            )
            
            if not has_service_indicator:
                continue
            
            # ผ่านการกรองแล้ว
            if 'computing.kku.ac.th' in full_url and full_url != main_url:
                    # สร้าง slug จาก URL
                    slug = href.split('/')[-1].strip()
                    if not slug:
                        slug = href.split('/')[-2].strip()
                    
                    # ทำความสะอาดชื่อ
                    clean_name = re.sub(r'\s+', ' ', text).strip()
                    
                    # ถ้ายังไม่มีใน dict และชื่อไม่ซ้ำ
                    if slug and clean_name and slug not in discovered_services:
                        # Categorize based on keywords
                        category = "service"
                        keywords = [clean_name.lower()]
                        
                        if any(word in clean_name.lower() for word in ['host', 'โฮส']):
                            category = "hosting"
                            keywords.extend(["hosting", "web hosting"])
                        elif any(word in clean_name.lower() for word in ['virtual', 'vm', 'เครื่องเสมือน']):
                            category = "infrastructure"
                            keywords.extend(["virtual machine", "vm"])
                        elif any(word in clean_name.lower() for word in ['apple', 'แอปเปิล']):
                            category = "software"
                            keywords.extend(["apple", "ios"])
                        elif any(word in clean_name.lower() for word in ['google', 'กูเกิล']):
                            category = "software"
                            keywords.extend(["google", "android"])
                        elif any(word in clean_name.lower() for word in ['ai', 'ปัญญา', 'chatgpt', 'gpt']):
                            category = "ai"
                            keywords.extend(["ai", "artificial intelligence"])
                        elif any(word in clean_name.lower() for word in ['server', 'เซิร์ฟเวอร์']):
                            category = "infrastructure"
                            keywords.extend(["server", "computing"])
                        elif any(word in clean_name.lower() for word in ['nas', 'storage', 'จัดเก็บ']):
                            category = "storage"
                            keywords.extend(["storage", "backup"])
                        elif any(word in clean_name.lower() for word in ['gpu', 'h100', 'graphics']):
                            category = "infrastructure"
                            keywords.extend(["gpu", "graphics", "computing"])
                        
                        discovered_services[slug] = {
                            "name": clean_name,
                            "name_en": clean_name,  # จะถูกปรับในภายหลังถ้าจำเป็น
                            "url": full_url,
                            "category": category,
                            "keywords": keywords,
                            "auto_discovered": True
                        }
        
        driver.quit()
        
        # ถ้าไม่พบบริการเลย ใช้ fallback
        if not discovered_services:
            print("⚠️ No services discovered, using fallback")
            return FALLBACK_SERVICES
        
        print(f"✅ Discovered {len(discovered_services)} services:")
        for slug, info in discovered_services.items():
            print(f"   • {info['name']} - {info['url']}")
        
        return discovered_services
        
    except Exception as e:
        print(f"❌ Error during auto-discovery: {e}")
        if driver:
            driver.quit()
        print("⚠️ Falling back to default services")
        return FALLBACK_SERVICES


# -------------------------------
# Fallback Services (ใช้ตอน auto-discovery ล้มเหลว)
# -------------------------------
FALLBACK_SERVICES = {
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
    "nas": {
        "name": "บริการ NAS (Network Attached Storage)",
        "name_en": "NAS Service",
        "url": "https://computing.kku.ac.th/nas",
        "category": "storage",
        "keywords": ["nas", "network storage", "จัดเก็บข้อมูล", "storage", "backup"]
    },
    "kku_snapdrop": {
        "name": "บริการ KKU Snapdrop",
        "name_en": "KKU Snapdrop Service",
        "url": "https://computing.kku.ac.th/kku-snapdrop",
        "category": "utility",
        "keywords": ["snapdrop", "แชร์ไฟล์", "file sharing", "transfer", "ส่งไฟล์"]
    },
    "h100": {
        "name": "บริการ H100 GPU",
        "name_en": "H100 GPU Service",
        "url": "https://computing.kku.ac.th/h100",
        "category": "infrastructure",
        "keywords": ["h100", "gpu", "nvidia", "graphics", "ประมวลผล"]
    }
}

# -------------------------------
# Digital Services (Auto-discover or Fallback)
# -------------------------------
print("\n" + "="*60)
print("🔍 Auto-discovering Digital Services...")
print("="*60)

try:
    DIGITAL_SERVICES = auto_discover_services()
    print(f"✅ Loaded {len(DIGITAL_SERVICES)} services")
except Exception as e:
    print(f"❌ Auto-discovery failed: {e}")
    print("⚠️ Using fallback services")
    DIGITAL_SERVICES = FALLBACK_SERVICES

print("="*60 + "\n")

# -------------------------------
# AstraDB Config
# -------------------------------
ASTRA_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE", "default_keyspace")
COLLECTION_NAME = "newdigital_services_embedding"  # NEW: Changed to use multilingual-e5-large (1024 dim)

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
    
    driver = None
    documents = []
    
    try:
        # ลองใช้ webdriver-manager เพื่อดาวน์โหลด ChromeDriver อัตโนมัติ
        try:
            from selenium.webdriver.chrome.service import Service as ChromeService
            from webdriver_manager.chrome import ChromeDriverManager
            service = ChromeService(ChromeDriverManager().install())
            print("   ✅ ใช้ webdriver-manager สำหรับ ChromeDriver")
        except ImportError:
            # Fallback: ใช้ chromedriver จากโฟลเดอร์ drivers
            driver_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'drivers', 'chromedriver.exe')
            if os.path.exists(driver_path):
                service = Service(driver_path)
                print(f"   ⚠️  ใช้ ChromeDriver จากโฟลเดอร์ drivers (อาจเวอร์ชันไม่ตรง)")
            else:
                raise FileNotFoundError(
                    f"ChromeDriver not found at {driver_path}. "
                    "Please install webdriver-manager: pip install webdriver-manager"
                )
        
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
            print("  - Vector Dimension: 1024")  # Updated for multilingual-e5-large
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
    
    # Initialize embeddings - Using multilingual-e5-large for better Thai support
    print("\n🧠 Initializing embeddings model (multilingual-e5-large)...")
    print("   ⏳ First run may take 5-10 minutes to download model (~2.2GB)")
    embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    print("   ✅ Model loaded successfully!")
    
    # Use incremental indexing
    print("\n🚀 Starting incremental indexing...")
    try:
        stats = enable_incremental_mode(
            collection=collection,
            embedding_model=embedding,
            new_documents=all_documents,
            metadata_filter={"type": "digital_service"},  # Filter สำหรับดึงเอกสารบริการดิจิทัล
            hash_keys=["service_name", "category"],  # Keys สำหรับสร้าง unique hash
            delete_missing=False  # ไม่ลบเอกสารเก่า
        )
        
        print(f"\n✅ Incremental indexing completed!")
        print(f"   - New documents inserted: {stats['inserted']}")
        print(f"   - Existing documents skipped: {stats['skipped']}")
        
    except Exception as e:
        print(f"❌ Failed to process incremental indexing: {e}")
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
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
