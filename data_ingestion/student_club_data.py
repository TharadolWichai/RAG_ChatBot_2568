import requests
from bs4 import BeautifulSoup
import json
from dotenv import load_dotenv
import os
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from astrapy import DataAPIClient
import uuid
import urllib3
import time

# Selenium imports for JavaScript rendering
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("⚠️ Selenium not available - will use requests only")

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

from incremental_utils import enable_incremental_mode

def scrape_with_selenium():
    """ใช้ Selenium สแครปหน้าเว็บที่มี JavaScript"""
    url = "https://computing.kku.ac.th/students"
    
    # ตั้งค่า Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # รันแบบไม่เปิด browser
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--ignore-certificate-errors')
    chrome_options.add_argument('--ignore-ssl-errors')
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    
    driver = None
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
        
        # รอให้ JavaScript โหลดเสร็จ (รอ <li> elements ปรากฏ)
        print("   ⏳ รอ JavaScript rendering...")
        time.sleep(3)  # รอ 3 วินาที
        
        # Parse HTML ด้วย BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        links_data = []
        seen_urls = set()
        
        print("🔍 กำลังค้นหาลิงก์จาก <li> elements...")
        
        # ค้นหา <li> ทั้งหมดที่มี <a> tag
        all_li_elements = soup.find_all('li')
        print(f"   พบ <li> elements ทั้งหมด: {len(all_li_elements)}")
        
        for li in all_li_elements:
            # หา <a> tag ภายใน <li>
            link = li.find('a', href=True)
            
            if link:
                href = link.get('href', '').strip()
                
                # ดึงข้อความจาก <span> ภายใน <a> (ถ้ามี)
                span = link.find('span')
                if span:
                    text = span.get_text(strip=True)
                else:
                    # ถ้าไม่มี <span> ให้ใช้ text จาก <a> โดยตรง
                    text = link.get_text(strip=True)
                
                # กรองลิงก์ที่ไม่ต้องการ
                if not text or not href:
                    continue
                    
                # ข้ามลิงก์ที่เป็น # หรือ javascript
                if href.startswith('#') or href.startswith('javascript:'):
                    continue
                
                # ข้ามข้อความสั้นเกินไป
                if len(text) < 3:
                    continue
                    
                # กรองคำที่ไม่ต้องการ (navigation, login, etc.)
                skip_keywords = ['เข้าสู่ระบบ', 'login', 'ค้นหา', 'search', 'menu', 'home', 'logo', 'th', 'en']
                if any(keyword in text.lower() for keyword in skip_keywords):
                    continue
                
                # แปลง relative URL เป็น absolute URL
                if href.startswith('/'):
                    href = f"https://computing.kku.ac.th{href}"
                elif not href.startswith('http'):
                    if not href.startswith('https://') and not href.startswith('http://'):
                        href = f"https://computing.kku.ac.th/{href}"
                
                # เช็คว่าไม่ซ้ำ
                if href in seen_urls:
                    continue
                
                seen_urls.add(href)
                links_data.append({
                    "text": text,
                    "url": href,
                    "keywords": extract_keywords_from_text(text)
                })
                
                print(f"   ✅ พบ: {text[:60]} -> {href}")
        
        print(f"\n📊 พบลิงก์จาก Selenium scraping: {len(links_data)} ลิงก์")
        
        # แสดงลิงก์ที่พบทั้งหมด
        if links_data:
            print("\n📋 รายการลิงก์ที่สแครปได้:")
            for i, link_data in enumerate(links_data, 1):
                print(f"  {i:2d}. {link_data['text'][:50]:<50} -> {link_data['url']}")
        
        return links_data
        
    finally:
        # ปิด browser
        if driver:
            driver.quit()
            print("   🔚 ปิด Chrome driver แล้ว")

def scrape_students_page():
    """สแครปข้อมูลจากหน้า students ตาม HTML structure ที่มี <li> -> <a> -> <span>"""
    print("🌐 Scraping students page...")
    
    # ลอง Selenium ก่อน (ถ้ามี) เพื่อรองรับ JavaScript rendering
    if SELENIUM_AVAILABLE:
        try:
            print("🔧 Using Selenium for JavaScript rendering...")
            return scrape_with_selenium()
        except Exception as e:
            print(f"⚠️ Selenium failed: {e}")
            print("   Falling back to requests...")
    
    # Fallback to requests
    try:
        url = "https://computing.kku.ac.th/students"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'th-TH,th;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br'
        }
        
        response = requests.get(url, headers=headers, verify=False, timeout=15)
        response.encoding = 'utf-8'
        
        if response.status_code != 200:
            print(f"❌ HTTP Error: {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        links_data = []
        seen_urls = set()
        
        print("🔍 กำลังค้นหาลิงก์จาก <li> elements...")
        
        # ค้นหา <li> ทั้งหมดที่มี <a> tag
        all_li_elements = soup.find_all('li')
        print(f"   พบ <li> elements ทั้งหมด: {len(all_li_elements)}")
        
        for li in all_li_elements:
            # หา <a> tag ภายใน <li>
            link = li.find('a', href=True)
            
            if link:
                href = link.get('href', '').strip()
                
                # ดึงข้อความจาก <span> ภายใน <a> (ถ้ามี)
                span = link.find('span')
                if span:
                    text = span.get_text(strip=True)
                else:
                    # ถ้าไม่มี <span> ให้ใช้ text จาก <a> โดยตรง
                    text = link.get_text(strip=True)
                
                # กรองลิงก์ที่ไม่ต้องการ
                if not text or not href:
                    continue
                    
                # ข้ามลิงก์ที่เป็น # หรือ javascript
                if href.startswith('#') or href.startswith('javascript:'):
                    continue
                
                # ข้ามข้อความสั้นเกินไป หรือเป็น placeholder
                if len(text) < 3:
                    continue
                    
                # กรองคำที่ไม่ต้องการ (navigation, login, etc.)
                skip_keywords = ['เข้าสู่ระบบ', 'login', 'ค้นหา', 'search', 'menu', 'home', 'logo']
                if any(keyword in text.lower() for keyword in skip_keywords):
                    continue
                
                # แปลง relative URL เป็น absolute URL
                if href.startswith('/'):
                    href = f"https://computing.kku.ac.th{href}"
                elif not href.startswith('http'):
                    # ข้าม relative path ที่ไม่มี http
                    if not href.startswith('https://') and not href.startswith('http://'):
                        href = f"https://computing.kku.ac.th/{href}"
                
                # เช็คว่าไม่ซ้ำ
                if href in seen_urls:
                    continue
                
                seen_urls.add(href)
                links_data.append({
                    "text": text,
                    "url": href,
                    "keywords": extract_keywords_from_text(text)
                })
                
                print(f"   ✅ พบ: {text[:60]} -> {href}")
        
        print(f"\n📊 พบลิงก์จาก scraping: {len(links_data)} ลิงก์")
        
        # แสดงลิงก์ที่พบทั้งหมด
        if links_data:
            print("\n📋 รายการลิงก์ที่สแครปได้:")
            for i, link_data in enumerate(links_data, 1):
                print(f"  {i:2d}. {link_data['text'][:50]:<50} -> {link_data['url']}")
        
        return links_data
        
    except Exception as e:
        print(f"❌ Error scraping students page: {e}")
        import traceback
        traceback.print_exc()
        return []

def extract_keywords_from_text(text):
    """สร้างคำสำคัญจากข้อความ"""
    # คำสำคัญพื้นฐาน
    keywords = [text.lower()]
    
    # แยกคำตามช่องว่าง
    words = text.split()
    keywords.extend([word.lower() for word in words if len(word) > 2])
    
    # เพิ่มคำสำคัญเฉพาะสำหรับนักศึกษา
    student_keywords = {
        "นักศึกษา": ["student", "students"],
        "ทุน": ["scholarship", "grant"],
        "กิจกรรม": ["activity", "activities"],
        "ชมรม": ["club", "society"],
        "การศึกษา": ["education", "academic"],
        "ปริญญา": ["degree", "graduation"],
        "วิชาการ": ["academic", "course"],
        "สมัคร": ["apply", "application", "register"],
        "ลงทะเบียน": ["register", "registration", "enroll"],
        "ตารางสอน": ["schedule", "timetable"],
        "เกรด": ["grade", "score"],
        "ผลการเรียน": ["result", "transcript"],
        "แบบฟอร์ม": ["form", "document"],
        "ระบบ": ["system", "online"]
    }
    
    text_lower = text.lower()
    for thai_word, eng_words in student_keywords.items():
        if thai_word in text_lower:
            keywords.extend(eng_words)
    
    # ลบคำซ้ำ
    return list(set(keywords))


def main():
    print("🚀 Starting Students data scraping and AstraDB ingestion...")
    
    # Check environment variables
    token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
    keyspace = os.getenv("ASTRA_DB_KEYSPACE", "default_keyspace")
    
    if not token or not api_endpoint:
        print("❌ Error: Missing AstraDB credentials in .env file")
        print("Please add:")
        print("ASTRA_DB_APPLICATION_TOKEN=your_token_here")
        print("ASTRA_DB_API_ENDPOINT=your_endpoint_here")
        return False
    
    print(f"🔑 Using endpoint: {api_endpoint}")
    print(f"🏠 Using keyspace: {keyspace}")
    
    # Initialize AstraDB client
    try:
        client = DataAPIClient(token=token)
        database = client.get_database_by_api_endpoint(api_endpoint)
        print("✅ Connected to AstraDB successfully!")
    except Exception as e:
        print(f"❌ Failed to connect to AstraDB: {e}")
        return False
    
    # Use services collection (รวมกับ contact และ links)
    collection_name = "services_embedding"
    try:
        # List existing collections first
        existing_collections = list(database.list_collection_names())
        print(f"📂 Existing collections: {existing_collections}")
        
        if collection_name in existing_collections:
            collection = database.get_collection(collection_name)
            print(f"📂 Using existing collection: {collection_name}")
            
            # Clear only student links (not all data)
            print("🗑️ Clearing existing student links from collection...")
            try:
                # Delete only documents with category="students"
                delete_result = collection.delete_many({"metadata.category": "students"})
                print(f"🗑️ Deleted {delete_result.deleted_count if hasattr(delete_result, 'deleted_count') else 'existing'} student link documents")
            except Exception as e:
                print(f"⚠️ Warning: Could not clear student links: {e}")
                
        else:
            print(f"❌ Collection {collection_name} not found!")
            print("Please create the collection via AstraDB UI with vector support:")
            print("  - Collection Name: services_embedding")
            print("  - Vector Dimension: 384")
            print("  - Vector Metric: cosine")
            return False
            
    except Exception as e:
        print(f"❌ Failed to access collection: {e}")
        return False
    
    # Scrape students page
    links_data = scrape_students_page()
    
    # Check if scraping was successful
    if len(links_data) == 0:
        print("❌ No links found from scraping!")
        return False
    
    # Create documents
    print("📚 Creating documents from students links...")
    try:
        docs = []
        link_count = 0
        
        for link_data in links_data:
            link_text = link_data["text"]
            link_url = link_data["url"]
            keywords = link_data["keywords"]
            
            # Create comprehensive content for better search
            content_parts = [
                f"ชื่อลิงก์: {link_text}",
                f"URL: {link_url}",
                f"ประเภท: ลิงก์บริการสำหรับนักศึกษา คณะคอมพิวเตอร์"
            ]
            
            if keywords:
                content_parts.append(f"คำสำคัญ: {', '.join(keywords)}")
            
            content = "\n".join(content_parts)
            
            # Create metadata with category
            metadata = {
                "link_text": link_text,
                "url": link_url,
                "type": "service_link",
                "keywords": keywords,
                "category": "students",  # Category สำหรับแยกประเภท
                "subcategory": "student_services"
            }
            
            docs.append(Document(page_content=content, metadata=metadata))
            link_count += 1
            print(f"✅ Created student link {link_count}: {link_text}")
        
        print(f"📊 Total student links created: {len(docs)}")
        
    except Exception as e:
        print(f"❌ Failed to create students links data: {e}")
        return False
    
    if len(docs) == 0:
        print("🚨 No student links found for vector creation")
        return False

    print(f"📝 Processed {len(docs)} student link documents")

    # Initialize embeddings
    print("🧠 Initializing embeddings model...")
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Use incremental indexing
    print("\n🚀 Starting incremental indexing...")
    try:
        stats = enable_incremental_mode(
            collection=collection,
            embedding_model=embedding,
            new_documents=docs,
            metadata_filter={"category": "students"},  # Filter สำหรับดึงเอกสารนักศึกษา
            hash_keys=["link_text", "url"],  # Keys สำหรับสร้าง unique hash
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
        print(f"🔍 Total documents in students collection: {count}")
    except Exception as e:
        print(f"⚠️ Could not verify document count: {e}")
    
    print("🎉 Students data scraping and AstraDB ingestion completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)