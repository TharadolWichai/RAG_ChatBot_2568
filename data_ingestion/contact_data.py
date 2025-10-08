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
COLLECTION_NAME = "services_embedding"  # รวมกับ links และ students

if not ASTRA_TOKEN or not ASTRA_ENDPOINT:
    raise ValueError("Missing AstraDB credentials in .env")

# -------------------------------
# Selenium setup
# -------------------------------
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
# ใช้ relative path ในโปรเจค
driver_path = os.path.join(os.path.dirname(__file__), "..", "drivers", "chromedriver.exe")
service = Service(driver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

# URL ของหน้าติดต่อ (ต้องแก้ไข URL ให้ถูกต้อง)
url = "https://computing.kku.ac.th/contact-us"  # หรือ URL ที่ถูกต้อง
driver.get(url)
time.sleep(5)  # รอโหลด JS
soup = BeautifulSoup(driver.page_source, "html.parser")
driver.quit()

# -------------------------------
# Scrape contact information
# -------------------------------
docs = []

def extract_contact_info():
    """ดึงข้อมูลติดต่อจากโครงสร้าง HTML ตามที่เห็นในรูป - ปรับปรุงให้ scrape ตารางอย่างถูกต้อง"""
    contact_sections = []
    
    # 1. ดึงข้อมูลหัวข้อหลักของวิทยาลัย
    college_info = extract_college_basic_info()
    if college_info:
        contact_sections.extend(college_info)
    
    # 2. ดึงข้อมูลบุคลากรจากตาราง
    staff_info = extract_staff_table_info()
    if staff_info:
        contact_sections.extend(staff_info)
    
    return contact_sections

def extract_college_basic_info():
    """ดึงข้อมูลพื้นฐานของวิทยาลัย"""
    basic_info = []
    
    # ค้นหาข้อมูลที่อยู่และข้อมูลติดต่อหลัก
    # ตามรูปที่เห็น
    
    # หาข้อมูลที่อยู่และโทรศัพท์หลัก
    text_content = soup.get_text()
    
    # หาชื่อวิทยาลัย
    if "วิทยาลัยการคอมพิวเตอร์" in text_content:
        basic_info.append("หน่วยงาน: วิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น")
    
    # หาที่อยู่
    address_match = re.search(r'123 อาคารวิทยวิภาส.*?ขอนแก่น 40002', text_content)
    if address_match:
        basic_info.append(f"ที่อยู่: {address_match.group()}")
    
    # หาโทรศัพท์หลัก
    phone_match = re.search(r'043-009700 ต่อ 50528', text_content)
    if phone_match:
        basic_info.append(f"โทรศัพท์: {phone_match.group()}")
    
    # หา Hot Line
    hotline_match = re.search(r'089-7102651, 089-7102645', text_content)
    if hotline_match:
        basic_info.append(f"Hot Line: {hotline_match.group()}")
    
    # หาอีเมล
    email_match = re.search(r'computing\.kku@kku\.ac\.th', text_content)
    if email_match:
        basic_info.append(f"อีเมล: {email_match.group()}")
    
    return basic_info

def extract_staff_table_info():
    """ดึงข้อมูลบุคลากรจากตารางอย่างถูกต้อง"""
    staff_info = []
    
    # หาตารางที่มีข้อมูลบุคลากร
    tables = soup.find_all('table')
    
    for table in tables:
        # หาแถวหัวข้อ (ถ้ามี)
        headers = table.find_all('th')
        if not headers:
            continue
            
        header_texts = [th.get_text(strip=True) for th in headers]
        print(f"พบหัวตาราง: {header_texts}")
        
        # ตรวจสอบว่าเป็นตารางบุคลากรหรือไม่
        if any(keyword in ' '.join(header_texts) for keyword in ['การะงานตำแหน่ง', 'ติดต่อ', 'ตำแหน่ง', 'เบอร์โทร']):
            # ดึงข้อมูลจากแต่ละแถว
            rows = table.find_all('tr')[1:]  # ข้ามหัวตาราง
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 4:  # ต้องมีอย่างน้อย 4 คอลัมน์
                    
                    # ดึงข้อมูลจากแต่ละ cell
                    position = cells[0].get_text(strip=True) if len(cells) > 0 else ''
                    name = cells[1].get_text(strip=True) if len(cells) > 1 else ''
                    department = cells[2].get_text(strip=True) if len(cells) > 2 else ''
                    phone_ext = cells[3].get_text(strip=True) if len(cells) > 3 else ''
                    
                    # ตรวจสอบว่ามีข้อมูลที่มีความหมาย
                    if position and name and len(position) > 2:
                        # สร้างข้อมูลบุคลากรแต่ละคน
                        staff_record = f"บุคลากร: {position}"
                        if name:
                            staff_record += f" - ชื่อ: {name}"
                        if department:
                            staff_record += f" - หน่วยงาน: {department}"
                        if phone_ext:
                            staff_record += f" - ต่อ: {phone_ext}"
                        
                        staff_info.append(staff_record)
                        print(f"เพิ่มข้อมูลบุคลากร: {staff_record}")
    
    return staff_info

# ดึงข้อมูลติดต่อ
print("🔍 เริ่มดึงข้อมูลติดต่อจากเว็บไซต์...")
contact_info = extract_contact_info()
print(f"📝 ดึงข้อมูลได้ {len(contact_info)} รายการ")

# หากไม่พบข้อมูลจากโครงสร้างเฉพาะ ให้ดึงจาก div ทั้งหมด
if not contact_info:
    print("🔍 ไม่พบข้อมูลจากโครงสร้างเฉพาะ กำลังดึงข้อมูลทั่วไป...")
    
    # ดึงข้อมูลจาก div ทั้งหมดที่อาจมีข้อมูลติดต่อ
    all_divs = soup.find_all("div")
    for div in all_divs:
        div_text = div.get_text(strip=True)
        
        # ตรวจสอบว่ามีข้อมูลติดต่อหรือไม่
        if any(keyword in div_text for keyword in [
            "วิทยาลัยการคอมพิวเตอร์", "มหาวิทยาลัยขอนแก่น", "043-", "089-", 
            "@kku.ac.th", "โทรศัพท์", "Hot Line", "อีเมล", "ที่อยู่"
        ]):
            if len(div_text) > 10 and len(div_text) < 500:  # กรองข้อความที่เหมาะสม
                contact_info.append(div_text)

# สร้าง Documents
if contact_info:
    # รวมข้อมูลทั้งหมดเป็น document เดียว
    full_content = "\n".join(contact_info)
    metadata = {
        "source": url,
        "type": "contact_info",
        "category": "contact",  # Category สำหรับแยกประเภท
        "subcategory": "contact_overview"
    }
    docs.append(Document(page_content=full_content, metadata=metadata))
    
    # แยกข้อมูลเป็น documents ย่อย
    for i, info in enumerate(contact_info):
        if len(info.strip()) > 5:
            metadata = {
                "source": url,
                "type": "contact_detail",
                "category": "contact",  # Category สำหรับแยกประเภท
                "subcategory": "contact_detail",
                "section": i+1
            }
            docs.append(Document(page_content=info, metadata=metadata))
else:
    print("⚠️ ไม่พบข้อมูลติดต่อ")

print(f"📝 Scraped {len(docs)} contact documents")

# แสดงตัวอย่างข้อมูลที่ดึงได้
for i, doc in enumerate(docs[:3]):
    print(f"\n--- Document {i+1} ---")
    print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)

# -------------------------------
# Split documents into chunks
# -------------------------------
if docs:
    splitter = CharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separator="\n"
    )
    chunks = splitter.split_documents(docs)
    print(f"📄 Created {len(chunks)} chunks")

    # -------------------------------
    # Embeddings
    # -------------------------------
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # -------------------------------
    # Connect to AstraDB
    # -------------------------------
    client = DataAPIClient(token=ASTRA_TOKEN)
    db = client.get_database_by_api_endpoint(ASTRA_ENDPOINT)
    collection = db.get_collection(COLLECTION_NAME)
    print(f"✅ Connected to AstraDB collection: {COLLECTION_NAME}")

    # -------------------------------
    # Insert chunks in batch
    # -------------------------------
    batch_size = 50
    documents_to_insert = []

    for i, chunk in enumerate(chunks):
        vector = embedding_model.embed_query(chunk.page_content)
        doc = {
            "_id": str(uuid.uuid4()),
            "content": chunk.page_content,
            "$vector": vector,
            "metadata": chunk.metadata
        }
        documents_to_insert.append(doc)
        
        if len(documents_to_insert) >= batch_size or i == len(chunks) - 1:
            try:
                result = collection.insert_many(documents_to_insert)
                print(f"📊 Inserted {len(result.inserted_ids)} documents (chunk {i+1}/{len(chunks)})")
                documents_to_insert = []
            except Exception as e:
                print(f"❌ Failed to insert batch: {e}")

    print("🎉 Contact info AstraDB ingestion completed successfully!")
else:
    print("❌ No contact documents to process")
