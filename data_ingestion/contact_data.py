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
COLLECTION_NAME = "contactus_embedding"

if not ASTRA_TOKEN or not ASTRA_ENDPOINT:
    raise ValueError("Missing AstraDB credentials in .env")

# -------------------------------
# Selenium setup
# -------------------------------
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
driver_path = "D:/CS YEAR 4/chatbot_kkucp2568/RAG_ChatBot_2568/chromedriver-win64/chromedriver.exe"
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
    """ดึงข้อมูลติดต่อจากโครงสร้าง HTML ตามที่เห็นในรูป"""
    contact_sections = []
    
    # ค้นหา div ที่มี class col-12 หรือ container ที่มีข้อมูลติดต่อ
    main_containers = soup.find_all("div", class_=["col-12", "container"])
    
    for container in main_containers:
        # ค้นหาข้อมูลในแต่ละ section
        
        # 1. ดึงข้อมูลหัวข้อหลัก (วิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น)
        title_elements = container.find_all(["h1", "h2", "h3", "strong"])
        for title in title_elements:
            title_text = title.get_text(strip=True)
            if "วิทยาลัยการคอมพิวเตอร์" in title_text or "มหาวิทยาลัยขอนแก่น" in title_text:
                contact_sections.append(f"หน่วยงาน: {title_text}")
        
        # 2. ดึงข้อมูลที่อยู่
        address_patterns = [
            r'\d+\s+อาคาร.*',
            r'.*อำเภอ.*จังหวัด.*',
            r'.*\d{5}.*'  # รหัสไปรษณีย์
        ]
        
        text_content = container.get_text()
        for pattern in address_patterns:
            matches = re.findall(pattern, text_content)
            for match in matches:
                if len(match.strip()) > 10:  # กรองข้อความที่สั้นเกินไป
                    contact_sections.append(f"ที่อยู่: {match.strip()}")
        
        # 3. ดึงข้อมูลโทรศัพท์
        phone_elements = container.find_all(text=re.compile(r'โทรศัพท์|Tel|Phone|043-\d+'))
        for phone in phone_elements:
            phone_text = str(phone).strip()
            if "043-" in phone_text or "โทรศัพท์" in phone_text:
                contact_sections.append(f"โทรศัพท์: {phone_text}")
        
        # 4. ดึงข้อมูล Hot Line
        hotline_elements = container.find_all(text=re.compile(r'Hot Line|089-\d+'))
        for hotline in hotline_elements:
            hotline_text = str(hotline).strip()
            if "089-" in hotline_text or "Hot Line" in hotline_text:
                contact_sections.append(f"Hot Line: {hotline_text}")
        
        # 5. ดึงข้อมูลอีเมล
        email_elements = container.find_all(text=re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'))
        for email in email_elements:
            email_text = str(email).strip()
            if "@" in email_text:
                contact_sections.append(f"อีเมล: {email_text}")
        
        # 6. ดึงข้อมูลจาก span elements ที่มี style font-size:16px
        span_elements = container.find_all("span", style=re.compile(r'font-size:\s*16px'))
        for span in span_elements:
            span_text = span.get_text(strip=True)
            if len(span_text) > 5:  # กรองข้อความที่สั้นเกินไป
                contact_sections.append(f"ข้อมูล: {span_text}")
        
        # 7. ดึงข้อมูลจาก p elements
        p_elements = container.find_all("p")
        for p in p_elements:
            p_text = p.get_text(strip=True)
            # ตรวจสอบว่าเป็นข้อมูลติดต่อหรือไม่
            if any(keyword in p_text for keyword in ["คณบดี", "รองคณบดี", "ผู้ช่วย", "งาน", "ฝ่าย", "แผนก"]):
                contact_sections.append(f"บุคลากร: {p_text}")
    
    return contact_sections

# ดึงข้อมูลติดต่อ
contact_info = extract_contact_info()

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
    metadata = {"source": url, "type": "contact_info"}
    docs.append(Document(page_content=full_content, metadata=metadata))
    
    # แยกข้อมูลเป็น documents ย่อย
    for i, info in enumerate(contact_info):
        if len(info.strip()) > 5:
            metadata = {"source": url, "type": "contact_detail", "section": i+1}
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
