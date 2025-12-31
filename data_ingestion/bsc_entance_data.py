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
from incremental_utils import enable_incremental_mode

load_dotenv()

# -------------------------------
# AstraDB config
# -------------------------------
ASTRA_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE", "default_keyspace")
COLLECTION_NAME = "bsc_entrance_embedding"

if not ASTRA_TOKEN or not ASTRA_ENDPOINT:
    raise ValueError("Missing AstraDB credentials in .env")

# -------------------------------
# Selenium setup
# -------------------------------
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--remote-debugging-port=9222")

# ลองใช้ webdriver-manager เพื่อดาวน์โหลด ChromeDriver อัตโนมัติ
try:
    from selenium.webdriver.chrome.service import Service as ChromeService
    from webdriver_manager.chrome import ChromeDriverManager
    service = ChromeService(ChromeDriverManager().install())
    print("✅ ใช้ webdriver-manager สำหรับ ChromeDriver")
except ImportError:
    # Fallback: ใช้ chromedriver จากโฟลเดอร์ drivers
    driver_path = os.path.join(os.path.dirname(__file__), "..", "drivers", "chromedriver.exe")
    if os.path.exists(driver_path):
        service = Service(driver_path)
        print(f"⚠️  ใช้ ChromeDriver จากโฟลเดอร์ drivers (อาจเวอร์ชันไม่ตรง)")
    else:
        raise FileNotFoundError(
            f"ChromeDriver not found at {driver_path}. "
            "Please install webdriver-manager: pip install webdriver-manager"
        )

driver = webdriver.Chrome(service=service, options=chrome_options)

url = "https://computing.kku.ac.th/bsc-entrance"
driver.get(url)
time.sleep(5)  # รอโหลด JS
soup = BeautifulSoup(driver.page_source, "html.parser")
driver.quit()

# -------------------------------
# Scrape kku-content
# -------------------------------
docs = []

content_divs = soup.find_all("div", class_="kku-content")
for div_index, div in enumerate(content_divs, start=1):
    parts = []
    
    for child in div.find_all(recursive=False):
        # h3, h4
        if child.name in ["h3", "h4"]:
            parts.append(child.get_text(strip=True))
        
        # p มี <a> อยู่ข้างใน
        elif child.name == "p":
            text_parts = []
            for elem in child.children:
                if getattr(elem, "name", None) == "a":
                    href = elem.get("href")
                    link_text = elem.get_text(strip=True)
                    text_parts.append(f"{link_text} ({href})")
                else:
                    text = elem.get_text(strip=True) if hasattr(elem, "get_text") else str(elem).strip()
                    if text:
                        text_parts.append(text)
            parts.append(" ".join(text_parts))
        
        # ol + li + year-links
        elif child.name == "ol":
            for li in child.find_all("li"):
                li_text = li.get_text(" ", strip=True)
                parts.append(f"- {li_text}")
                
                year_links_divs = li.find_all("div", class_="year-links")
                for yl_div in year_links_divs:
                    for a in yl_div.find_all("a"):
                        href = a.get("href")
                        link_text = a.get_text(strip=True)
                        parts.append(f"  > Link: {link_text} ({href})")
        
        # table
        elif child.name == "table" and "kku-table" in child.get("class", []):
            table_rows = []
            for row in child.find_all("tr"):
                cells = row.find_all(["th", "td"])
                cell_texts = [cell.get_text(" ", strip=True) for cell in cells]
                table_rows.append("\t".join(cell_texts))
            parts.append("Table:\n" + "\n".join(table_rows))
        
        # kku-note
        elif child.name == "div" and "kku-note" in child.get("class", []):
            strong_tags = child.find_all("strong")
            links = child.find_all("a")
            
            for i, a in enumerate(links):
                description = ""
                if i < len(strong_tags):
                    strong = strong_tags[i]
                    u_tag = strong.find("u")
                    if u_tag:
                        description = u_tag.get_text(strip=True)
                    else:
                        description = strong.get_text(strip=True)
                        if description.strip() in [a.get_text(strip=True).strip(), a.get("href").strip()]:
                            description = ""
                if description:
                    parts.append(f"Description: {description}")
                link_text = a.get_text(strip=True)
                href = a.get("href")
                parts.append(f"- {link_text} ({href})")
    
    content = "\n".join(parts)
    metadata = {"div_index": div_index, "source": url}
    docs.append(Document(page_content=content, metadata=metadata))

print(f"📝 Scraped {len(docs)} documents")

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
# Use incremental indexing
# -------------------------------
print("\n🚀 Starting incremental indexing...")
try:
    stats = enable_incremental_mode(
        collection=collection,
        embedding_model=embedding_model,
        new_documents=docs,
        metadata_filter={"type": "bsc_entrance"},  # Filter สำหรับดึงเอกสารคณะ
        hash_keys=["source", "type"],  # Keys สำหรับสร้าง unique hash
        delete_missing=False  # ไม่ลบเอกสารเก่า
    )
    
    print(f"\n✅ Incremental indexing completed!")
    print(f"   - New documents inserted: {stats['inserted']}")
    print(f"   - Existing documents skipped: {stats['skipped']}")
    print("🎉 AstraDB ingestion completed successfully!")
    
except Exception as e:
    print(f"❌ Failed to process incremental indexing: {e}")
