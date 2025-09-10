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
driver_path = "D:/CS YEAR 4/chatbot_kkucp2568/RAG_ChatBot_2568/chromedriver-win64/chromedriver.exe"
service = Service(driver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

url = "https://computing.kku.ac.th/bsc-entrance"
driver.get(url)
time.sleep(3)  # รอโหลด JS
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
# Split documents into chunks
# -------------------------------
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

print("🎉 AstraDB ingestion completed successfully!")
