import os
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
COLLECTION_NAME = "researchgroup_embeddings"

if not ASTRA_TOKEN or not ASTRA_ENDPOINT:
    raise ValueError("Missing AstraDB credentials in .env")

# -------------------------------
# Selenium setup
# -------------------------------
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

driver_path = "D:/CS YEAR 4/chatbot_kkucp2568/RAG_ChatBot_2568/drivers/chromedriver.exe"
service = Service(driver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

# -------------------------------
# List URLs
# -------------------------------
urls = [
    "https://computing.kku.ac.th/hardware-human",
    "https://computing.kku.ac.th/mlislab",
    "https://computing.kku.ac.th/aiii",
    "https://computing.kku.ac.th/agtlab",
    "https://computing.kku.ac.th/asclab",
    "https://computing.kku.ac.th/nlsplab",
    "https://computing.kku.ac.th/aidalab",
    "https://computing.kku.ac.th/i-serg"
]

# -------------------------------
# Extract text structured
# -------------------------------
def extract_group_content(content_div):
    seen_texts = set()
    parts = []

    for tag in content_div.find_all(["h1","h2","h3","p","ul","ol","li"], recursive=True):
        texts = []
        if tag.name == "h2":
            strong_tags = tag.find_all("strong")
            texts = [st.get_text(" ", strip=True) for st in strong_tags] if strong_tags else [tag.get_text(" ", strip=True)]
        elif tag.name in ["ul", "ol"]:
            texts = [li.get_text(" ", strip=True) for li in tag.find_all("li")]
        else:
            texts = [tag.get_text(" ", strip=True)]

        for t in texts:
            clean_t = " ".join(t.split())
            if clean_t and clean_t not in seen_texts:
                parts.append(clean_t)
                seen_texts.add(clean_t)

    return parts

# -------------------------------
# Scrape all URLs
# -------------------------------
docs = []

for url_index, url in enumerate(urls, start=1):
    print(f"\n🌐 [URL {url_index}] Fetching: {url}")
    driver.get(url)
    time.sleep(4)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    content_div = (
        soup.find("div", class_="w-100 h-100") or
        soup.find("main") or
        soup.find("div", class_="elementor-section") or
        soup.find("div", class_="container") or
        soup.body
    )

    if not content_div:
        print("❌ ไม่เจอแท็กหลักในหน้า:", url)
        continue

    parts = extract_group_content(content_div)
    if parts:
        content = "\n".join(parts)
        metadata = {"source": url}
        docs.append(Document(page_content=content, metadata=metadata))
        print(f"📝 Scraped {len(parts)} items from {url}")
    else:
        print(f"⚠️ No content found in {url}")

driver.quit()
print(f"\n✅ รวมทั้งหมด {len(docs)} documents ที่ scrape ได้")

# -------------------------------
# Split documents into chunks
# -------------------------------
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=150, separator="\n")
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

if COLLECTION_NAME not in [c.name for c in db.list_collections()]:
    print(f"📦 Creating collection '{COLLECTION_NAME}' (dimension=384)...")
    db.create_collection(COLLECTION_NAME, dimension=384)

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
            collection.insert_many(documents_to_insert)
            print(f"📊 Inserted {len(documents_to_insert)} documents (chunk {i+1}/{len(chunks)})")
            documents_to_insert = []
        except Exception as e:
            print(f"❌ Failed to insert batch: {repr(e)}")

print("🎉 AstraDB ingestion completed successfully!")
