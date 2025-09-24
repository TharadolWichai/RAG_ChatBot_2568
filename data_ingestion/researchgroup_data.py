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
ASTRA_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE", "default_keyspace")
COLLECTION_NAME = "researchgroup_embedding"

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
# Scrape multiple URLs
# -------------------------------
docs = []

for url_index, url in enumerate(urls, start=1):
    print(f"\n🌐 [URL {url_index}] Fetching: {url}")
    driver.get(url)
    time.sleep(3)  # wait for JS

    soup = BeautifulSoup(driver.page_source, "html.parser")

    # -------------------------
    # หา div class="w-100 h-100"
    # -------------------------
    content_div = soup.find("div", class_="w-100 h-100")

    if not content_div:
        print("❌ ไม่เจอ div.w-100.h-100")
        continue

    parts = []
    seen_texts = set()

    for child in content_div.find_all(recursive=False):
        # h2, h3
        if child.name in ["h2", "h3"]:
            text = child.get_text(strip=True)
            if text and text not in seen_texts:
                parts.append(text)
                seen_texts.add(text)

        # p
        elif child.name == "p":
            texts = []
            for elem in child.children:
                if getattr(elem, "name", None) == "a":
                    href = elem.get("href")
                    link_text = elem.get_text(strip=True)
                    texts.append(f"{link_text} ({href})")
                else:
                    text = elem.get_text(strip=True) if hasattr(elem, "get_text") else str(elem).strip()
                    if text:
                        texts.append(text)
            text_joined = " ".join(texts)
            if text_joined and text_joined not in seen_texts:
                parts.append(text_joined)
                seen_texts.add(text_joined)

        # ul/ol + li
        elif child.name in ["ul", "ol"]:
            for li in child.find_all("li"):
                li_text = li.get_text(" ", strip=True)
                if li_text and li_text not in seen_texts:
                    parts.append(f"- {li_text}")
                    seen_texts.add(li_text)

        # table
        elif child.name == "table":
            rows = []
            for row in child.find_all("tr"):
                cells = row.find_all(["th", "td"])
                cell_texts = [cell.get_text(" ", strip=True) for cell in cells]
                rows.append("\t".join(cell_texts))
            table_text = "Table:\n" + "\n".join(rows)
            if table_text not in seen_texts:
                parts.append(table_text)
                seen_texts.add(table_text)

        # div.note
        elif child.name == "div" and "note" in child.get("class", []):
            note_text = child.get_text(" ", strip=True)
            if note_text and note_text not in seen_texts:
                parts.append(f"Note: {note_text}")
                seen_texts.add(note_text)

    if parts:
        content = "\n".join(parts)
        metadata = {"source": url}
        docs.append(Document(page_content=content, metadata=metadata))
        print(f"📝 Scraped {len(parts)} items from {url}")

driver.quit()

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
