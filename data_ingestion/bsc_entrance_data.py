import os
import re
import time
import hashlib

from astrapy import DataAPIClient
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

try:
    from selenium.webdriver.chrome.service import Service as ChromeService
    from webdriver_manager.chrome import ChromeDriverManager
    service = ChromeService(ChromeDriverManager().install())
    print("✅ ใช้ webdriver-manager สำหรับ ChromeDriver")
except ImportError:
    driver_path = os.path.join(os.path.dirname(__file__), "..", "drivers", "chromedriver.exe")
    if os.path.exists(driver_path):
        service = Service(driver_path)
        print("⚠️  ใช้ ChromeDriver จากโฟลเดอร์ drivers (อาจเวอร์ชันไม่ตรง)")
    else:
        raise FileNotFoundError(
            f"ChromeDriver not found at {driver_path}. "
            "Please install webdriver-manager: pip install webdriver-manager"
        )

URL = "https://computing.kku.ac.th/bsc-entrance"
DOC_TYPE = "bsc_entrance"  # ✅ ใช้ type แบบเดียวกับไฟล์อื่น

# -------------------------------
# Helpers
# -------------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\t", " ")
    text = re.sub(r"[ \u00A0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def stable_id(
    source: str,
    doc_type: str,
    content_kind: str,
    div_index: int,
    table_row: int,
    chunk_id: int,
    content: str
) -> str:
    h = hashlib.sha256()
    key = f"{doc_type}|{content_kind}|{source}|div={div_index}|row={table_row}|chunk={chunk_id}|"
    h.update(key.encode("utf-8"))
    h.update(content.encode("utf-8"))
    return h.hexdigest()

# -------------------------------
# Scrape page
# -------------------------------
driver = webdriver.Chrome(service=service, options=chrome_options)
try:
    driver.get(URL)
    time.sleep(5)
    soup = BeautifulSoup(driver.page_source, "html.parser")
finally:
    driver.quit()

docs = []
docs_table_rows = [] #สร้าง list สำหรับเก็บ row table
content_divs = soup.find_all("div", class_="kku-content")

for div_index, div in enumerate(content_divs, start=1):
    parts = []
    current_round = None 

    for child in div.find_all(recursive=False):
        if child.name in ["h3", "h4"]:
            heading = child.get_text(strip=True)
            parts.append(heading)

            # ✅ จับ "รอบที่ X"
            m = re.search(r"รอบ(?:ที่)?\s*(\d+)", heading)
            if m:
                current_round = int(m.group(1))

        elif child.name == "p":
            text_parts = []
            for elem in child.children:
                if getattr(elem, "name", None) == "a":
                    href = elem.get("href")
                    link_text = elem.get_text(strip=True)
                    text_parts.append(f"{link_text} ({href})" if href else link_text)
                else:
                    text = elem.get_text(strip=True) if hasattr(elem, "get_text") else str(elem).strip()
                    if text:
                        text_parts.append(text)
            parts.append(" ".join(text_parts))

        elif child.name == "ol":
            for li in child.find_all("li"):
                parts.append(f"- {li.get_text(' ', strip=True)}")
                for yl_div in li.find_all("div", class_="year-links"):
                    for a in yl_div.find_all("a"):
                        href = a.get("href")
                        link_text = a.get_text(strip=True)
                        parts.append(f"  > Link: {link_text} ({href})" if href else f"  > Link: {link_text}")

        elif child.name == "table" and "kku-table" in (child.get("class") or []):
            def is_subject_code(code: str) -> bool:
                """รับเฉพาะรหัสวิชาที่เป็นคอลัมน์คะแนนจริง"""
                if not code:
                    return False
                code = code.strip()
                # รอบ 2: 101-204, รอบ 3: 61/64/65/66/82, และ TGAT2/TGAT3
                return bool(re.fullmatch(r"\d{2,3}", code)) or code in {"TGAT2", "TGAT3"}

            # -------- 1) อ่านหัวตารางให้ถูก --------
            header_map = []  # list of (code, subject_name)
            thead = child.find("thead")
            if thead:
                ths = thead.find_all("th")
                for th in ths:
                    strong = th.find("strong")
                    code = strong.get_text(strip=True) if strong else ""
                    if not is_subject_code(code):
                        continue

                    span = th.find("span")
                    subj = span.get_text(strip=True) if span else ""
                    # ถ้าไม่มีชื่อวิชา ให้ใช้ code เป็นชื่อกันสับสน
                    subj = subj if subj else code

                    header_map.append((code, subj))

            # -------- 2) อ่าน tbody แล้ว map คะแนนแบบไม่เลื่อน --------
            tbody = child.find("tbody")
            if tbody and header_map:
                for row_i, tr in enumerate(tbody.find_all("tr"), start=1):
                    tds = tr.find_all("td")
                    if len(tds) < 2:
                        continue

                    # ชื่อหลักสูตรอยู่ td แรก
                    first_td = tds[0]
                    prog_strong = first_td.find("strong")
                    prog_span = first_td.find("span")

                    program = (prog_strong.get_text(" ", strip=True) if prog_strong else first_td.get_text(" ", strip=True)).strip()
                    note = (prog_span.get_text(" ", strip=True) if prog_span else "").strip()

                    # ค่าคะแนนอยู่ td ที่เหลือ
                    vals = []
                    for td in tds[1:]:
                        v = td.get_text(strip=True).replace("%", "").strip()
                        vals.append(v)

                    # ✅ ตัด "รวม" ออก (มักเป็น 100 หรือ 100%)
                    total_val = None
                    if vals:
                        last = vals[-1]
                        if last.isdigit() and int(last) == 100:
                            total_val = last
                            vals = vals[:-1]  # เอาออกก่อน map

                    # ✅ map แค่จำนวนเท่ากับหัวตาราง (กันเลื่อน)
                    n = min(len(header_map), len(vals))
                    pairs = [f"{header_map[i][0]} {header_map[i][1]} {vals[i]}%" for i in range(n)]

                    # ทำ round_text ให้ retrieval แม่น
                    round_text = f"รอบ {current_round}" if current_round else "ไม่ระบุรอบ"

                    line = (
                        f"{round_text} | ตารางค่าน้ำหนักคะแนนรายวิชา\n"
                        f"หลักสูตร: {program} {note}\n"
                        + " | ".join(pairs)
                        + (f" | รวม {total_val}%" if total_val else "")
                        + f"\nที่มา: {URL}"
                    )

                    md_row = {
                        "source": URL,
                        "type": DOC_TYPE,
                        "page": DOC_TYPE,
                        "content_kind": "weight_table",
                        "round": current_round,
                        "div_index": div_index,
                        "table_row": row_i,
                    }

                    docs_table_rows.append(Document(page_content=line, metadata=md_row))

            # (optional) debug ชั่วคราว
            # print("HEADER_MAP:", header_map)

        elif child.name == "div" and "kku-note" in (child.get("class") or []):
            strong_tags = child.find_all("strong")
            links = child.find_all("a")

            for i, a in enumerate(links):
                description = ""
                if i < len(strong_tags):
                    strong = strong_tags[i]
                    u_tag = strong.find("u")
                    description = (u_tag.get_text(strip=True) if u_tag else strong.get_text(strip=True)).strip()
                    if description in [a.get_text(strip=True).strip(), (a.get("href") or "").strip()]:
                        description = ""
                if description:
                    parts.append(f"Description: {description}")

                link_text = a.get_text(strip=True)
                href = a.get("href")
                parts.append(f"- {link_text} ({href})" if href else f"- {link_text}")

    content = clean_text("\n".join(parts))
    md = {"div_index": div_index, "source": URL, "type": DOC_TYPE, "page": DOC_TYPE}
    docs.append(Document(page_content=content, metadata=md))

docs.extend(docs_table_rows)
print(f"📊 table row docs: {len(docs_table_rows)}")
print(f"📝 Scraped {len(docs)} documents")

# -------------------------------
# Chunking
# -------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " ", ""],
)

chunked_docs = []
for d in docs:
    chunks = splitter.split_text(d.page_content)
    for i, ch in enumerate(chunks):
        md = dict(d.metadata)
        md["chunk_id"] = i
        chunked_docs.append(Document(page_content=ch, metadata=md))

print(f"✂️ Chunked into {len(chunked_docs)} chunks")

# -------------------------------
# Embeddings
# -------------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large"
)
# ถ้าอยากแม่นไทย แนะนำเปลี่ยนเป็น multilingual ภายหลัง

# -------------------------------
# Connect AstraDB
# -------------------------------
client = DataAPIClient(token=ASTRA_TOKEN)
db = client.get_database_by_api_endpoint(ASTRA_ENDPOINT)
collection = db.get_collection(COLLECTION_NAME)
print(f"✅ Connected to AstraDB collection: {COLLECTION_NAME}")

# -------------------------------
# Upsert documents (incremental by stable id)
# -------------------------------
inserted = 0
updated = 0

for d in chunked_docs:
    content = d.page_content
    md = d.metadata

    _id = stable_id(
        source=md.get("source", URL),
        doc_type=md.get("type", DOC_TYPE),
        content_kind=md.get("content_kind", "general"),
        div_index=md.get("div_index", 0),
        table_row=md.get("table_row", 0),
        chunk_id=md.get("chunk_id", 0),
        content=content
    )
    vec = embedding_model.embed_query("passage: " + content)
    
    record = {
        "_id": _id,
        "content": content,
        "metadata": md,   # ✅ type จะไม่หายแน่นอน
        "$vector": vec,
    }

    try:
        # ลอง insert ก่อน
        collection.insert_one(record)
        inserted += 1
    except Exception:
        # ถ้าชน id ให้ replace/update
        try:
            collection.replace_one({"_id": _id}, record, upsert=True)
            updated += 1
        except Exception as e2:
            print(f"❌ Upsert failed for id={_id[:8]}... : {e2}")

print(f"\n✅ Done! inserted={inserted}, updated={updated}")