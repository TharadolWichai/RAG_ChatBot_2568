import json
import os
import re
import uuid

import requests
import urllib3
from astrapy import DataAPIClient
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

def clean_html(html_content):
    """
    Clean HTML content → plain text.
    - ลบทุก tag HTML
    - ลบ style, &nbsp;, tab, newline ซ้ำ
    - รวม whitespace ให้เป็น space เดียว
    """
    soup = BeautifulSoup(html_content or "", "html.parser")
    text = soup.get_text(separator="\n")
    text = text.replace("\xa0", " ")  # &nbsp;
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    print("🚀 Starting AstraDB ingestion with astrapy (curriculum data)...")
    
    # Load environment variables
    token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
    keyspace = os.getenv("ASTRA_DB_KEYSPACE", "default_keyspace")

    if not token or not api_endpoint:
        print("❌ Error: Missing AstraDB credentials in .env file")
        return False

    print(f"🔑 Using endpoint: {api_endpoint}")
    print(f"🏠 Using keyspace: {keyspace}")

    # Connect to AstraDB
    try:
        client = DataAPIClient(token=token)
        database = client.get_database_by_api_endpoint(api_endpoint)
        print("✅ Connected to AstraDB successfully!")
    except Exception as e:
        print(f"❌ Failed to connect to AstraDB: {e}")
        return False

    # Prepare collection
    collection_name = "curriculum_embeddings"
    try:
        existing_collections = list(database.list_collection_names())
        print(f"📂 Existing collections: {existing_collections}")

        if collection_name in existing_collections:
            collection = database.get_collection(collection_name)
            print(f"📂 Using existing collection: {collection_name}")
        else:
            print(f"❌ Collection {collection_name} not found!")
            return False
    except Exception as e:
        print(f"❌ Failed to access collection: {e}")
        return False

    # Fetch data from API
    print("🌐 Fetching curriculum data from API...")
    try:
        url = "https://api.computing.kku.ac.th/api/v1/department/getDepartmentByCourseId/29?courseId=29&orderBy=asc&sortBy=displayOrder"
        res = requests.get(url, verify=False, timeout=20)

        # Debug: HTTP-level information
        print(f"🔁 HTTP {res.status_code} {res.reason}")
        print(f"🔎 Content-Type: {res.headers.get('content-type')}")
        print(f"📦 Response size: {len(res.content)} bytes")

        # Try parse JSON safely and print summary
        try:
            data = res.json()
            # Print top-level type and keys/lengths (truncate to avoid huge output)
            if isinstance(data, dict):
                print("🧭 JSON object keys:", list(data.keys()))
                # If 'data' key exists, show length/sample
                if "data" in data:
                    d = data["data"]
                    if isinstance(d, list):
                        print(f"📋 data → list length: {len(d)}")
                        if len(d) > 0:
                            sample = d[0]
                            print("🔎 Sample item keys:", list(sample.keys()) if isinstance(sample, dict) else type(sample))
                    else:
                        print(f"📋 data → type: {type(d)}")
                else:
                    # show if any value is list and its length
                    for k, v in data.items():
                        if isinstance(v, list):
                            print(f"📌 key '{k}' → list length: {len(v)}")
            elif isinstance(data, list):
                print(f"🧾 JSON array length: {len(data)}")
                if len(data) > 0 and isinstance(data[0], dict):
                    print("🔎 Sample item keys:", list(data[0].keys()))
            else:
                print("ℹ️ JSON parsed but top-level type is", type(data))
        except ValueError:
            # not JSON: print a truncated text preview
            text = res.text or ""
            preview = text[:2000] + ("...[truncated]" if len(text) > 2000 else "")
            print("⚠️ Response is not valid JSON. Text preview:")
            print(preview)
            data = None

        # --- Improved normalization: when data["data"] is a dict, search recursively for candidate lists
        def find_lists(obj):
            """Return list of lists-of-dicts found in obj (recursive)."""
            results = []
            if isinstance(obj, list):
                # accept list if it contains dicts (likely candidate)
                if len(obj) > 0 and any(isinstance(i, dict) for i in obj):
                    results.append(obj)
                # still recurse into list items
                for item in obj:
                    results.extend(find_lists(item))
            elif isinstance(obj, dict):
                for v in obj.values():
                    results.extend(find_lists(v))
            return results

        departments = []
        if isinstance(data, dict) and "data" in data:
            top = data["data"]
            if isinstance(top, list):
                departments = top
            elif isinstance(top, dict):
                # find candidate lists inside this dict
                lists = find_lists(top)
                if lists:
                    # choose the longest candidate list (heuristic)
                    lists_sorted = sorted(lists, key=lambda l: len(l), reverse=True)
                    departments = lists_sorted[0]
                    print(f"🔎 Found {len(lists)} candidate lists inside data; choosing list of length {len(departments)}")
                else:
                    # fallback: treat the dict as single department record
                    departments = [top]
            else:
                departments = []
        elif isinstance(data, list):
            departments = data
        else:
            # fallback: find any list anywhere in top-level object
            if isinstance(data, dict):
                lists = find_lists(data)
                departments = lists[0] if lists else []

        print(f"📊 Retrieved {len(departments)} curriculum records (after normalization)")
    except Exception as e:
        print(f"❌ Failed to fetch data from API: {e}")
        return False

    # Process data (extract only required fields: ชื่อ, คำอธิบาย, โครงสร้าง/แผนการเรียน)
    docs = []
    seen_keys = set()  # deduplicate by composite key (dept + title + curriculum snippet)

    for dep in departments:
        # some entries may be JSON strings
        if isinstance(dep, str):
            try:
                dep = json.loads(dep)
            except Exception:
                continue

        # ชื่อสาขา/หลักสูตร (จากหลายคีย์ที่เป็นไปได้)
        dep_name = (
            dep.get("departmentName")
            or dep.get("displayName")
            or dep.get("name")
            or (dep.get("departmentLocalized") or {}).get("name", "")
        )
        dep_name = (dep_name or "").strip()

        # พยายามดึงคำอธิบายและแผนการเรียน/โครงสร้างจากโครงสร้างที่เป็นไปได้
        # หลาย API เก็บข้อมูลในหลายคีย์ เช่น description, content, detail, curriculum, structure, studyPlan, syllabus
        # เราจะค้นหาในหลายตำแหน่ง แต่ไม่สนใจ languageId
        title_candidates = []
        description_candidates = []
        curriculum_candidates = []

        # top-level possible fields
        if dep.get("title"):
            title_candidates.append(dep.get("title"))
        if dep.get("description"):
            description_candidates.append(dep.get("description"))
        if dep.get("content"):
            description_candidates.append(dep.get("content"))
        for k in ("curriculum", "structure", "studyPlan", "plan", "syllabus"):
            if dep.get(k):
                curriculum_candidates.append(dep.get(k))

        # nested mappings (common structure in this API)
        mappings = dep.get("departmentDetail_Mapping") or dep.get("departmentDetailMapping") or dep.get("details") or []
        if isinstance(mappings, list):
            for m in mappings:
                detail = m.get("departmentDetail") or m.get("detail") or m
                # localized list or direct fields
                localized = detail.get("departmentDetailLocalized") or detail.get("localized") or [detail]
                for loc in localized:
                    if not isinstance(loc, dict):
                        continue
                    if loc.get("title"):
                        title_candidates.append(loc.get("title"))
                    if loc.get("name"):
                        title_candidates.append(loc.get("name"))
                    if loc.get("description"):
                        description_candidates.append(loc.get("description"))
                    if loc.get("content"):
                        description_candidates.append(loc.get("content"))
                    for k in ("curriculum", "structure", "studyPlan", "plan", "syllabus"):
                        if loc.get(k):
                            curriculum_candidates.append(loc.get(k))

        # fallback: look for any dict values that look like text
        for k, v in dep.items():
            if isinstance(v, str) and k.lower() in ("description", "detail", "content", "summary"):
                description_candidates.append(v)

        # clean and pick first non-empty values
        def pick_text(cands):
            for c in cands:
                if not c:
                    continue
                text = clean_html(c) if (isinstance(c, str) and ("<" in c or "\n" in c)) else str(c).strip()
                if text:
                    return text
            return ""

        title = pick_text(title_candidates)
        description_text = pick_text(description_candidates)
        curriculum_text = pick_text(curriculum_candidates)

        # if nothing useful found, skip
        if not (dep_name or title or description_text or curriculum_text):
            continue

        # build content with only the requested fields (ตามที่ต้องการ)
        content_parts = []
        if dep_name:
            content_parts.append(f"ชื่อสาขา/หลักสูตร: {dep_name}")
        if title and title.lower() not in (dep_name or "").lower():
            content_parts.append(f"ชื่อหัวข้อ: {title}")
        if description_text:
            content_parts.append(f"คำอธิบาย: {description_text}")
        if curriculum_text:
            content_parts.append(f"โครงสร้าง/แผนการเรียน: {curriculum_text}")

        content = "\n\n".join(content_parts)

        # dedupe by name+title+first 100 chars of curriculum to avoid near-duplicates
        dedupe_key = f"{(dep_name or '')}||{(title or '')}||{(curriculum_text or '')[:100]}"
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)

        metadata = {
            "department_name": dep_name,
            "title": title,
            "source_api": url,
            "type": "curriculum"
        }

        docs.append(Document(page_content=content, metadata=metadata))
        print(f"✅ Processed (minimal): {dep_name} - {title}")

    if len(docs) == 0:
        print("🚨 No documents found for embedding")
        return False

    print(f"📝 Processed {len(docs)} documents")

    # Split documents
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"📄 Created {len(chunks)} chunks")

    # Initialize embedding model
    print("🧠 Initializing embeddings model...")
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Generate embeddings and insert
    print("💾 Inserting data into AstraDB...")
    documents_to_insert = []
    for i, chunk in enumerate(chunks):
        vector = embedding.embed_query(chunk.page_content)
        doc = {
            "_id": str(uuid.uuid4()),
            "content": chunk.page_content,
            "$vector": vector,
            "metadata": chunk.metadata
        }
        documents_to_insert.append(doc)

        if i % 10 == 0:
            print(f"📊 Processed {i+1}/{len(chunks)} chunks...")

    # Insert all into AstraDB
    try:
        result = collection.insert_many(documents_to_insert)
        print(f"✅ Successfully inserted {len(result.inserted_ids)} documents into AstraDB!")
    except Exception as e:
        print(f"❌ Failed to insert documents: {e}")
        return False

    # Verify count
    try:
        count = collection.count_documents({})
        print(f"🔍 Total documents in collection: {count}")
    except Exception as e:
        print(f"⚠️ Could not verify document count: {e}")

    print("🎉 Curriculum ingestion completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
