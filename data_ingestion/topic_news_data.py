import requests
import json
from dotenv import load_dotenv
import os
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from astrapy import DataAPIClient
import uuid
import time
from datetime import datetime

load_dotenv()

def fetch_topic_news_list(page=1, size=10):
    """ดึงรายการหัวข้อข่าวจาก API"""
    try:
        url = f"https://api.computing.kku.ac.th/api/v1/article/list?search=&page={page}&size={size}&orderBy=desc&sortBy=publishedAt&prefix=news"
        print(f"🔍 Fetching topic news from page {page}: {url}")
        
        res = requests.get(url, timeout=15)
        if res.status_code == 200:
            data = res.json()
            if data.get("status") == "success":
                return data.get("data", {})
            else:
                print(f"❌ API returned error for page {page}: {data}")
                return None
        else:
            print(f"❌ HTTP error {res.status_code} for page {page}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to fetch topic news page {page}: {e}")
        return None

def fetch_article_detail_by_slug(slug):
    """ดึงรายละเอียดข่าวแต่ละเรื่องตาม slug"""
    try:
        url = f"https://api.computing.kku.ac.th/api/v1/article/getArticleBySlug/{slug}"
        print(f"📄 Fetching article detail: {slug}")
        res = requests.get(url, timeout=10)
        
        if res.status_code == 200:
            data = res.json()
            if data.get("status") == "success":
                return data.get("data")
            else:
                print(f"❌ API returned error for {slug}: {data}")
                return None
        else:
            print(f"❌ HTTP error {res.status_code} for {slug}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to fetch article detail {slug}: {e}")
        return None

def clean_html_text(html_text):
    """ลบ HTML tags และจัดรูปแบบข้อความ"""
    import re
    if not html_text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', html_text)
    
    # Replace HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def process_topic_news_item(item, detailed_data=None):
    """ประมวลผลข้อมูลหัวข้อข่าวสำหรับ embedding"""
    if not item:
        return None
    
    try:
        # ข้อมูลอยู่ใน item.article
        article = item.get("article", {})
        if not article:
            print("⚠️ No article data found in item")
            return None
            
        # ข้อมูลพื้นฐานจาก list API
        article_id = article.get("id", "")
        slug = article.get("slug", "")
        
        # ข้อมูลภาษาไทยจาก list
        localized = article.get("articleLocalized", {})
        title = clean_html_text(localized.get("name", ""))
        short_desc = clean_html_text(localized.get("shortDescription", ""))
        
        # วันที่
        created_at = article.get("createdAt", "")
        published_at = article.get("publishedAt", "")
        event_at = article.get("eventAt", "")
        
        # ข้อมูลผู้เขียน (อาจไม่มีใน list API)
        created_user = article.get("createdUser", {})
        author_name = f"{created_user.get('firstname', '')} {created_user.get('lastname', '')}".strip()
        author_position = created_user.get("academicPosition", "").strip()
        
        # หมวดหมู่ (อาจไม่มีใน list API)
        categories = article.get("articleCategory_Mapping", [])
        category_names = []
        for cat in categories:
            cat_localized = cat.get("categoryLocalized", {})
            if cat_localized.get("name"):
                category_names.append(cat_localized.get("name"))
        
        # รายละเอียดเพิ่มเติมจาก detail API (ถ้ามี)
        full_description = ""
        if detailed_data:
            detail_localized = detailed_data.get("articleLocalized", {})
            full_description = clean_html_text(detail_localized.get("description", ""))
        
        # สร้างเนื้อหาสำหรับ embedding
        content_parts = [
            f"ชื่อข่าว: {title}",
            f"คำอธิบายสั้น: {short_desc}" if short_desc else "",
        ]
        
        # เพิ่มรายละเอียดเต็มถ้ามี (แต่จำกัดความยาว)
        if full_description:
            max_desc_length = 1500  # จำกัดเนื้อหาหลัก
            if len(full_description) > max_desc_length:
                full_description = full_description[:max_desc_length] + "..."
            content_parts.append(f"เนื้อหา: {full_description}")
        
        # เพิ่มข้อมูลอื่นๆ
        if author_name:
            content_parts.append(f"ผู้เขียน: {author_name}")
        if author_position:
            content_parts.append(f"ตำแหน่งผู้เขียน: {author_position}")
        if category_names:
            content_parts.append(f"หมวดหมู่: {', '.join(category_names)}")
        if published_at:
            content_parts.append(f"วันที่เผยแพร่: {published_at}")
        
        content_parts.append(f"Slug: {slug}")
        
        # กรองเนื้อหาที่ไม่ว่าง
        content_parts = [part for part in content_parts if part.strip()]
        content = "\n".join(content_parts)
        
        # Final check - ตัดให้ไม่เกิน 2500 characters
        if len(content) > 2500:
            content = content[:2500] + "..."
            print(f"⚠️ ตัดเนื้อหาให้สั้นลง: {len(content)} characters")
        
        # สร้าง metadata
        metadata = {
            "article_id": str(article_id),
            "slug": slug,
            "title": title,
            "short_description": short_desc,
            "author_name": author_name,
            "author_position": author_position,
            "categories": category_names,
            "created_at": created_at,
            "published_at": published_at,
            "event_at": event_at,
            "type": "topic_news",
            "has_full_content": bool(detailed_data)
        }
        
        return Document(page_content=content, metadata=metadata)
        
    except Exception as e:
        print(f"❌ Error processing topic news item: {e}")
        return None

def main():
    print("🚀 Starting Topic News AstraDB ingestion with astrapy...")
    
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
    
    # Get existing collection for topic news
    collection_name = "topicnews_embedding"
    try:
        # List existing collections first
        existing_collections = list(database.list_collection_names())
        print(f"📂 Existing collections: {existing_collections}")
        
        if collection_name in existing_collections:
            collection = database.get_collection(collection_name)
            print(f"📂 Using existing collection: {collection_name}")
        else:
            print(f"❌ Collection {collection_name} not found!")
            print("Please create the collection via AstraDB UI with vector support:")
            print(f"  - Collection Name: {collection_name}")
            print("  - Vector Dimension: 384")
            print("  - Vector Metric: cosine")
            return False
            
    except Exception as e:
        print(f"❌ Failed to access collection: {e}")
        return False
    
    # ดึงข้อมูลจากหน้า 1 และหน้า 2
    docs = []
    processed_count = 0
    
    for page in [1, 2]:
        print(f"\n🔄 Processing page {page}...")
        page_data = fetch_topic_news_list(page=page, size=10)
        
        if not page_data:
            print(f"⚠️ No data from page {page}, skipping...")
            continue
        
        # ดึงรายการข่าวจากหน้านี้
        items = page_data.get("items", [])
        total = page_data.get("total", 0)
        print(f"📰 Found {len(items)} items on page {page} (total: {total})")
        
        for i, item in enumerate(items, 1):
            # ข้อมูลอยู่ใน item.article
            article = item.get("article", {})
            slug = article.get("slug", "")
            title = article.get("articleLocalized", {}).get("name", "")
            
            print(f"🔍 Processing item {i}/{len(items)}: {title[:50]}...")
            
            # ประมวลผลข้อมูลหัวข้อ (อาจดึงรายละเอียดเพิ่มหรือไม่ก็ได้)
            # ลองดึงรายละเอียดเต็ม (แต่ไม่บังคับ)
            detailed_data = None
            if slug:
                print(f"   📄 Fetching full content for: {slug}")
                detailed_data = fetch_article_detail_by_slug(slug)
                if detailed_data:
                    print(f"   ✅ Got detailed content")
                else:
                    print(f"   ⚠️ Using topic info only")
            
            # ประมวลผลข้อมูล
            doc = process_topic_news_item(item, detailed_data)
            if doc:
                docs.append(doc)
                processed_count += 1
                print(f"   ✅ Processed: {doc.metadata.get('title', slug)}")
            else:
                print(f"   ❌ Failed to process item")
            
            # หน่วงเวลาเล็กน้อยเพื่อไม่ให้ส่งคำขอเร็วเกินไป
            time.sleep(0.3)
    
    if len(docs) == 0:
        print("🚨 No topic news data found for vector creation")
        return False

    print(f"\n📝 Processed {len(docs)} topic news articles from 2 pages")

    # Split documents - ใช้ขนาดเล็กลงเพื่อไม่เกิด AstraDB limit
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"📄 Created {len(chunks)} chunks")

    # Initialize embeddings
    print("🧠 Initializing embeddings model...")
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Generate embeddings and insert to AstraDB
    print("💾 Inserting topic news data into AstraDB...")
    
    documents_to_insert = []
    for i, chunk in enumerate(chunks):
        # Generate embedding
        vector = embedding.embed_query(chunk.page_content)
        
        # Prepare document for insertion
        doc = {
            "_id": str(uuid.uuid4()),
            "content": chunk.page_content,
            "$vector": vector,
            "metadata": chunk.metadata
        }
        documents_to_insert.append(doc)
        
        if i % 5 == 0:
            print(f"📊 Processed {i+1}/{len(chunks)} chunks...")
    
    # Insert all documents
    try:
        result = collection.insert_many(documents_to_insert)
        print(f"✅ Successfully inserted {len(result.inserted_ids)} topic news documents into AstraDB!")
    except Exception as e:
        print(f"❌ Failed to insert documents: {e}")
        return False
    
    # Verify insertion
    try:
        count = collection.count_documents({}, upper_bound=1000)
        print(f"🔍 Total documents in {collection_name} collection: {count}")
    except Exception as e:
        print(f"⚠️ Could not verify document count: {e}")
    
    print("🎉 Topic News AstraDB ingestion completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
