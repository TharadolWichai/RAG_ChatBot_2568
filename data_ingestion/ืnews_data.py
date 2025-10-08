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

def fetch_article_list():
    """ดึงรายการข่าวทั้งหมดจาก API"""
    print("🌐 Fetching article list from API...")
    try:
        # ลองใช้ endpoint หลายแบบเพื่อหารายการข่าว
        possible_endpoints = [
            "https://api.computing.kku.ac.th/api/v1/article/getArticles",
            "https://api.computing.kku.ac.th/api/v1/article/getArticleList",
            "https://api.computing.kku.ac.th/api/v1/article/getAllArticles",
            "https://api.computing.kku.ac.th/api/v1/article/getArticles?limit=100",
        ]
        
        for endpoint in possible_endpoints:
            try:
                print(f"🔍 Trying endpoint: {endpoint}")
                res = requests.get(endpoint, timeout=10)
                if res.status_code == 200:
                    data = res.json()
                    print(f"✅ Success with endpoint: {endpoint}")
                    return data, endpoint
                else:
                    print(f"❌ Failed with status {res.status_code}: {endpoint}")
            except Exception as e:
                print(f"❌ Error with {endpoint}: {e}")
                continue
        
        # หากไม่สามารถหารายการได้ ให้ใช้ข่าวตัวอย่างที่มี
        print("⚠️ Cannot fetch article list, using sample article")
        return None, None
        
    except Exception as e:
        print(f"❌ Failed to fetch article list: {e}")
        return None, None

def fetch_article_by_slug(slug):
    """ดึงข้อมูลข่าวแต่ละเรื่องตาม slug"""
    try:
        url = f"https://api.computing.kku.ac.th/api/v1/article/getArticleBySlug/{slug}"
        print(f"🔍 Fetching article: {slug}")
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
        print(f"❌ Failed to fetch article {slug}: {e}")
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
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def process_article_data(article_data):
    """ประมวลผลข้อมูลข่าวสำหรับ embedding"""
    if not article_data:
        return None
    
    try:
        # ข้อมูลพื้นฐาน
        article_id = article_data.get("id", "")
        slug = article_data.get("slug", "")
        
        # ข้อมูลภาษาไทย
        localized = article_data.get("articleLocalized", {})
        title = clean_html_text(localized.get("name", ""))
        short_desc = clean_html_text(localized.get("shortDescription", ""))
        description = clean_html_text(localized.get("description", ""))
        
        # วันที่
        created_at = article_data.get("createdAt", "")
        published_at = article_data.get("publishedAt", "")
        event_at = article_data.get("eventAt", "")
        
        # ข้อมูลผู้เขียน
        created_user = article_data.get("createdUser", {})
        author_name = f"{created_user.get('firstname', '')} {created_user.get('lastname', '')}".strip()
        author_position = created_user.get("academicPosition", "").strip()
        
        # หมวดหมู่
        categories = article_data.get("articleCategory_Mapping", [])
        category_names = []
        for cat in categories:
            cat_localized = cat.get("categoryLocalized", {})
            if cat_localized.get("name"):
                category_names.append(cat_localized.get("name"))
        
        # สร้างเนื้อหาสำหรับ embedding (จำกัดความยาวเข้มงวด)
        # ตัด description ให้สั้นลงถ้ายาวเกินไป
        max_desc_length = 2000  # จำกัดเนื้อหาหลักไม่เกิน 2000 ตัวอักษร
        if len(description) > max_desc_length:
            description = description[:max_desc_length] + "..."
        
        content_parts = [
            f"ชื่อข่าว: {title}",
            f"คำอธิบายสั้น: {short_desc}" if short_desc else "",
            f"เนื้อหา: {description}" if description else "",
            f"ผู้เขียน: {author_name}" if author_name else "",
            f"ตำแหน่งผู้เขียน: {author_position}" if author_position else "",
            f"หมวดหมู่: {', '.join(category_names)}" if category_names else "",
            f"Slug: {slug}",
        ]
        
        # กรองเนื้อหาที่ไม่ว่าง
        content_parts = [part for part in content_parts if part.strip()]
        content = "\n".join(content_parts)
        
        # Final check - ตัดให้ไม่เกิน 2500 characters เพื่อให้ปลอดภัยจาก AstraDB limit (8000 bytes)
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
            "type": "news_article"
        }
        
        return Document(page_content=content, metadata=metadata)
        
    except Exception as e:
        print(f"❌ Error processing article data: {e}")
        return None

def main():
    print("🚀 Starting News AstraDB ingestion with astrapy...")
    
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
    
    # Get or create collection for news
    collection_name = "news_embedding"
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
    
    # ลองดึงรายการข่าวทั้งหมด
    article_list, endpoint_used = fetch_article_list()
    
    # รายการ slug ข่าวที่ต้องการดึง (ถ้าไม่มี API รายการ)
    sample_slugs = [
        "2025-08-19-ai-mou",  # ข่าวตัวอย่างที่มี
        # เพิ่ม slug อื่นๆ ตามต้องการ
    ]
    
    docs = []
    processed_count = 0
    
    if article_list and endpoint_used:
        print(f"📰 Processing articles from {endpoint_used}")
        # ประมวลผลรายการข่าวที่ได้
        articles = []
        
        # ลองดึงข้อมูลจากโครงสร้าง API ที่แตกต่างกัน
        if isinstance(article_list, dict):
            if "data" in article_list:
                if "items" in article_list["data"]:
                    articles = article_list["data"]["items"]
                elif isinstance(article_list["data"], list):
                    articles = article_list["data"]
                else:
                    articles = [article_list["data"]]
            elif "items" in article_list:
                articles = article_list["items"]
            elif isinstance(article_list, list):
                articles = article_list
        elif isinstance(article_list, list):
            articles = article_list
        
        print(f"📊 Found {len(articles)} articles to process")
        
        for article in articles[:10]:  # จำกัด 10 ข่าวแรก
            slug = article.get("slug") if isinstance(article, dict) else str(article)
            if slug:
                article_data = fetch_article_by_slug(slug)
                if article_data:
                    doc = process_article_data(article_data)
                    if doc:
                        docs.append(doc)
                        processed_count += 1
                        print(f"✅ Processed: {doc.metadata.get('title', slug)}")
                
                # หน่วงเวลาเล็กน้อยเพื่อไม่ให้ส่งคำขอเร็วเกินไป
                time.sleep(0.5)
    else:
        print("📰 Using sample articles...")
        # ใช้ข่าวตัวอย่าง
        for slug in sample_slugs:
            article_data = fetch_article_by_slug(slug)
            if article_data:
                doc = process_article_data(article_data)
                if doc:
                    docs.append(doc)
                    processed_count += 1
                    print(f"✅ Processed: {doc.metadata.get('title', slug)}")
            
            # หน่วงเวลาเล็กน้อย
            time.sleep(0.5)
    
    if len(docs) == 0:
        print("🚨 No news data found for vector creation")
        return False

    print(f"📝 Processed {len(docs)} news articles")

    # Split documents - ใช้ขนาดเล็กลงเพื่อไม่เกิน AstraDB limit
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"📄 Created {len(chunks)} chunks")

    # Initialize embeddings
    print("🧠 Initializing embeddings model...")
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Generate embeddings and insert to AstraDB
    print("💾 Inserting news data into AstraDB...")
    
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
        print(f"✅ Successfully inserted {len(result.inserted_ids)} news documents into AstraDB!")
    except Exception as e:
        print(f"❌ Failed to insert documents: {e}")
        return False
    
    # Verify insertion
    try:
        count = collection.count_documents({})
        print(f"🔍 Total documents in news collection: {count}")
    except Exception as e:
        print(f"⚠️ Could not verify document count: {e}")
    
    print("🎉 News AstraDB ingestion completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
