import requests
import json
from dotenv import load_dotenv
import os
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from astrapy import DataAPIClient
import uuid
import urllib3

# Disable SSL warnings (for development only)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

def main():
    print("🚀 Starting AstraDB ingestion with astrapy...")
    
    # Check environment variables
    token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
    keyspace = os.getenv("ASTRA_DB_KEYSPACE")  # ให้เป็น None ถ้าไม่ระบุ
    
    if not token or not api_endpoint:
        print("❌ Error: Missing AstraDB credentials in .env file")
        print("Please add:")
        print("ASTRA_DB_APPLICATION_TOKEN=your_token_here")
        print("ASTRA_DB_API_ENDPOINT=your_endpoint_here")
        return False
    
    print(f"🔑 Using endpoint: {api_endpoint}")
    if keyspace:
        print(f"🏠 Using keyspace: {keyspace}")
    else:
        print(f"🏠 Using default keyspace")
    
    # Initialize AstraDB client
    try:
        client = DataAPIClient(token=token)
        database = client.get_database_by_api_endpoint(api_endpoint)
        print("✅ Connected to AstraDB successfully!")
    except Exception as e:
        print(f"❌ Failed to connect to AstraDB: {e}")
        return False
    
    # Get existing collection (should be created via AstraDB UI with vector support)
    collection_name = "allpeople_embedding"
    try:
        # Try to connect with keyspace first, if fails try without
        try:
            if keyspace:
                database = database.with_options(keyspace=keyspace)
                print(f"🔄 Trying with keyspace: {keyspace}")
                existing_collections = list(database.list_collection_names())
                print(f"✅ Successfully connected with keyspace: {keyspace}")
            else:
                print(f"🔄 Connecting without keyspace...")
                existing_collections = list(database.list_collection_names())
        except Exception as e:
            if "does not exist" in str(e).lower() and keyspace:
                print(f"⚠️  Keyspace '{keyspace}' not found, trying without keyspace...")
                # Reset database to original (without keyspace)
                client = DataAPIClient(token=token)
                database = client.get_database_by_api_endpoint(api_endpoint)
                existing_collections = list(database.list_collection_names())
                print(f"✅ Successfully connected without keyspace")
            else:
                raise
        
        # List existing collections
        print(f"📂 Existing collections: {existing_collections}")
        
        if collection_name in existing_collections:
            collection = database.get_collection(collection_name)
            print(f"📂 Using existing collection: {collection_name}")
        else:
            print(f"❌ Collection {collection_name} not found!")
            print("Please create the collection via AstraDB UI with vector support:")
            print("  - Collection Name: faculty_embeddings")
            print("  - Vector Dimension: 384")
            print("  - Vector Metric: cosine")
            return False
            
    except Exception as e:
        print(f"❌ Failed to access collection: {e}")
        return False
    
    # Fetch data from API (same as original)
    print("🌐 Fetching faculty data from API...")
    try:
        url = "https://api.computing.kku.ac.th/api/v1/user/getUserByClassIds/?classId=[1,+2,+3]"
        res = requests.get(url, verify=False)  # Disable SSL verification for development
        data = res.json()
        users_raw = data["data"]["items"]
        print(f"📊 Retrieved {len(users_raw)} faculty records")
    except Exception as e:
        print(f"❌ Failed to fetch data from API: {e}")
        return False
    
    # Process data
    docs = []
    for item in users_raw:
        raw_localized = item.get("userLocalized", [])
        if isinstance(raw_localized, str):
            try:
                localized_list = json.loads(raw_localized)
            except json.JSONDecodeError:
                localized_list = []
        else:
            localized_list = raw_localized

        # Get both Thai (languageId=1) and English (languageId=2) data
        thai_localized = next((u for u in localized_list if u.get("languageId") == 1), {})
        english_localized = next((u for u in localized_list if u.get("languageId") == 2), {})

        # Basic info
        slug = item.get("slug", "").strip()
        
        # Thai info
        firstname_th = (thai_localized.get("firstname") or "").strip()
        lastname_th = (thai_localized.get("lastname") or "").strip()
        
        # English info
        firstname_en = (english_localized.get("firstname") or "").strip()
        lastname_en = (english_localized.get("lastname") or "").strip()
        
        # Other info
        position = (item.get("academicPosition") or "").strip()
        email = (item.get("email") or "").strip()
        tel = (item.get("telephone") or "").strip()
        
        # Get specialized and research descriptions from localized data
        specialize_desc = (thai_localized.get("specializeDescription") or "").strip()
        research_desc = (thai_localized.get("researchDescription") or "").strip()
        
        # Clean up HTML/markdown formatting from descriptions
        import re
        if specialize_desc:
            specialize_desc = re.sub(r'<[^>]+>', '', specialize_desc)  # Remove HTML tags
            specialize_desc = re.sub(r'\n+', ' ', specialize_desc)     # Replace multiple newlines
            specialize_desc = re.sub(r'\s+', ' ', specialize_desc).strip()  # Clean whitespace
        
        if research_desc:
            research_desc = re.sub(r'<[^>]+>', '', research_desc)      # Remove HTML tags
            research_desc = re.sub(r'\n+', ' ', research_desc)         # Replace multiple newlines
            research_desc = re.sub(r'\s+', ' ', research_desc).strip()  # Clean whitespace
            # Truncate research description if too long (keep first 1000 chars)
            if len(research_desc) > 1000:
                research_desc = research_desc[:1000] + "..."

        if firstname_th:
            # Create comprehensive content with both Thai and English names
            content_parts = [
                f"ชื่อ: {firstname_th} {lastname_th}",
                f"ชื่อภาษาอังกฤษ: {firstname_en} {lastname_en}" if firstname_en else "",
                f"Slug: {slug}" if slug else "",
                f"ตำแหน่ง: {position}",
                f"อีเมล: {email}",
                f"เบอร์โทร: {tel}",
                f"ความเชี่ยวชาญ: {specialize_desc}" if specialize_desc else "",
                f"ผลงานวิจัย: {research_desc}" if research_desc else ""
            ]
            
            # Filter out empty parts
            content_parts = [part for part in content_parts if part.strip()]
            content = "\n".join(content_parts)
            
            # Create metadata with slug for better linking
            metadata = {
                "slug": slug,
                "firstname_th": firstname_th,
                "lastname_th": lastname_th,
                "firstname_en": firstname_en,
                "lastname_en": lastname_en,
                "name": f"{firstname_th} {lastname_th}",  # เพิ่ม name field
                "position": position,  # เพิ่ม position field
                "email": email,
                "type": "allpeople"  # เปลี่ยนจาก basic_faculty เป็น allpeople
            }
            
            docs.append(Document(page_content=content, metadata=metadata))
            display_name = f"{firstname_th} {lastname_th}"
            if firstname_en:
                display_name += f" ({firstname_en} {lastname_en})"
            print(f"✅ Processed: {display_name} [slug: {slug}]")

    if len(docs) == 0:
        print("🚨 No data found for vector creation")
        return False

    print(f"📝 Processed {len(docs)} documents")

    # Initialize embeddings
    print("🧠 Initializing embeddings model...")
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Use incremental indexing
    print("\n🚀 Starting incremental indexing...")
    try:
        from incremental_utils import enable_incremental_mode
        
        stats = enable_incremental_mode(
            collection=collection,
            embedding_model=embedding,
            new_documents=docs,
            metadata_filter={"type": "allpeople"},  # Filter สำหรับดึงเอกสารบุคลากร
            hash_keys=["name", "position"],  # Keys สำหรับสร้าง unique hash
            delete_missing=False  # ไม่ลบเอกสารเก่า
        )
        
        print(f"\n✅ Incremental indexing completed!")
        print(f"   - New documents inserted: {stats['inserted']}")
        print(f"   - Existing documents skipped: {stats['skipped']}")
        
    except Exception as e:
        print(f"❌ Failed to process incremental indexing: {e}")
        return False
    
    # Verify insertion
    try:
        count = collection.count_documents({})
        print(f"🔍 Total documents in collection: {count}")
    except Exception as e:
        print(f"⚠️ Could not verify document count: {e}")
    
    print("🎉 AstraDB ingestion completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)