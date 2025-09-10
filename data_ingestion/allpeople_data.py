import json
import os
import uuid

import requests
from astrapy import DataAPIClient
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

def main():
    print("🚀 Starting AstraDB ingestion with astrapy...")
    
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
    
    # Get existing collection (should be created via AstraDB UI with vector support)
    collection_name = "allpeople_embedding"
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
        res = requests.get(url)
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
                "email": email,
                "type": "basic_faculty"
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

    # Split documents
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"📄 Created {len(chunks)} chunks")

    # Initialize embeddings
    print("🧠 Initializing embeddings model...")
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Generate embeddings and insert to AstraDB
    print("💾 Inserting data into AstraDB...")
    
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
        
        if i % 10 == 0:
            print(f"📊 Processed {i+1}/{len(chunks)} chunks...")
    
    # Insert all documents
    try:
        result = collection.insert_many(documents_to_insert)
        print(f"✅ Successfully inserted {len(result.inserted_ids)} documents into AstraDB!")
    except Exception as e:
        print(f"❌ Failed to insert documents: {e}")
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
