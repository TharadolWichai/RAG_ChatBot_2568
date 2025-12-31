import requests
import json
from dotenv import load_dotenv
import os
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from astrapy import DataAPIClient
import uuid
import re
from incremental_utils import enable_incremental_mode

load_dotenv()

# ข้อมูล API endpoints ของทุนการศึกษา
SCHOLARSHIP_APIS = {
    "research_support": {
        "id": 9,
        "url": "https://api.computing.kku.ac.th/api/v1/scholarshipItem/9",
        "name": "ทุนส่งเสริมความเข้มแข็งด้านการวิจัยของอาจารย์ผ่านการสนับสนุนนักศึกษาระดับบัณฑิตศึกษา"
    },
    "international_graduate": {
        "id": 13,
        "url": "https://api.computing.kku.ac.th/api/v1/scholarshipItem/13",
        "name": "ทุนระดับบัณฑิตศึกษาสำหรับนักศึกษาระดับบัณฑิตศึกษาหลักสูตรนานาชาติ"
    },
    "asean_gms": {
        "id": 14,
        "url": "https://api.computing.kku.ac.th/api/v1/scholarshipItem/14",
        "name": "ทุนสนับสนุนการศึกษาสำหรับบุคลากรจากประเทศในภูมิภาคอาเซียนและอนุภูมิภาคลุ่มแม่น้ำโขง (ทุน ASEAN & GMS)"
    },
    "excellent_student": {
        "id": 15,
        "url": "https://api.computing.kku.ac.th/api/v1/scholarshipItem/15",
        "name": "ทุนผู้มีผลการเรียนดีเยี่ยมระดับบัณฑิตศึกษา"
    }
}

def clean_html_text(text):
    """ทำความสะอาดข้อความจาก HTML tags และจัดรูปแบบ"""
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Replace multiple newlines with single space
    text = re.sub(r'\n+', ' ', text)
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_scholarship_content(scholarship_data, scholarship_info):
    """แยกข้อมูลทุนการศึกษาออกเป็นส่วนต่างๆ ตามโครงสร้าง API จริง"""
    
    documents = []
    
    # ข้อมูลพื้นฐานของทุน
    scholarship_id = scholarship_data.get("id")
    slug = scholarship_data.get("slug", "")
    
    # ข้อมูลจาก scholarshipItemLocalized (ชื่อหลัก)
    main_localized = scholarship_data.get("scholarshipItemLocalized", {})
    if main_localized is None:
        main_localized = {}
    main_title = clean_html_text(main_localized.get("name", ""))
    main_description = clean_html_text(main_localized.get("description", ""))
    
    # ข้อมูลจาก scholarship.scholarshipLocalized (ชื่อรอง)
    scholarship_info_data = scholarship_data.get("scholarship", {})
    if scholarship_info_data is None:
        scholarship_info_data = {}
    scholarship_localized = scholarship_info_data.get("scholarshipLocalized", {})
    if scholarship_localized is None:
        scholarship_localized = {}
    sub_title = clean_html_text(scholarship_localized.get("name", ""))
    sub_description = clean_html_text(scholarship_localized.get("description", ""))
    
    # ข้อมูลจาก scholarshipItemDetail_Mapping (รายละเอียดแต่ละส่วน)
    detail_mappings = scholarship_data.get("scholarshipItemDetail_Mapping", [])
    
    # สร้าง document หลักของทุน
    main_content_parts = []
    
    # ชื่อทุนหลัก
    if main_title:
        main_content_parts.append(f"ชื่อทุน: {main_title}")
    if sub_title and sub_title != main_title:
        main_content_parts.append(f"ชื่อทุนย่อย: {sub_title}")
    
    # คำอธิบายทุน
    if main_description:
        main_content_parts.append(f"รายละเอียดทุน: {main_description}")
    if sub_description and sub_description != main_description:
        main_content_parts.append(f"รายละเอียดเพิ่มเติม: {sub_description}")
    
    # Slug
    if slug:
        main_content_parts.append(f"Slug: {slug}")
    
    main_content = "\n".join(main_content_parts)
    
    # สร้าง metadata หลัก
    main_metadata = {
        "scholarship_id": scholarship_id,
        "scholarship_type": scholarship_info["name"],
        "scholarship_key": scholarship_info.get("key", ""),
        "title": main_title,
        "slug": slug,
        "type": "scholarship_main",
        "section": "overview"
    }
    
    # เพิ่ม document หลัก
    if main_content.strip():
        documents.append(Document(page_content=main_content, metadata=main_metadata))
    
    # ประมวลผลรายละเอียดแต่ละส่วน
    for detail_mapping in detail_mappings:
        detail_info = detail_mapping.get("scholarshipItemDetail", {})
        detail_localizations = detail_info.get("scholarshipItemDetailLocalized", [])
        
        # หาข้อมูลภาษาไทย (languageId = 1) และอังกฤษ (languageId = 2)
        thai_detail = None
        english_detail = None
        
        for localization in detail_localizations:
            if localization.get("languageId") == 1:
                thai_detail = localization
            elif localization.get("languageId") == 2:
                english_detail = localization
        
        # สร้าง document สำหรับแต่ละส่วน (ใช้ข้อมูลภาษาไทยเป็นหลัก)
        if thai_detail:
            section_title = clean_html_text(thai_detail.get("title", ""))
            section_description = clean_html_text(thai_detail.get("description", ""))
            
            # แปลง JSON description ถ้าเป็น string
            if section_description.startswith('[{'):
                try:
                    import json
                    desc_data = json.loads(section_description)
                    if isinstance(desc_data, list) and len(desc_data) > 0:
                        section_description = clean_html_text(desc_data[0].get("text", ""))
                except:
                    pass
            
            # สร้างเนื้อหาส่วนนี้
            section_content_parts = [
                f"ทุน: {main_title}",
                f"หัวข้อ: {section_title}"
            ]
            
            if section_description:
                section_content_parts.append(f"รายละเอียด: {section_description}")
            
            # เพิ่มข้อมูลภาษาอังกฤษถ้ามี
            if english_detail:
                english_title = clean_html_text(english_detail.get("title", ""))
                english_desc = clean_html_text(english_detail.get("description", ""))
                
                # แปลง JSON description ถ้าเป็น string
                if english_desc.startswith('[{'):
                    try:
                        import json
                        desc_data = json.loads(english_desc)
                        if isinstance(desc_data, list) and len(desc_data) > 0:
                            english_desc = clean_html_text(desc_data[0].get("text", ""))
                    except:
                        pass
                
                if english_title:
                    section_content_parts.append(f"หัวข้อภาษาอังกฤษ: {english_title}")
                if english_desc:
                    section_content_parts.append(f"รายละเอียดภาษาอังกฤษ: {english_desc}")
            
            section_content = "\n".join(section_content_parts)
            
            # สร้าง metadata สำหรับส่วนนี้
            section_metadata = {
                "scholarship_id": scholarship_id,
                "scholarship_type": scholarship_info["name"],
                "scholarship_key": scholarship_info.get("key", ""),
                "title": main_title,
                "section_title": section_title,
                "slug": slug,
                "type": "scholarship_detail",
                "section": section_title.lower().replace(" ", "_")
            }
            
            # เพิ่ม document สำหรับส่วนนี้
            if section_content.strip():
                documents.append(Document(page_content=section_content, metadata=section_metadata))
    
    return documents

def fetch_scholarship_data(api_info):
    """ดึงข้อมูลทุนจาก API"""
    try:
        print(f"🌐 Fetching data from: {api_info['name']}")
        
        # Disable SSL warnings and verification for this API
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        response = requests.get(api_info['url'], verify=False)
        response.raise_for_status()
        
        data = response.json()
        if data.get("status") == "success" and data.get("data"):
            return data["data"]
        else:
            print(f"❌ No data found in API response for {api_info['name']}")
            return None
            
    except Exception as e:
        print(f"❌ Error fetching data from {api_info['url']}: {e}")
        return None

def main():
    print("🚀 Starting Scholarship Data Ingestion to AstraDB...")
    
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
    
    # Get or create collection for scholarships
    collection_name = "scholarship_embedding"
    try:
        existing_collections = list(database.list_collection_names())
        print(f"📂 Existing collections: {existing_collections}")
        
        if collection_name in existing_collections:
            collection = database.get_collection(collection_name)
            print(f"📂 Using existing collection: {collection_name}")
            print("🔄 Using incremental indexing mode (no full delete)")
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
    
    # Fetch and process scholarship data
    all_documents = []
    
    for key, api_info in SCHOLARSHIP_APIS.items():
        print(f"\n📚 Processing scholarship: {api_info['name']}")
        
        # Fetch data from API
        scholarship_data = fetch_scholarship_data(api_info)
        if not scholarship_data:
            print(f"⚠️ Skipping {api_info['name']} due to fetch error")
            continue
        
        # Add key to api_info for metadata
        api_info_with_key = api_info.copy()
        api_info_with_key["key"] = key
        
        # Extract content and metadata (returns list of documents)
        try:
            documents = extract_scholarship_content(scholarship_data, api_info_with_key)
            
            if documents:
                all_documents.extend(documents)
                main_title = documents[0].metadata.get('title', 'Unknown')
                print(f"✅ Processed: {main_title[:50]}... ({len(documents)} sections)")
            else:
                print(f"⚠️ No content extracted for {api_info['name']}")
                
        except Exception as e:
            print(f"❌ Error processing {api_info['name']}: {e}")
            continue
    
    if len(all_documents) == 0:
        print("🚨 No scholarship data found for vector creation")
        return False
    
    print(f"\n📝 Processed {len(all_documents)} scholarship documents")
    
    # Initialize embeddings
    print("🧠 Initializing embeddings model...")
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Use incremental indexing
    print("\n🚀 Starting incremental indexing...")
    try:
        stats = enable_incremental_mode(
            collection=collection,
            embedding_model=embedding,
            new_documents=all_documents,
            metadata_filter={"type": "scholarship_main"},  # Filter สำหรับดึงเอกสารทุนการศึกษา
            hash_keys=["scholarship_id", "scholarship_type", "section"],  # Keys สำหรับสร้าง unique hash
            delete_missing=False  # ไม่ลบเอกสารเก่า (ปรับเป็น True ถ้าต้องการลบข้อมูลที่หายไป)
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
        print(f"🔍 Total scholarship documents in collection: {count}")
    except Exception as e:
        print(f"⚠️ Could not verify document count: {e}")
    
    # Show summary of scholarships processed
    print("\n📋 Summary of scholarships processed:")
    print("-" * 60)
    for doc in all_documents:
        title = doc.metadata.get('title', 'Unknown')
        scholarship_type = doc.metadata.get('scholarship_type', 'Unknown')
        print(f"• {title[:50]}...")
        print(f"  Type: {scholarship_type}")
        print("-" * 40)
    
    print("🎉 Scholarship data ingestion completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
