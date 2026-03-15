"""
Test และเปรียบเทียบผลลัพธ์ระหว่าง embedding models เก่าและใหม่
"""
from dotenv import load_dotenv
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from astrapy import DataAPIClient

load_dotenv()

def test_comparison():
    print("="*80)
    print("🔍 Embedding Model Comparison Test")
    print("="*80)
    print()
    
    # Connect to AstraDB
    token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
    
    if not token or not api_endpoint:
        print("❌ Error: Missing AstraDB credentials")
        return
    
    client = DataAPIClient(token=token)
    database = client.get_database_by_api_endpoint(api_endpoint)
    
    # Get both collections
    print("📦 Loading collections...")
    try:
        old_collection = database.get_collection("allpeople_embedding")
        print("   ✅ OLD: allpeople_embedding (384 dim)")
    except Exception as e:
        print(f"   ⚠️ OLD collection not found: {e}")
        old_collection = None
    
    try:
        new_collection = database.get_collection("newallpeople_embedding")
        print("   ✅ NEW: newallpeople_embedding (1024 dim)")
    except Exception as e:
        print(f"   ❌ NEW collection not found: {e}")
        return
    
    # Initialize embeddings
    print("\n🧠 Loading embedding models...")
    print("   Loading OLD model (all-MiniLM-L6-v2)...")
    old_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("   ✅ OLD model loaded")
    
    print("   Loading NEW model (multilingual-e5-large)...")
    new_embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    print("   ✅ NEW model loaded")
    
    # Test queries
    test_queries = [
        "อาจารย์ที่สอนวิชา AI และ machine learning",
        "ผู้เชี่ยวชาญด้านปัญญาประดิษฐ์",
        "นักวิจัยด้าน deep learning",
        "งานวิจัยของอาจารย์พุธษดี",
        "ข้อมูลอาจารย์พุธษดี"
    ]
    
    print("\n" + "="*80)
    print("🧪 Running Tests")
    print("="*80)
    
    for query_idx, test_query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Test {query_idx}/{len(test_queries)}: '{test_query}'")
        print("="*80)
        
        # Test OLD embedding (if available)
        if old_collection:
            print("\n📊 OLD Embedding (all-MiniLM-L6-v2 - 384 dim):")
            try:
                query_vector_old = old_embedding.embed_query(test_query)
                results_old = list(old_collection.find(
                    {},
                    sort={"$vector": query_vector_old},
                    limit=5
                ))
                
                for i, doc in enumerate(results_old, 1):
                    metadata = doc.get("metadata", {})
                    name = metadata.get("name", "Unknown")
                    position = metadata.get("position", "")
                    print(f"  {i}. {name}")
                    if position:
                        print(f"     ตำแหน่ง: {position}")
                        
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Test NEW embedding
        print("\n📊 NEW Embedding (multilingual-e5-large - 1024 dim):")
        try:
            query_vector_new = new_embedding.embed_query(test_query)
            results_new = list(new_collection.find(
                {},
                sort={"$vector": query_vector_new},
                limit=5
            ))
            
            for i, doc in enumerate(results_new, 1):
                metadata = doc.get("metadata", {})
                name = metadata.get("name", "Unknown")
                position = metadata.get("position", "")
                print(f"  {i}. {name}")
                if position:
                    print(f"     ตำแหน่ง: {position}")
                    
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "="*80)
    print("✅ Test completed!")
    print("="*80)
    print()
    print("💡 Tips:")
    print("   - เปรียบเทียบว่า NEW model หาได้ตรงกว่าไหม")
    print("   - ดู ranking ว่าอันไหนดีกว่า")
    print("   - ทดสอบกับคำถามที่หลากหลาย")

if __name__ == "__main__":
    test_comparison()
