"""
Test บริการดิจิตอล - ทดสอบด้วยคำถามที่ตั้งไว้ล่วงหน้า
Simple Q&A testing for Digital Services with preset questions
"""
from dotenv import load_dotenv
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from astrapy import DataAPIClient

load_dotenv()

# Collection name
COLLECTION_NAME = "newdigital_services_embedding"

def test_digital_services():
    print("="*80)
    print("🧪 Digital Services Q&A Test")
    print("="*80)
    print()
    
    # Connect to AstraDB
    token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
    keyspace = os.getenv("ASTRA_DB_KEYSPACE", "default_keyspace")
    
    if not token or not api_endpoint:
        print("❌ Error: Missing AstraDB credentials")
        return
    
    # Connect to collection
    print(f"📦 Connecting to collection: {COLLECTION_NAME}...")
    client = DataAPIClient(token=token)
    database = client.get_database(api_endpoint, keyspace=keyspace)
    
    try:
        collection = database.get_collection(COLLECTION_NAME)
        print("   ✅ Connected successfully")
    except Exception as e:
        print(f"   ❌ Collection not found: {e}")
        return
    
    # Check document count
    try:
        count = collection.count_documents({}, upper_bound=1000)
        print(f"   📊 Total documents: {count}")
    except Exception as e:
        print(f"   ⚠️ Could not count documents: {e}")
    
    # Initialize embedding model
    print("\n🧠 Loading embedding model (multilingual-e5-large)...")
    embedding = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("   ✅ Model loaded successfully")
    
    # Test queries - คำถามที่ตั้งไว้ล่วงหน้า
    test_queries = [
        
        
        # General
        "บริการดิจิตอลทั้งหมด",
        
    ]
    
    print("\n" + "="*80)
    print("🧪 Running Tests")
    print("="*80)
    
    successful = 0
    failed = 0
    
    for query_idx, test_query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"❓ Test {query_idx}/{len(test_queries)}: '{test_query}'")
        print("-"*80)
        
        try:
            # Generate query embedding
            query_vector = embedding.embed_query(test_query)
            
            # Search
            results = list(collection.find(
                sort={"$vector": query_vector},
                limit=3,
                projection={
                    "service_name": True,
                    "category": True, 
                    "url": True,
                    "content": True
                },
                include_similarity=True
            ))
            
            if not results:
                print("   ❌ No results found")
                failed += 1
                continue
            
            print(f"   ✅ Found {len(results)} results:\n")
            successful += 1
            
            for i, doc in enumerate(results, 1):
                similarity = doc.get('$similarity', 0)
                service_name = doc.get('service_name', 'N/A')
                category = doc.get('category', 'N/A')
                url = doc.get('url', 'N/A')
                
                print(f"   {i}. {service_name}")
                print(f"      Category: {category}")
                print(f"      Similarity: {similarity:.4f}")
                print(f"      URL: {url}")
                
                # Show snippet of content
                content = doc.get('content', '')
                if content:
                    snippet = content[:150] + "..." if len(content) > 150 else content
                    print(f"      Preview: {snippet}")
                print()
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            failed += 1
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    print(f"   Total queries: {len(test_queries)}")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📈 Success rate: {(successful / len(test_queries) * 100):.1f}%")
    print("="*80)
    print()
    
    if successful == len(test_queries):
        print("🎉 All tests passed!")
    elif successful > 0:
        print("⚠️ Some tests failed, but system is partially working")
    else:
        print("❌ All tests failed, please check the system")
    
    print()
    print("💡 Tips:")
    print("   - ตรวจสอบว่าผลลัพธ์ตรงกับคำถามหรือไม่")
    print("   - ดู Similarity score ว่าสูงพอหรือเปล่า (>0.5 ดี)")
    print("   - ทดสอบทั้งภาษาไทยและอังกฤษ")

if __name__ == "__main__":
    test_digital_services()
