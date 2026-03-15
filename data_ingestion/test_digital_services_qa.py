# test_digital_services_qa.py - Q&A Testing for Digital Services
# ทดสอบการถามคำถามเกี่ยวกับบริการดิจิตอล

import os
import numpy as np
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from astrapy import DataAPIClient
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Collection name
COLLECTION_NAME = "newdigital_services_embedding"

# Initialize embedding model
print("\n🧠 Loading embedding model (multilingual-e5-large)...")
embedding = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("✅ Model loaded successfully!\n")

# Initialize AstraDB with new API
token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
keyspace = os.getenv("ASTRA_DB_KEYSPACE", "default_keyspace")

if not token or not api_endpoint:
    raise ValueError("❌ Missing ASTRA_DB credentials in .env file")

# Connect to collection using DataAPIClient
print(f"🔌 Connecting to collection: {COLLECTION_NAME}...")
client = DataAPIClient(token)
database = client.get_database(api_endpoint, keyspace=keyspace)
collection = database.get_collection(COLLECTION_NAME)
print("✅ Connected successfully!\n")


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """คำนวณ cosine similarity ระหว่าง 2 vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def search_and_display(query: str, k: int = 3, show_content: bool = True):
    """
    ค้นหาและแสดงผลลัพธ์
    """
    print(f"❓ Query: {query}")
    print("-" * 70)
    
    try:
        # Generate query embedding
        query_vector = embedding.embed_query(query)
        
        # Search using vector similarity
        results = collection.find(
            sort={"$vector": query_vector},
            limit=k,
            projection={"*": True},
            include_similarity=True
        )
        
        results_list = list(results)
        
        if not results_list:
            print("❌ No results found\n")
            return []
        
        print(f"✅ Found {len(results_list)} results:\n")
        
        for idx, doc in enumerate(results_list, 1):
            similarity = doc.get('$similarity', 0)
            metadata = doc
            content = metadata.get('content', metadata.get('text', 'N/A'))
            
            print(f"📄 Result {idx} (Similarity: {similarity:.4f})")
            print(f"   Service: {metadata.get('service_name', 'N/A')}")
            print(f"   Category: {metadata.get('category', 'N/A')}")
            print(f"   URL: {metadata.get('url', 'N/A')}")
            
            if show_content and content != 'N/A':
                display_content = content[:300] + "..." if len(content) > 300 else content
                print(f"   Content: {display_content}")
            
            print()
        
        return results_list
        
    except Exception as e:
        print(f"❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        return []


def run_qa_tests():
    """
    รัน Q&A tests ทั้งหมด
    """
    print("\n" + "="*70)
    print("🧪 Digital Services Q&A Testing")
    print("="*70 + "\n")
    
    # Test queries (ภาษาไทย + อังกฤษ)
    test_queries = [
        # Web Hosting
        {
            "category": "Web Hosting",
            "queries": [
                "อยากทำเว็บไซต์ต้องใช้บริการอะไร",
                "ต้องการ host เว็บไซต์",
                "I need web hosting service",
                "How to host my website at KKU"
            ]
        },
        
        # Virtual Machine
        {
            "category": "Virtual Machine / Cloud",
            "queries": [
                "ต้องการเครื่องเสมือนสำหรับทดสอบโปรแกรม",
                "อยากได้ VM",
                "Need a virtual machine for development",
                "Cloud computing service"
            ]
        },
        
        # AI Services
        {
            "category": "AI Services",
            "queries": [
                "อยากใช้ ChatGPT แบบ Plus",
                "บริการ AI สำหรับนักศึกษา",
                "Need AI tools for research",
                "Server for machine learning"
            ]
        },
        
        # Storage
        {
            "category": "Storage / Backup",
            "queries": [
                "ต้องการพื้นที่จัดเก็บข้อมูลขนาดใหญ่",
                "อยากสำรองข้อมูล backup ไฟล์",
                "Need network storage",
                "NAS service for research data"
            ]
        },
        
        # GPU Computing
        {
            "category": "GPU / High Performance",
            "queries": [
                "ต้องการใช้ GPU สำหรับ deep learning",
                "อยาก train โมเดล AI",
                "Need H100 GPU for AI training",
                "High performance computing"
            ]
        },
        
        # Software & Tools
        {
            "category": "Software & Apps",
            "queries": [
                "อยากได้ app จาก App Store",
                "ต้องการโปรแกรมตรวจไวยากรณ์",
                "Need Grammarly for writing",
                "Download apps from Google Play"
            ]
        },
        
        # File Sharing
        {
            "category": "File Sharing",
            "queries": [
                "อยากส่งไฟล์ขนาดใหญ่",
                "แชร์ไฟล์ระหว่างเครื่อง",
                "Transfer files between devices",
                "Share files quickly"
            ]
        },
        
        # General Questions
        {
            "category": "General / Multi-Service",
            "queries": [
                "มีบริการอะไรบ้างสำหรับนักศึกษา",
                "อยากรู้บริการดิจิตอลทั้งหมด",
                "What digital services are available",
                "Services for computer science students"
            ]
        }
    ]
    
    # Run tests
    total_queries = 0
    successful_queries = 0
    
    for test_group in test_queries:
        category = test_group["category"]
        queries = test_group["queries"]
        
        print("\n" + "="*70)
        print(f"📂 Testing Category: {category}")
        print("="*70 + "\n")
        
        for query in queries:
            total_queries += 1
            results = search_and_display(query, k=2, show_content=False)
            
            if results:
                successful_queries += 1
            
            print()
    
    # Summary
    print("\n" + "="*70)
    print("📊 TESTING SUMMARY")
    print("="*70 + "\n")
    print(f"   Total queries: {total_queries}")
    print(f"   Successful: {successful_queries}")
    print(f"   Failed: {total_queries - successful_queries}")
    print(f"   Success rate: {(successful_queries / total_queries * 100):.1f}%")
    print("\n" + "="*70 + "\n")


def interactive_mode():
    """
    โหมดถามคำถามแบบ interactive
    """
    print("\n" + "="*70)
    print("💬 Interactive Q&A Mode")
    print("="*70)
    print("Type your question (or 'exit' to quit)\n")
    
    while True:
        try:
            query = input("❓ Your question: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not query:
                continue
            
            print()
            search_and_display(query, k=3, show_content=True)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


def compare_thai_english():
    """
    เปรียบเทียบผลลัพธ์ภาษาไทย vs อังกฤษ
    """
    print("\n" + "="*70)
    print("🔄 Thai vs English Comparison")
    print("="*70 + "\n")
    
    comparison_pairs = [
        ("อยากทำเว็บไซต์", "I want to create a website"),
        ("ต้องการเครื่องเสมือน", "I need a virtual machine"),
        ("ใช้ ChatGPT", "Use ChatGPT"),
        ("จัดเก็บข้อมูล", "Store data"),
        ("ส่งไฟล์", "Share files")
    ]
    
    for thai_q, eng_q in comparison_pairs:
        print(f"🇹🇭 Thai: {thai_q}")
        thai_vector = embedding.embed_query(thai_q)
        thai_results = list(collection.find(
            sort={"$vector": thai_vector},
            limit=1,
            projection={"service_name": True, "$similarity": True},
            include_similarity=True
        ))
        if thai_results:
            doc = thai_results[0]
            print(f"   → {doc.get('service_name', 'N/A')} (similarity: {doc.get('$similarity', 0):.4f})")
        
        print(f"🇬🇧 English: {eng_q}")
        eng_vector = embedding.embed_query(eng_q)
        eng_results = list(collection.find(
            sort={"$vector": eng_vector},
            limit=1,
            projection={"service_name": True, "$similarity": True},
            include_similarity=True
        ))
        if eng_results:
            doc = eng_results[0]
            print(f"   → {doc.get('service_name', 'N/A')} (similarity: {doc.get('$similarity', 0):.4f})")
        
        print("-" * 70 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        # Interactive mode
        interactive_mode()
    elif len(sys.argv) > 1 and sys.argv[1] == "compare":
        # Comparison mode
        compare_thai_english()
    else:
        # Run all Q&A tests
        run_qa_tests()
        
        # Ask if user wants interactive mode
        print("\n💡 Want to try interactive mode?")
        print("   Run: python test_digital_services_qa.py interactive\n")
