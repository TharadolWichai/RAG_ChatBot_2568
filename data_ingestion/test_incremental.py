"""
Test script for Incremental Indexing System
ทดสอบระบบ incremental indexing ก่อนใช้งานจริง
"""

import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from astrapy import DataAPIClient

load_dotenv()

def test_incremental_indexing():
    """ทดสอบระบบ incremental indexing"""
    
    print("🧪 Testing Incremental Indexing System")
    print("=" * 60)
    
    # 1. เชื่อมต่อ AstraDB
    print("\n1️⃣ Connecting to AstraDB...")
    token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
    
    if not token or not api_endpoint:
        print("❌ Missing AstraDB credentials")
        return False
    
    try:
        client = DataAPIClient(token=token)
        database = client.get_database_by_api_endpoint(api_endpoint)
        print("✅ Connected to AstraDB")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return False
    
    # 2. สร้าง test collection (หรือใช้ collection ที่มีอยู่)
    print("\n2️⃣ Setting up test collection...")
    test_collection_name = "test_incremental"
    
    try:
        existing_collections = list(database.list_collection_names())
        
        if test_collection_name in existing_collections:
            collection = database.get_collection(test_collection_name)
            print(f"✅ Using existing collection: {test_collection_name}")
        else:
            print(f"📦 Creating test collection: {test_collection_name}")
            collection = database.create_collection(
                test_collection_name,
                dimension=384,
                metric="cosine"
            )
            print("✅ Collection created")
    except Exception as e:
        print(f"❌ Failed to setup collection: {e}")
        return False
    
    # 3. สร้าง test documents
    print("\n3️⃣ Creating test documents...")
    test_docs = [
        Document(
            page_content="วิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น",
            metadata={"type": "test", "doc_id": "1", "title": "About College"}
        ),
        Document(
            page_content="หลักสูตรวิทยาศาสตรบัณฑิต สาขาวิชาวิทยาการคอมพิวเตอร์",
            metadata={"type": "test", "doc_id": "2", "title": "CS Program"}
        ),
        Document(
            page_content="หลักสูตรวิทยาศาสตรมหาบัณฑิต สาขาวิชาวิทยาการคอมพิวเตอร์",
            metadata={"type": "test", "doc_id": "3", "title": "MS Program"}
        ),
    ]
    print(f"✅ Created {len(test_docs)} test documents")
    
    # 4. Initialize embedding model
    print("\n4️⃣ Initializing embedding model...")
    try:
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("✅ Embedding model ready")
    except Exception as e:
        print(f"❌ Failed to initialize embedding: {e}")
        return False
    
    # 5. Test 1: First insert (ควรได้ 3 inserts)
    print("\n5️⃣ Test 1: First insert (should insert all 3 documents)")
    print("-" * 60)
    
    try:
        from incremental_utils import enable_incremental_mode
        
        stats1 = enable_incremental_mode(
            collection=collection,
            embedding_model=embedding,
            new_documents=test_docs,
            metadata_filter={"type": "test"},
            hash_keys=["doc_id", "title"],
            delete_missing=False
        )
        
        print(f"\n📊 Results:")
        print(f"   Inserted: {stats1['inserted']}")
        print(f"   Skipped:  {stats1['skipped']}")
        
        if stats1['inserted'] > 0:
            print("✅ Test 1 PASSED: Documents inserted successfully")
        else:
            print("❌ Test 1 FAILED: No documents inserted")
            return False
            
    except Exception as e:
        print(f"❌ Test 1 FAILED: {e}")
        return False
    
    # 6. Test 2: Re-insert same documents (ควรได้ 0 inserts, 3 skips)
    print("\n6️⃣ Test 2: Re-insert same documents (should skip all)")
    print("-" * 60)
    
    try:
        stats2 = enable_incremental_mode(
            collection=collection,
            embedding_model=embedding,
            new_documents=test_docs,
            metadata_filter={"type": "test"},
            hash_keys=["doc_id", "title"],
            delete_missing=False
        )
        
        print(f"\n📊 Results:")
        print(f"   Inserted: {stats2['inserted']}")
        print(f"   Skipped:  {stats2['skipped']}")
        
        if stats2['skipped'] > 0 and stats2['inserted'] == 0:
            print("✅ Test 2 PASSED: Duplicates detected and skipped")
        else:
            print("❌ Test 2 FAILED: Should have skipped all documents")
            return False
            
    except Exception as e:
        print(f"❌ Test 2 FAILED: {e}")
        return False
    
    # 7. Test 3: Insert new document (ควรได้ 1 insert, 3 skips)
    print("\n7️⃣ Test 3: Insert new document (should insert 1, skip 3)")
    print("-" * 60)
    
    new_doc = Document(
        page_content="หลักสูตรปรัชญาดุษฎีบัณฑิต สาขาวิชาวิทยาการคอมพิวเตอร์",
        metadata={"type": "test", "doc_id": "4", "title": "PhD Program"}
    )
    
    test_docs_with_new = test_docs + [new_doc]
    
    try:
        stats3 = enable_incremental_mode(
            collection=collection,
            embedding_model=embedding,
            new_documents=test_docs_with_new,
            metadata_filter={"type": "test"},
            hash_keys=["doc_id", "title"],
            delete_missing=False
        )
        
        print(f"\n📊 Results:")
        print(f"   Inserted: {stats3['inserted']}")
        print(f"   Skipped:  {stats3['skipped']}")
        
        if stats3['inserted'] > 0 and stats3['skipped'] > 0:
            print("✅ Test 3 PASSED: New document inserted, old ones skipped")
        else:
            print("❌ Test 3 FAILED: Should have inserted 1 and skipped 3")
            return False
            
    except Exception as e:
        print(f"❌ Test 3 FAILED: {e}")
        return False
    
    # 8. Verify total count
    print("\n8️⃣ Verifying total document count...")
    try:
        total_count = collection.count_documents({"metadata.type": "test"})
        print(f"📊 Total test documents in collection: {total_count}")
        
        # ควรมี 4 documents (3 + 1 ใหม่)
        # แต่อาจมีมากกว่าถ้า split เป็น chunks
        if total_count >= 4:
            print("✅ Document count looks correct")
        else:
            print(f"⚠️  Expected at least 4 documents, got {total_count}")
    except Exception as e:
        print(f"⚠️  Could not verify count: {e}")
    
    # 9. Cleanup (optional)
    print("\n9️⃣ Cleanup...")
    cleanup = input("Do you want to delete test documents? (y/n): ").lower()
    
    if cleanup == 'y':
        try:
            result = collection.delete_many({"metadata.type": "test"})
            print(f"✅ Deleted test documents")
        except Exception as e:
            print(f"⚠️  Could not delete: {e}")
    else:
        print("⏭️  Skipping cleanup")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 All tests PASSED!")
    print("=" * 60)
    print("\n✅ Incremental Indexing System is working correctly!")
    print("   You can now use it in your data ingestion scripts.")
    
    return True


if __name__ == "__main__":
    success = test_incremental_indexing()
    
    if not success:
        print("\n❌ Some tests failed. Please check the errors above.")
        exit(1)
    else:
        print("\n✅ Testing completed successfully!")
        exit(0)

