"""
Incremental Indexing Utilities
ระบบสำหรับจัดการการดึงและอัพเดทข้อมูลแบบ incremental
โดยจะดึงเฉพาะข้อมูลใหม่หรือที่เปลี่ยนแปลง ไม่ต้องดึงซ้ำทั้งหมด
"""

import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document


class IncrementalIndexManager:
    """
    จัดการ incremental indexing สำหรับ AstraDB
    
    Features:
    - สร้าง unique identifier จากเนื้อหา (content hash)
    - ตรวจสอบข้อมูลที่มีอยู่แล้ว
    - Insert เฉพาะข้อมูลใหม่
    - Update ข้อมูลที่เปลี่ยนแปลง
    - ลบข้อมูลเก่าที่ไม่มีในแหล่งข้อมูล (optional)
    """
    
    def __init__(self, collection, embedding_model):
        """
        Args:
            collection: AstraDB collection object
            embedding_model: Embedding model สำหรับสร้าง vectors
        """
        self.collection = collection
        self.embedding_model = embedding_model
        self.stats = {
            "inserted": 0,
            "updated": 0,
            "skipped": 0,
            "deleted": 0,
            "errors": 0
        }
    
    def generate_content_hash(self, content: str, additional_keys: Optional[Dict[str, Any]] = None) -> str:
        """
        สร้าง unique hash จากเนื้อหาและ metadata สำคัญ
        
        Args:
            content: เนื้อหาของ document
            additional_keys: metadata เพิ่มเติมที่ต้องการรวมในการสร้าง hash (เช่น article_id, slug)
            
        Returns:
            SHA-256 hash string
        """
        hash_input = content
        
        # รวม metadata สำคัญเข้าไปใน hash (ถ้ามี)
        if additional_keys:
            # Sort keys เพื่อให้ hash เหมือนกันเสมอ
            sorted_keys = sorted(additional_keys.items())
            hash_input += "|" + "|".join([f"{k}:{v}" for k, v in sorted_keys])
        
        # สร้าง SHA-256 hash
        return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
    
    def get_existing_document_hashes(self, metadata_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """
        ดึง content_hash ของเอกสารทั้งหมดที่มีอยู่ใน collection
        
        Args:
            metadata_filter: filter สำหรับดึงเฉพาะเอกสารบางประเภท (เช่น {"type": "news"})
            
        Returns:
            Dictionary ที่ map content_hash -> document info (_id, metadata)
        """
        existing_hashes = {}
        
        try:
            # Query เอกสารทั้งหมดที่ตรงกับ filter
            filter_query = {}
            if metadata_filter:
                for key, value in metadata_filter.items():
                    filter_query[f"metadata.{key}"] = value
            
            # ดึงเฉพาะ _id และ metadata (ไม่ระบุ projection เพื่อหลีกเลี่ยง conflict)
            # Note: ไม่ใช้ projection เพราะอาจเกิด conflict ระหว่าง metadata และ metadata.content_hash
            # Query documents
            cursor = self.collection.find(filter_query)
            
            for doc in cursor:
                content_hash = doc.get("metadata", {}).get("content_hash")
                if content_hash:
                    existing_hashes[content_hash] = {
                        "_id": doc.get("_id"),
                        "metadata": doc.get("metadata", {})
                    }
            
            print(f"📊 พบเอกสารเดิมในระบบ: {len(existing_hashes)} รายการ")
            
        except Exception as e:
            print(f"⚠️ ไม่สามารถดึงข้อมูลเอกสารเดิม: {e}")
            print("   จะดำเนินการแบบ insert ทั้งหมด")
        
        return existing_hashes
    
    def prepare_documents_with_hash(self, documents: List[Document], 
                                     hash_keys: Optional[List[str]] = None) -> List[Document]:
        """
        เพิ่ม content_hash ให้กับ documents
        
        Args:
            documents: List of LangChain Document objects
            hash_keys: List ของ metadata keys ที่ต้องการรวมในการสร้าง hash
            
        Returns:
            List of Documents ที่มี content_hash ใน metadata
        """
        for doc in documents:
            # สร้าง additional keys สำหรับ hash
            additional_keys = {}
            if hash_keys:
                for key in hash_keys:
                    if key in doc.metadata:
                        additional_keys[key] = doc.metadata[key]
            
            # สร้าง content hash
            content_hash = self.generate_content_hash(doc.page_content, additional_keys)
            doc.metadata["content_hash"] = content_hash
        
        return documents
    
    def process_incremental_update(self, 
                                   new_documents: List[Document],
                                   metadata_filter: Optional[Dict[str, Any]] = None,
                                   hash_keys: Optional[List[str]] = None,
                                   delete_missing: bool = False) -> Tuple[int, int, int, int]:
        """
        ประมวลผล incremental update
        
        Args:
            new_documents: Documents ใหม่ที่ต้องการเพิ่มหรืออัพเดท
            metadata_filter: Filter สำหรับดึงเอกสารเดิม
            hash_keys: Metadata keys ที่ใช้ในการสร้าง hash
            delete_missing: ถ้า True จะลบเอกสารเก่าที่ไม่มีในชุดข้อมูลใหม่
            
        Returns:
            Tuple (inserted, updated, skipped, deleted)
        """
        print("\n🔄 เริ่มต้น Incremental Indexing...")
        
        # 1. เพิ่ม content_hash ให้กับ documents ใหม่
        print("🔐 สร้าง content hash สำหรับเอกสารใหม่...")
        new_documents = self.prepare_documents_with_hash(new_documents, hash_keys)
        
        # 2. ดึง hash ของเอกสารที่มีอยู่แล้ว
        print("📥 ตรวจสอบเอกสารที่มีอยู่ในระบบ...")
        existing_hashes = self.get_existing_document_hashes(metadata_filter)
        
        # 3. จัดหมวดหมู่เอกสาร
        new_hash_to_doc = {}
        for doc in new_documents:
            content_hash = doc.metadata.get("content_hash")
            if content_hash:
                new_hash_to_doc[content_hash] = doc
        
        to_insert = []  # เอกสารใหม่ที่ต้อง insert
        to_update = []  # เอกสารที่ต้อง update
        to_skip = []    # เอกสารที่มีอยู่แล้วและไม่เปลี่ยนแปลง
        
        # 4. จำแนกเอกสารแต่ละตัว
        print("🔍 จำแนกเอกสารว่าต้อง insert, update หรือ skip...")
        for content_hash, doc in new_hash_to_doc.items():
            if content_hash in existing_hashes:
                # เอกสารมีอยู่แล้ว - ข้ามไป (หรืออาจตรวจสอบเพิ่มเติมว่าต้อง update หรือไม่)
                to_skip.append(doc)
            else:
                # เอกสารใหม่ - ต้อง insert
                to_insert.append(doc)
        
        # 5. Insert เอกสารใหม่
        if to_insert:
            print(f"\n✨ กำลัง insert เอกสารใหม่ {len(to_insert)} รายการ...")
            inserted = self._insert_documents(to_insert)
            self.stats["inserted"] = inserted
        else:
            print("\n✅ ไม่มีเอกสารใหม่ที่ต้อง insert")
        
        # 6. Update เอกสารที่เปลี่ยนแปลง (ถ้ามี logic เพิ่มเติม)
        if to_update:
            print(f"\n🔄 กำลัง update เอกสาร {len(to_update)} รายการ...")
            updated = self._update_documents(to_update, existing_hashes)
            self.stats["updated"] = updated
        
        # 7. Skip เอกสารที่มีอยู่แล้ว
        if to_skip:
            print(f"\n⏭️  ข้ามเอกสารที่มีอยู่แล้ว {len(to_skip)} รายการ")
            self.stats["skipped"] = len(to_skip)
        
        # 8. ลบเอกสารเก่าที่ไม่มีในชุดข้อมูลใหม่ (optional)
        if delete_missing and existing_hashes:
            print(f"\n🗑️  กำลังตรวจสอบเอกสารเก่าที่ต้องลบ...")
            deleted = self._delete_missing_documents(existing_hashes, new_hash_to_doc)
            self.stats["deleted"] = deleted
        
        # 9. แสดงสรุปผลลัพธ์
        self._print_summary()
        
        return (self.stats["inserted"], self.stats["updated"], 
                self.stats["skipped"], self.stats["deleted"])
    
    def _insert_documents(self, documents: List[Document]) -> int:
        """Insert documents ใหม่เข้า collection"""
        import uuid
        from langchain.text_splitter import CharacterTextSplitter
        
        # Split documents if needed
        splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separator="\n"
        )
        chunks = splitter.split_documents(documents)
        
        inserted_count = 0
        batch_size = 20
        documents_to_insert = []
        
        for i, chunk in enumerate(chunks):
            try:
                # Check content size (AstraDB limit ~8000 bytes)
                content_size = len(chunk.page_content.encode('utf-8'))
                if content_size > 7500:
                    print(f"⚠️  ข้าม chunk ขนาดใหญ่เกินไป: {content_size} bytes")
                    self.stats["errors"] += 1
                    continue
                
                # Generate embedding
                vector = self.embedding_model.embed_query(chunk.page_content)
                
                # Prepare document
                doc = {
                    "_id": str(uuid.uuid4()),
                    "content": chunk.page_content,
                    "$vector": vector,
                    "metadata": chunk.metadata
                }
                documents_to_insert.append(doc)
                
                # Insert in batches
                if len(documents_to_insert) >= batch_size or i == len(chunks) - 1:
                    result = self.collection.insert_many(documents_to_insert)
                    inserted_count += len(result.inserted_ids)
                    print(f"   📊 Inserted {len(result.inserted_ids)} documents ({i+1}/{len(chunks)})")
                    documents_to_insert = []
                    
            except Exception as e:
                print(f"   ❌ Error inserting chunk {i+1}: {e}")
                self.stats["errors"] += 1
        
        return inserted_count
    
    def _update_documents(self, documents: List[Document], existing_hashes: Dict) -> int:
        """
        Update เอกสารที่มีการเปลี่ยนแปลง
        Note: AstraDB ไม่รองรับ update vector โดยตรง จึงต้องลบแล้ว insert ใหม่
        """
        updated_count = 0
        
        for doc in documents:
            content_hash = doc.metadata.get("content_hash")
            if content_hash in existing_hashes:
                try:
                    # ลบเอกสารเก่า
                    old_id = existing_hashes[content_hash]["_id"]
                    self.collection.delete_one({"_id": old_id})
                    
                    # Insert เอกสารใหม่
                    self._insert_documents([doc])
                    updated_count += 1
                    
                except Exception as e:
                    print(f"   ❌ Error updating document {content_hash}: {e}")
                    self.stats["errors"] += 1
        
        return updated_count
    
    def _delete_missing_documents(self, existing_hashes: Dict, new_hash_to_doc: Dict) -> int:
        """ลบเอกสารเก่าที่ไม่มีในชุดข้อมูลใหม่"""
        deleted_count = 0
        
        for content_hash, doc_info in existing_hashes.items():
            if content_hash not in new_hash_to_doc:
                try:
                    # ลบเอกสารที่ไม่มีในข้อมูลใหม่
                    result = self.collection.delete_one({"_id": doc_info["_id"]})
                    if result.deleted_count > 0:
                        deleted_count += 1
                        print(f"   🗑️  ลบเอกสารเก่า: {content_hash[:16]}...")
                        
                except Exception as e:
                    print(f"   ❌ Error deleting document {content_hash}: {e}")
                    self.stats["errors"] += 1
        
        return deleted_count
    
    def _print_summary(self):
        """แสดงสรุปผลการดำเนินการ"""
        print("\n" + "="*60)
        print("📊 สรุปผลการ Incremental Indexing")
        print("="*60)
        print(f"✅ Insert ใหม่:        {self.stats['inserted']:>6} รายการ")
        print(f"🔄 Update:             {self.stats['updated']:>6} รายการ")
        print(f"⏭️  Skip (มีอยู่แล้ว):  {self.stats['skipped']:>6} รายการ")
        print(f"🗑️  Delete (หายไป):   {self.stats['deleted']:>6} รายการ")
        if self.stats['errors'] > 0:
            print(f"❌ Error:              {self.stats['errors']:>6} รายการ")
        print("="*60)
    
    def get_stats(self) -> Dict[str, int]:
        """ดึงสถิติการดำเนินการ"""
        return self.stats.copy()


def enable_incremental_mode(collection, embedding_model, 
                            new_documents: List[Document],
                            metadata_filter: Optional[Dict[str, Any]] = None,
                            hash_keys: Optional[List[str]] = None,
                            delete_missing: bool = False) -> Dict[str, int]:
    """
    Helper function สำหรับเปิดใช้งาน incremental indexing
    
    Args:
        collection: AstraDB collection
        embedding_model: Embedding model
        new_documents: Documents ใหม่
        metadata_filter: Filter สำหรับดึงเอกสารเดิม (เช่น {"type": "news"})
        hash_keys: Metadata keys ที่ใช้สร้าง hash (เช่น ["article_id", "slug"])
        delete_missing: ลบเอกสารเก่าที่ไม่มีในข้อมูลใหม่หรือไม่
        
    Returns:
        Dictionary ของสถิติการดำเนินการ
    """
    manager = IncrementalIndexManager(collection, embedding_model)
    manager.process_incremental_update(
        new_documents=new_documents,
        metadata_filter=metadata_filter,
        hash_keys=hash_keys,
        delete_missing=delete_missing
    )
    return manager.get_stats()


if __name__ == "__main__":
    print("🔧 Incremental Indexing Utilities")
    print("=" * 60)
    print("ไฟล์นี้ใช้สำหรับ import ไปใช้ในไฟล์อื่น")
    print("ไม่ได้ออกแบบมาให้รันโดยตรง")
    print("\nตัวอย่างการใช้งาน:")
    print("""
    from incremental_utils import enable_incremental_mode
    
    # ใช้งาน incremental indexing
    stats = enable_incremental_mode(
        collection=collection,
        embedding_model=embedding,
        new_documents=documents,
        metadata_filter={"type": "news"},
        hash_keys=["article_id", "slug"],
        delete_missing=False
    )
    
    print(f"Inserted: {stats['inserted']}")
    print(f"Skipped: {stats['skipped']}")
    """)
    print("=" * 60)

