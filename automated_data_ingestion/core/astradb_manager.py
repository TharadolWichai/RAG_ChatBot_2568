"""
AstraDB manager สำหรับจัดการการเชื่อมต่อและเก็บข้อมูล
"""
import os
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from astrapy import DataAPIClient
import sys

# Import incremental utils from original data_ingestion
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'data_ingestion'))
try:
    from incremental_utils import enable_incremental_mode
except ImportError:
    print("⚠️ Could not import incremental_utils, will use basic insertion")
    enable_incremental_mode = None


class AstraDBManager:
    """จัดการการเชื่อมต่อและเก็บข้อมูลใน AstraDB"""
    
    def __init__(self, token: Optional[str] = None, endpoint: Optional[str] = None, 
                 keyspace: Optional[str] = None, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Args:
            token: AstraDB application token
            endpoint: AstraDB API endpoint
            keyspace: AstraDB keyspace name
            embedding_model_name: ชื่อ embedding model
        """
        self.token = token or os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.endpoint = endpoint or os.getenv("ASTRA_DB_API_ENDPOINT")
        self.keyspace = keyspace or os.getenv("ASTRA_DB_KEYSPACE", "default_keyspace")
        
        if not self.token or not self.endpoint:
            raise ValueError("AstraDB credentials are required. Set ASTRA_DB_APPLICATION_TOKEN and ASTRA_DB_API_ENDPOINT")
        
        # Initialize client
        self.client = DataAPIClient(token=self.token)
        self.database = self.client.get_database_by_api_endpoint(self.endpoint)
        
        # Initialize embedding model
        print("🧠 Initializing embedding model...")
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        print(f"✅ Connected to AstraDB: {self.endpoint}")
        print(f"🏠 Using keyspace: {self.keyspace}")
    
    def get_or_create_collection(self, collection_name: str, dimension: int = 384, 
                                  auto_create: bool = False) -> Any:
        """
        Get existing collection or create new one
        
        Args:
            collection_name: ชื่อ collection
            dimension: Vector dimension (default 384 สำหรับ all-MiniLM-L6-v2)
            auto_create: ถ้า True จะพยายามสร้าง collection อัตโนมัติ (default: False)
            
        Returns:
            Collection object
        """
        existing_collections = list(self.database.list_collection_names())
        
        if collection_name in existing_collections:
            collection = self.database.get_collection(collection_name)
            print(f"📂 Using existing collection: {collection_name}")
            return collection
        elif not auto_create:
            # Don't auto-create, just raise error with instructions
            print(f"❌ Collection '{collection_name}' not found!")
            print("\n💡 Please create the collection manually via AstraDB UI:")
            print(f"   1. Go to AstraDB Console")
            print(f"   2. Navigate to your database")
            print(f"   3. Click 'Create Collection'")
            print(f"   4. Collection Name: {collection_name}")
            print(f"   5. Vector Dimension: {dimension}")
            print(f"   6. Similarity Metric: cosine")
            raise Exception(f"Collection '{collection_name}' does not exist. Please create it via AstraDB UI.")
        else:
            print(f"📦 Creating new collection: {collection_name} (dimension={dimension})...")
            print("⚠️  Note: AstraDB may not allow creating collections via code in some plans.")
            print("   If this fails, please create the collection manually via AstraDB UI.")
            
            # Try multiple methods based on different API versions
            methods_to_try = [
                # Method 1: Positional arguments with dimension only (researchgroup_data.py style)
                lambda: self.database.create_collection(collection_name, dimension=dimension),
                
                # Method 2: Positional arguments with dimension and metric (test_incremental.py style)
                lambda: self.database.create_collection(collection_name, dimension=dimension, metric="cosine"),
                
                # Method 3: Keyword arguments with name (setup_astradb.py style)
                lambda: self.database.create_collection(name=collection_name, dimension=dimension, metric="cosine"),
                
                # Method 4: Keyword arguments without metric
                lambda: self.database.create_collection(name=collection_name, dimension=dimension),
                
                # Method 5: Just name (if dimension is set via other means)
                lambda: self.database.create_collection(name=collection_name),
            ]
            
            last_error = None
            for i, method in enumerate(methods_to_try, 1):
                try:
                    print(f"   Trying method {i}...")
                    collection = method()
                    print(f"✅ Collection created: {collection_name}")
                    return collection
                except TypeError as e:
                    last_error = e
                    continue
                except Exception as e:
                    # If it's not a TypeError, it might be a different issue (like permission)
                    last_error = e
                    break
            
            # If all methods failed
            print(f"❌ Failed to create collection after trying all methods")
            print(f"   Last error: {last_error}")
            print("\n💡 Solution: Create collection manually via AstraDB UI:")
            print(f"   1. Go to AstraDB Console")
            print(f"   2. Navigate to your database")
            print(f"   3. Click 'Create Collection'")
            print(f"   4. Collection Name: {collection_name}")
            print(f"   5. Vector Dimension: {dimension}")
            print(f"   6. Similarity Metric: cosine")
            print("\n   Or, collection may already exist. The system will use existing collection.")
            
            # Don't raise error - let user create manually or use existing
            raise Exception(f"Cannot create collection '{collection_name}' programmatically. Please create it manually via AstraDB UI.")
    
    def insert_documents(self, collection_name: str, documents: List[Document],
                        metadata_filter: Optional[Dict[str, Any]] = None,
                        hash_keys: Optional[List[str]] = None,
                        delete_missing: bool = False,
                        chunk_size: int = 500,
                        chunk_overlap: int = 50) -> Dict[str, int]:
        """
        Insert documents into collection (with optional incremental indexing)
        
        Args:
            collection_name: ชื่อ collection
            documents: List of Document objects
            metadata_filter: Filter สำหรับ incremental indexing
            hash_keys: Keys สำหรับสร้าง hash
            delete_missing: ลบเอกสารเก่าที่ไม่มีในข้อมูลใหม่หรือไม่
            chunk_size: ขนาด chunk
            chunk_overlap: overlap สำหรับ chunking
            
        Returns:
            Dictionary with stats (inserted, skipped, updated, deleted)
        """
        # Get or create collection
        collection = self.get_or_create_collection(collection_name)
        
        # Split documents if needed
        if len(documents) > 0 and len(documents[0].page_content) > chunk_size:
            print(f"📄 Splitting documents (chunk_size={chunk_size}, overlap={chunk_overlap})...")
            splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="\n"
            )
            documents = splitter.split_documents(documents)
            print(f"   Created {len(documents)} chunks")
        
        # Use incremental indexing if available
        if enable_incremental_mode and (metadata_filter or hash_keys):
            print("\n🚀 Starting incremental indexing...")
            try:
                stats = enable_incremental_mode(
                    collection=collection,
                    embedding_model=self.embedding_model,
                    new_documents=documents,
                    metadata_filter=metadata_filter,
                    hash_keys=hash_keys,
                    delete_missing=delete_missing
                )
                return stats
            except Exception as e:
                print(f"⚠️ Incremental indexing failed: {e}")
                print("   Falling back to basic insertion...")
        
        # Basic insertion (fallback)
        return self._basic_insert(collection, documents)
    
    def _basic_insert(self, collection: Any, documents: List[Document]) -> Dict[str, int]:
        """Basic document insertion (without incremental mode)"""
        import uuid
        
        inserted_count = 0
        batch_size = 20
        documents_to_insert = []
        
        print(f"\n📥 Inserting {len(documents)} documents...")
        
        for i, doc in enumerate(documents):
            try:
                # Check content size
                content_size = len(doc.page_content.encode('utf-8'))
                if content_size > 7500:
                    print(f"⚠️  Skipping large document: {content_size} bytes")
                    continue
                
                # Generate embedding
                vector = self.embedding_model.embed_query(doc.page_content)
                
                # Prepare document
                doc_dict = {
                    "_id": str(uuid.uuid4()),
                    "content": doc.page_content,
                    "$vector": vector,
                    "metadata": doc.metadata
                }
                documents_to_insert.append(doc_dict)
                
                # Insert in batches
                if len(documents_to_insert) >= batch_size or i == len(documents) - 1:
                    result = collection.insert_many(documents_to_insert)
                    inserted_count += len(result.inserted_ids)
                    print(f"   📊 Inserted {len(result.inserted_ids)} documents ({i+1}/{len(documents)})")
                    documents_to_insert = []
                    
            except Exception as e:
                print(f"   ❌ Error inserting document {i+1}: {e}")
        
        return {
            "inserted": inserted_count,
            "skipped": 0,
            "updated": 0,
            "deleted": 0
        }
    
    def list_collections(self) -> List[str]:
        """List all collections in the database"""
        return list(self.database.list_collection_names())
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection"""
        try:
            collection = self.database.get_collection(collection_name)
            count = collection.count_documents({})
            return {
                "name": collection_name,
                "document_count": count,
                "exists": True
            }
        except Exception as e:
            return {
                "name": collection_name,
                "document_count": 0,
                "exists": False,
                "error": str(e)
            }
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection (use with caution!)"""
        try:
            self.database.delete_collection(collection_name)
            print(f"🗑️  Collection deleted: {collection_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to delete collection: {e}")
            return False

