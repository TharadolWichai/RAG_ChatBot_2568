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
            
            # Try to verify vector search is enabled by attempting a test query
            # If vector search is not enabled, this will fail gracefully
            try:
                # Try to do a simple find with vector sort to check if vector search is enabled
                # This is a lightweight check that won't fail if collection is empty
                test_result = list(collection.find({}, limit=1))
                # If we can access the collection, try to check if it supports vector operations
                # by checking if we can use $vector in sort (this will fail if vector search is disabled)
                try:
                    # Create a dummy vector to test
                    test_vector = [0.0] * dimension
                    # Try to use vector sort (this will fail if vector search is not enabled)
                    list(collection.find({}, sort={"$vector": test_vector}, limit=1))
                    print(f"   ✅ Vector search is enabled for this collection")
                except Exception as vector_check_error:
                    error_msg = str(vector_check_error).lower()
                    if "vector" in error_msg and ("not enabled" in error_msg or "not supported" in error_msg):
                        print(f"\n❌ ERROR: Collection '{collection_name}' exists but does NOT have vector search enabled!")
                        print(f"   This collection was created as a 'Non-vector Collection'.")
                        print(f"\n💡 Solution:")
                        print(f"   1. Go to https://astra.datastax.com")
                        print(f"   2. Navigate to your database")
                        print(f"   3. Find and DELETE the collection '{collection_name}'")
                        print(f"   4. Create a NEW collection with the same name:")
                        print(f"      - Collection Name: {collection_name}")
                        print(f"      - Enable 'Vector Search' (IMPORTANT!)")
                        print(f"      - Vector Dimension: {dimension}")
                        print(f"      - Similarity Metric: cosine")
                        print(f"\n   ⚠️  You MUST enable 'Vector Search' when creating the collection!")
                        raise Exception(f"Collection '{collection_name}' does not have vector search enabled. Please delete it and create a new one with vector search enabled via AstraDB UI.")
            except Exception as e:
                # If collection is empty or other error, we'll try to proceed
                # The actual error will show when we try to insert documents
                pass
            
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
            print("⚠️  Note: AstraDB free tier may not allow creating collections via code.")
            print("   If this fails, please create the collection manually via AstraDB UI.")
            
            # Try to create collection with vector search enabled
            # Try multiple methods based on different API versions
            try:
                print(f"   Creating collection with vector search enabled...")
                # Method 1: Positional arguments (researchgroup_data.py style)
                try:
                    collection = self.database.create_collection(
                        collection_name,
                        dimension=dimension,
                        metric="cosine"
                    )
                except TypeError:
                    # Method 2: Positional arguments without metric
                    try:
                        collection = self.database.create_collection(
                            collection_name,
                            dimension=dimension
                        )
                    except TypeError:
                        # Method 3: Keyword arguments with name (setup_astradb.py style)
                        collection = self.database.create_collection(
                            name=collection_name,
                            dimension=dimension,
                            metric="cosine"
                        )
                print(f"✅ Collection created successfully: {collection_name}")
                print(f"   - Vector Dimension: {dimension}")
                print(f"   - Similarity Metric: cosine")
                
                # Verify that collection has vector search enabled
                # Try to get collection info to verify
                try:
                    collection_info = collection.find_one({})
                    print(f"   ✅ Collection verified and ready for vector search")
                except:
                    # Collection exists but may not have documents yet, which is fine
                    pass
                
                return collection
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Failed to create collection: {error_msg}")
                
                # Check if collection was created but without vector search
                if "VECTOR_SEARCH_NOT_SUPPORTED" in error_msg or "vector search is not enabled" in error_msg.lower():
                    print(f"\n⚠️  Collection '{collection_name}' may have been created but without vector search enabled.")
                    print(f"   This can happen with free tier plans.")
                    print(f"\n💡 Solution: Create collection manually via AstraDB UI with vector search:")
                    print(f"   1. Go to https://astra.datastax.com")
                    print(f"   2. Navigate to your database")
                    print(f"   3. Click 'Create Collection' or 'Add Collection'")
                    print(f"   4. Collection Name: {collection_name}")
                    print(f"   5. Enable 'Vector Search' or 'Vector Support'")
                    print(f"   6. Vector Dimension: {dimension}")
                    print(f"   7. Similarity Metric: cosine")
                    print(f"\n   After creating manually, the system will use the existing collection.")
                    
                    # Try to get the collection anyway (it might exist but without vector search)
                    try:
                        collection = self.database.get_collection(collection_name)
                        print(f"   ⚠️  Using collection '{collection_name}' but vector search may not be enabled")
                        return collection
                    except:
                        pass
                
                print(f"\n💡 Alternative: Create collection manually via AstraDB UI:")
                print(f"   1. Go to https://astra.datastax.com")
                print(f"   2. Navigate to your database")
                print(f"   3. Click 'Create Collection'")
                print(f"   4. Collection Name: {collection_name}")
                print(f"   5. Vector Dimension: {dimension}")
                print(f"   6. Similarity Metric: cosine")
                print(f"   7. Enable Vector Search")
                
                raise Exception(f"Cannot create collection '{collection_name}' with vector search. Please create it manually via AstraDB UI.")
    
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
        # Get or create collection (auto-create if not exists)
        collection = self.get_or_create_collection(collection_name, auto_create=True)
        
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
            # Try to count documents with upper_bound parameter
            try:
                count = collection.count_documents({}, upper_bound=10000)
            except TypeError:
                # Fallback: try without upper_bound (older API version)
                try:
                    count = collection.count_documents({})
                except Exception:
                    # If count fails, try to get a sample to estimate
                    sample = list(collection.find({}, limit=1))
                    count = len(sample) if sample else 0
            
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

