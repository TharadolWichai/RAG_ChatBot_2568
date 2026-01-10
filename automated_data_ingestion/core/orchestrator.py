"""
Main orchestrator ที่รวม scraping, extraction และ database storage เข้าด้วยกัน
"""
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain.schema import Document

from .scraper import WebScraper, is_api_url
from .llm_extractor import LLMExtractor
from .astradb_manager import AstraDBManager
from ..models.job_config import ScrapingJobConfig, ScrapingJobResult


class DataIngestionOrchestrator:
    """Orchestrator สำหรับจัดการกระบวนการดึงข้อมูลทั้งหมด"""
    
    def __init__(self):
        """Initialize orchestrator"""
        self.astradb_manager = None
        self.scraper = None
        self.extractor = None
    
    def initialize_astradb(self, token: Optional[str] = None, endpoint: Optional[str] = None, 
                          keyspace: Optional[str] = None):
        """Initialize AstraDB manager"""
        self.astradb_manager = AstraDBManager(token=token, endpoint=endpoint, keyspace=keyspace)
        return self.astradb_manager
    
    def execute_job(self, config: ScrapingJobConfig) -> ScrapingJobResult:
        """
        Execute scraping job ตาม configuration
        
        Args:
            config: ScrapingJobConfig object
            
        Returns:
            ScrapingJobResult object
        """
        result = ScrapingJobResult(
            job_id=config.job_id,
            status="running",
            started_at=datetime.now().isoformat()
        )
        
        start_time = time.time()
        
        try:
            print(f"\n{'='*60}")
            print(f"🚀 Starting job: {config.name}")
            print(f"{'='*60}")
            print(f"📋 Job ID: {config.job_id}")
            print(f"🌐 URL: {config.url}")
            print(f"📦 Collection: {config.collection_name}")
            print(f"💬 Extraction Prompt: {config.extraction_prompt[:100]}...")
            print(f"{'='*60}\n")
            
            # 1. Initialize components
            if not self.astradb_manager:
                self.initialize_astradb()
            
            # Initialize scraper
            self.scraper = WebScraper(
                use_selenium=config.use_selenium,
                wait_time=config.wait_time,
                custom_headers=config.custom_headers
            )
            
            # Initialize extractor
            self.extractor = LLMExtractor(use_openai=True)
            
            # 2. Scrape or fetch data
            print("📥 Step 1: Fetching data...")
            content = None
            content_type = "html"
            
            if is_api_url(config.url):
                # Fetch from API
                content_type = "json"
                api_data = self.scraper.fetch_api(config.url)
                # Convert to JSON string for LLM processing
                import json
                content = json.dumps(api_data, ensure_ascii=False, indent=2)
                
                # Log structure info for debugging
                if isinstance(api_data, dict):
                    if "data" in api_data and isinstance(api_data["data"], dict):
                        if "items" in api_data["data"] and isinstance(api_data["data"]["items"], list):
                            items_count = len(api_data["data"]["items"])
                            print(f"   ✅ Found nested structure: data.data.items with {items_count} items")
                        elif "pageComponent_Mapping" in api_data["data"]:
                            comp_count = len(api_data["data"]["pageComponent_Mapping"]) if isinstance(api_data["data"]["pageComponent_Mapping"], list) else 0
                            if comp_count > 0:
                                print(f"   ✅ Found pageComponent_Mapping with {comp_count} components")
            else:
                # Scrape webpage
                soup = self.scraper.scrape(config.url)
                content = str(soup)
                content_type = "html"
            
            if not content:
                raise Exception("No content fetched from URL")
            
            print(f"   ✅ Fetched {len(content)} characters")
            
            # Store content for preview (if needed by dashboard)
            result.raw_content = content
            result.content_type = content_type
            
            # Prepare LLM preview (for dashboard display)
            if self.extractor.use_openai and self.extractor.llm:
                try:
                    llm_preview, llm_prompt = self.extractor.prepare_llm_content(
                        content=content,
                        prompt=config.extraction_prompt,
                        content_type=content_type
                    )
                    result.llm_content_preview = llm_preview
                    result.llm_prompt_preview = llm_prompt
                except Exception as e:
                    print(f"   ⚠️ Could not prepare LLM preview: {e}")
                    result.llm_content_preview = None
                    result.llm_prompt_preview = None
            
            # 3. Extract data using LLM (primary) or rule-based (fallback)
            print("\n🔍 Step 2: Extracting data...")
            print("   🤖 Using LLM-based extraction (primary method)...")
            print(f"   💬 Prompt: {config.extraction_prompt[:100]}...")
            
            # Always try LLM first for flexibility (no bypass to rule-based)
            documents = self.extractor.extract(
                content=content,
                prompt=config.extraction_prompt,
                content_type=content_type
            )
            print(f"   📊 Extraction result: {len(documents)} documents")
            
            if not documents:
                raise Exception("No documents extracted")
            
            print(f"   ✅ Extracted {len(documents)} documents")
            
            # Add metadata to documents
            for doc in documents:
                doc.metadata.update({
                    "job_id": config.job_id,
                    "job_name": config.name,
                    "source_url": config.url,
                    "collection": config.collection_name
                })
                # Merge with custom metadata filter
                if config.metadata_filter:
                    doc.metadata.update(config.metadata_filter)
            
            result.documents_processed = len(documents)
            
            # Auto-detect hash_keys if not provided
            hash_keys = config.hash_keys if config.hash_keys else []
            
            # If no hash_keys provided, try to detect from document metadata
            if not hash_keys and documents:
                # Check what fields are available in first document
                sample_metadata = documents[0].metadata
                # Common hash keys for different data types
                possible_hash_keys = []
                
                # For allpeople/faculty data (based on allpeople_data.py pattern)
                if "slug" in sample_metadata:
                    possible_hash_keys.append("slug")
                if "name" in sample_metadata:
                    possible_hash_keys.append("name")
                if "email" in sample_metadata:
                    possible_hash_keys.append("email")
                if "position" in sample_metadata and "name" in sample_metadata:
                    # Use name + position for unique identification (like allpeople_data.py)
                    possible_hash_keys = ["name", "position"]
                
                # For news/article data
                if "article_id" in sample_metadata:
                    possible_hash_keys.append("article_id")
                if "url" in sample_metadata:
                    possible_hash_keys.append("url")
                
                # Use first few keys found
                hash_keys = possible_hash_keys[:3]  # Limit to 3 keys
                
                if hash_keys:
                    print(f"   🔑 Auto-detected hash_keys: {hash_keys}")
                    print(f"      (This enables incremental indexing to skip duplicates)")
            
            # Auto-set metadata_filter if not provided but we have category in metadata
            metadata_filter = config.metadata_filter if config.metadata_filter else {}
            if not metadata_filter and documents:
                sample_metadata = documents[0].metadata
                # Check for common category fields
                if "type" in sample_metadata:
                    metadata_filter["type"] = sample_metadata["type"]
                    print(f"   🔍 Auto-detected metadata_filter: {metadata_filter}")
            
            # 4. Store in AstraDB
            print("\n💾 Step 3: Storing in AstraDB...")
            stats = self.astradb_manager.insert_documents(
                collection_name=config.collection_name,
                documents=documents,
                metadata_filter=metadata_filter if metadata_filter else None,
                hash_keys=hash_keys if hash_keys else None,
                delete_missing=config.delete_missing,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
            
            result.documents_inserted = stats.get("inserted", 0)
            result.documents_skipped = stats.get("skipped", 0)
            result.documents_updated = stats.get("updated", 0)
            result.documents_deleted = stats.get("deleted", 0)
            
            print(f"   ✅ Inserted: {result.documents_inserted}")
            print(f"   ⏭️  Skipped: {result.documents_skipped}")
            if result.documents_updated > 0:
                print(f"   🔄 Updated: {result.documents_updated}")
            if result.documents_deleted > 0:
                print(f"   🗑️  Deleted: {result.documents_deleted}")
            
            # 5. Success
            result.status = "success"
            result.completed_at = datetime.now().isoformat()
            result.execution_time = time.time() - start_time
            
            print(f"\n✅ Job completed successfully in {result.execution_time:.2f} seconds!")
            
        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
            result.completed_at = datetime.now().isoformat()
            result.execution_time = time.time() - start_time
            
            print(f"\n❌ Job failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            if self.scraper:
                self.scraper.close()
                self.scraper = None
        
        return result
    
    def list_collections(self) -> List[str]:
        """List all collections"""
        if not self.astradb_manager:
            self.initialize_astradb()
        return self.astradb_manager.list_collections()
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get collection information"""
        if not self.astradb_manager:
            self.initialize_astradb()
        return self.astradb_manager.get_collection_info(collection_name)

