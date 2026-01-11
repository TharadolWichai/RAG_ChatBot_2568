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
from .batch_processor import BatchProcessor
from ..models.job_config import ScrapingJobConfig, ScrapingJobResult
import uuid


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
        # Check if batch mode is enabled
        if config.batch_mode:
            print(f"\n{'='*60}")
            print(f"🔄 Batch Mode Enabled")
            print(f"{'='*60}\n")
            return self.execute_batch_job(config)
        
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
            html_content = None
            json_content = None
            
            # Fetch HTML from URL if provided (optional)
            # Skip if URL is "batch_mode" placeholder
            if config.url and config.url != "batch_mode":
                print(f"   🌐 Fetching HTML from URL: {config.url}")
                try:
                    soup = self.scraper.scrape(config.url)
                    html_content = str(soup)
                    print(f"   ✅ Fetched HTML ({len(html_content)} characters)")
                except Exception as e:
                    print(f"   ⚠️ Failed to fetch HTML: {e}")
                    if not config.api_url:
                        raise Exception(f"Failed to fetch HTML from URL and no API URL provided: {e}")
                    print(f"   ⚠️ Continuing with API only...")
            
            # Fetch JSON from API if provided (optional)
            if config.api_url:
                print(f"   🔌 Fetching JSON from API: {config.api_url}")
                try:
                    api_data = self.scraper.fetch_api(config.api_url)
                    import json
                    json_content = json.dumps(api_data, ensure_ascii=False, indent=2)
                    print(f"   ✅ Fetched JSON ({len(json_content)} characters)")
                    
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
                except Exception as e:
                    print(f"   ⚠️ Failed to fetch API: {e}")
                    if not html_content:
                        raise Exception(f"Failed to fetch API and no HTML content available: {e}")
                    print(f"   ⚠️ Continuing with HTML only...")
            
            # Validate that we have at least one content source
            if not html_content and not json_content:
                raise Exception("No content fetched. Please provide at least URL or API endpoint.")
            
            # Combine content for LLM
            if html_content and json_content:
                # If both HTML and JSON exist, combine them
                url_label = config.url if config.url else "URL"
                api_label = config.api_url if config.api_url else "API"
                content = f"=== HTML CONTENT FROM {url_label} ===\n{html_content}\n\n=== JSON CONTENT FROM {api_label} ===\n{json_content}"
                content_type = "combined"
                print(f"   ✅ Combined HTML + JSON ({len(html_content)} + {len(json_content)} = {len(content)} characters)")
            elif json_content:
                # Only JSON
                content = json_content
                content_type = "json"
                print(f"   ✅ Using JSON only ({len(content)} characters)")
            else:
                # Only HTML
                content = html_content
                content_type = "html"
                print(f"   ✅ Using HTML only ({len(content)} characters)")
            
            # Store content for preview (if needed by dashboard)
            result.raw_content = content
            result.content_type = content_type
            
            # Prepare LLM preview (for dashboard display)
            if self.extractor.use_openai and self.extractor.llm:
                try:
                    llm_preview, llm_prompt = self.extractor.prepare_llm_content(
                        content=content,
                        prompt=config.extraction_prompt,
                        content_type=content_type,
                        html_content=html_content,
                        json_content=json_content
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
            
            # For detail pages (single item detail), merge all documents into one
            # This ensures all information about one item (e.g., research group) stays together
            # Similar to researchgroup_data.py which creates one document per URL
            # Always merge for batch processing (each detail is one item)
            # Also merge if prompt suggests detail page or URL pattern suggests detail
            is_batch_detail = config.batch_mode or (
                config.api_url and "/getPageMappingBySlug/" in config.api_url
            )
            is_detail_prompt = (
                "รายละเอียด" in config.extraction_prompt.lower() or
                "detail" in config.extraction_prompt.lower() or
                "กลุ่มวิจัย" in config.extraction_prompt.lower() or
                "research group" in config.extraction_prompt.lower()
            )
            is_detail_url = (
                config.url and 
                config.url != "batch_mode" and 
                not is_api_url(config.url) and
                ("/mlislab" in config.url or "/aiii" in config.url or "/agtlab" in config.url or
                 "/asclab" in config.url or "/nlsplab" in config.url or "/aidalab" in config.url or
                 "/i-serg" in config.url or "/hardware-human" in config.url)
            )
            
            is_detail_page = is_batch_detail or is_detail_prompt or is_detail_url
            
            if is_detail_page and len(documents) > 1:
                print(f"   🔗 Detected detail page - merging {len(documents)} documents into one...")
                print(f"      Reason: batch_detail={is_batch_detail}, detail_prompt={is_detail_prompt}, detail_url={is_detail_url}")
                # Merge all documents into one
                merged_content = "\n\n".join([doc.page_content for doc in documents])
                # Merge metadata (keep common fields, combine unique ones)
                merged_metadata = {}
                if documents:
                    # Start with first document's metadata
                    merged_metadata = documents[0].metadata.copy()
                    # Add any unique fields from other documents
                    for doc in documents[1:]:
                        for key, value in doc.metadata.items():
                            if key not in merged_metadata:
                                merged_metadata[key] = value
                            elif merged_metadata[key] != value:
                                # If different values, combine them
                                if isinstance(merged_metadata[key], list):
                                    if value not in merged_metadata[key]:
                                        merged_metadata[key].append(value)
                                elif isinstance(merged_metadata[key], str) and isinstance(value, str):
                                    merged_metadata[key] = f"{merged_metadata[key]}, {value}"
                
                documents = [Document(page_content=merged_content, metadata=merged_metadata)]
                print(f"   ✅ Merged into 1 document ({len(merged_content)} characters)")
            elif is_detail_page and len(documents) == 1:
                print(f"   ℹ️  Detail page detected, but only 1 document - no merge needed")
            
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
    
    def execute_batch_job(self, config: ScrapingJobConfig) -> ScrapingJobResult:
        """
        Execute batch job - extract list from API and create jobs for each item
        
        Args:
            config: ScrapingJobConfig with batch_mode=True
            
        Returns:
            ScrapingJobResult with summary of all batch jobs
        """
        batch_result = ScrapingJobResult(
            job_id=config.job_id,
            status="running",
            started_at=datetime.now().isoformat()
        )
        
        batch_start_time = time.time()
        
        try:
            # Initialize batch processor
            batch_processor = BatchProcessor(use_openai=True)
            
            # Process list API
            print(f"\n📋 Step 1: Processing batch list...")
            detail_configs = batch_processor.process_batch_list(
                list_api_url=config.batch_list_api,
                detail_url_pattern=config.detail_url_pattern,
                detail_api_pattern=config.detail_api_pattern
            )
            
            if not detail_configs:
                raise Exception("No items found in list API")
            
            print(f"\n📋 Step 2: Creating {len(detail_configs)} jobs...")
            
            # Initialize orchestrator for individual jobs
            if not self.astradb_manager:
                self.initialize_astradb()
            
            # Execute jobs for each detail
            total_processed = 0
            total_inserted = 0
            total_skipped = 0
            total_updated = 0
            failed_count = 0
            
            for i, detail_config in enumerate(detail_configs, 1):
                slug = detail_config.get("slug", f"item_{i}")
                detail_url = detail_config.get("url")
                detail_api = detail_config.get("api_url")
                
                print(f"\n{'='*60}")
                print(f"🔄 Processing item {i}/{len(detail_configs)}: {slug}")
                print(f"{'='*60}")
                
                # Create job config for this detail
                detail_job_config = ScrapingJobConfig(
                    job_id=str(uuid.uuid4()),
                    name=f"{config.name} - {slug}",
                    url=detail_url or "",
                    api_url=detail_api,
                    collection_name=config.collection_name,
                    extraction_prompt=config.extraction_prompt,
                    description=f"Batch job item: {slug}",
                    use_selenium=config.use_selenium,
                    wait_time=config.wait_time,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                    metadata_filter=config.metadata_filter,
                    hash_keys=config.hash_keys,
                    delete_missing=config.delete_missing,
                    custom_headers=config.custom_headers
                )
                
                # Execute job
                try:
                    detail_result = self.execute_job(detail_job_config)
                    
                    total_processed += detail_result.documents_processed
                    total_inserted += detail_result.documents_inserted
                    total_skipped += detail_result.documents_skipped
                    total_updated += detail_result.documents_updated
                    
                    if detail_result.status != "success":
                        failed_count += 1
                        print(f"   ⚠️ Job failed: {detail_result.error_message}")
                    
                except Exception as e:
                    failed_count += 1
                    print(f"   ❌ Error processing {slug}: {e}")
            
            # Set batch result
            batch_result.status = "success" if failed_count == 0 else "partial_success"
            batch_result.documents_processed = total_processed
            batch_result.documents_inserted = total_inserted
            batch_result.documents_skipped = total_skipped
            batch_result.documents_updated = total_updated
            batch_result.execution_time = time.time() - batch_start_time
            batch_result.completed_at = datetime.now().isoformat()
            
            if failed_count > 0:
                batch_result.error_message = f"{failed_count} jobs failed out of {len(detail_configs)}"
            
            print(f"\n{'='*60}")
            print(f"✅ Batch processing completed!")
            print(f"   - Total items: {len(detail_configs)}")
            print(f"   - Successful: {len(detail_configs) - failed_count}")
            print(f"   - Failed: {failed_count}")
            print(f"   - Documents processed: {total_processed}")
            print(f"   - Documents inserted: {total_inserted}")
            print(f"{'='*60}\n")
            
        except Exception as e:
            batch_result.status = "failed"
            batch_result.error_message = str(e)
            batch_result.completed_at = datetime.now().isoformat()
            print(f"\n❌ Batch job failed: {e}")
        
        return batch_result
    
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

