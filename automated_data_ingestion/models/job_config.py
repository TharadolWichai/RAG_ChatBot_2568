"""
Data models สำหรับการจัดการ scraping jobs
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import json


@dataclass
class ScrapingJobConfig:
    """Configuration สำหรับ scraping job"""
    job_id: str
    name: str
    url: str  # URL หรือ API endpoint
    collection_name: str  # ชื่อ collection ใน AstraDB
    extraction_prompt: str  # Prompt สำหรับระบุว่าต้องการดึงข้อมูลส่วนไหน
    description: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Optional configurations
    use_selenium: bool = True  # ใช้ Selenium สำหรับ JavaScript rendering
    wait_time: int = 3  # รอเวลา (วินาที) สำหรับ JavaScript loading
    chunk_size: int = 500  # ขนาด chunk สำหรับ text splitting
    chunk_overlap: int = 50  # overlap สำหรับ text splitting
    
    # Metadata filters
    metadata_filter: Dict[str, Any] = field(default_factory=dict)
    hash_keys: List[str] = field(default_factory=list)
    delete_missing: bool = False
    
    # Advanced options
    custom_headers: Dict[str, str] = field(default_factory=dict)
    exclude_patterns: List[str] = field(default_factory=list)
    include_patterns: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "job_id": self.job_id,
            "name": self.name,
            "url": self.url,
            "collection_name": self.collection_name,
            "extraction_prompt": self.extraction_prompt,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "use_selenium": self.use_selenium,
            "wait_time": self.wait_time,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "metadata_filter": self.metadata_filter,
            "hash_keys": self.hash_keys,
            "delete_missing": self.delete_missing,
            "custom_headers": self.custom_headers,
            "exclude_patterns": self.exclude_patterns,
            "include_patterns": self.include_patterns
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScrapingJobConfig":
        """Create from dictionary"""
        return cls(
            job_id=data["job_id"],
            name=data["name"],
            url=data["url"],
            collection_name=data["collection_name"],
            extraction_prompt=data["extraction_prompt"],
            description=data.get("description"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            use_selenium=data.get("use_selenium", True),
            wait_time=data.get("wait_time", 3),
            chunk_size=data.get("chunk_size", 500),
            chunk_overlap=data.get("chunk_overlap", 50),
            metadata_filter=data.get("metadata_filter", {}),
            hash_keys=data.get("hash_keys", []),
            delete_missing=data.get("delete_missing", False),
            custom_headers=data.get("custom_headers", {}),
            exclude_patterns=data.get("exclude_patterns", []),
            include_patterns=data.get("include_patterns", [])
        )
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> "ScrapingJobConfig":
        """Load configuration from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class ScrapingJobResult:
    """Result from scraping job execution"""
    job_id: str
    status: str  # "success", "failed", "running"
    documents_processed: int = 0
    documents_inserted: int = 0
    documents_skipped: int = 0
    documents_updated: int = 0
    documents_deleted: int = 0
    error_message: Optional[str] = None
    execution_time: float = 0.0  # seconds
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "documents_processed": self.documents_processed,
            "documents_inserted": self.documents_inserted,
            "documents_skipped": self.documents_skipped,
            "documents_updated": self.documents_updated,
            "documents_deleted": self.documents_deleted,
            "error_message": self.error_message,
            "execution_time": self.execution_time,
            "started_at": self.started_at,
            "completed_at": self.completed_at
        }

