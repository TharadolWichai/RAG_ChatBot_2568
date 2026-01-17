"""
Configuration management for automated data ingestion
"""
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


class Config:
    """Configuration class for managing environment variables and settings"""
    
    # AstraDB Configuration
    ASTRA_DB_TOKEN: Optional[str] = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    ASTRA_DB_ENDPOINT: Optional[str] = os.getenv("ASTRA_DB_API_ENDPOINT")
    ASTRA_DB_KEYSPACE: str = os.getenv("ASTRA_DB_KEYSPACE", "default_keyspace")
    
    # OpenAI/LLM Configuration (สำหรับใช้ LLM ในการ extract ข้อมูล)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Embedding Model
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # Default settings
    DEFAULT_CHUNK_SIZE: int = 500
    DEFAULT_CHUNK_OVERLAP: int = 50
    DEFAULT_WAIT_TIME: int = 3
    
    # Jobs storage directory
    JOBS_DIR: str = "automated_data_ingestion/jobs"
    JOBS_CONFIG_FILE: str = "automated_data_ingestion/jobs/jobs_config.json"
    
    @classmethod
    def validate(cls) -> tuple[bool, list[str]]:
        """
        Validate configuration
        Returns: (is_valid, list_of_errors)
        """
        errors = []
        
        if not cls.ASTRA_DB_TOKEN:
            errors.append("ASTRA_DB_APPLICATION_TOKEN is not set in .env file")
        
        if not cls.ASTRA_DB_ENDPOINT:
            errors.append("ASTRA_DB_API_ENDPOINT is not set in .env file")
        
        return len(errors) == 0, errors
    
    @classmethod
    def ensure_jobs_dir(cls):
        """Ensure jobs directory exists"""
        os.makedirs(cls.JOBS_DIR, exist_ok=True)
        
        # Create jobs config file if not exists
        if not os.path.exists(cls.JOBS_CONFIG_FILE):
            import json
            with open(cls.JOBS_CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump({"jobs": []}, f, ensure_ascii=False, indent=2)

