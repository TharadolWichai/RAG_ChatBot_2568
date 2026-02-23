"""
ตัวอย่างการใช้งาน Automated Data Ingestion System

วิธีรัน:
    python automated_data_ingestion/example_usage.py
"""
import sys
import os
from dotenv import load_dotenv

# Get the root project directory (parent of automated_data_ingestion)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file in project root
env_path = os.path.join(project_root, '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
    print(f"✅ Loaded .env from: {env_path}\n")
else:
    print(f"⚠️ .env file not found at: {env_path}\n")
    load_dotenv()  # Try to load from current directory

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from automated_data_ingestion.core.orchestrator import DataIngestionOrchestrator
from automated_data_ingestion.models.job_config import ScrapingJobConfig


def example_basic_scraping():
    """ตัวอย่างการ scrape หน้าเว็บพื้นฐาน"""
    print("="*60)
    print("ตัวอย่าง 1: Basic Web Scraping")
    print("="*60)
    
    # Initialize orchestrator
    orchestrator = DataIngestionOrchestrator()
    orchestrator.initialize_astradb()
    
    # Create job config
    config = ScrapingJobConfig(
        job_id="example-001",
        name="Scrape Students Page Example",
        url="https://computing.kku.ac.th/students",
        collection_name="example_students_embedding",
        extraction_prompt="""ดึงข้อมูลลิงก์ทั้งหมดที่มีอยู่ในหน้าเว็บ
สำหรับแต่ละลิงก์ให้เก็บ:
- ชื่อลิงก์ (text)
- URL
- คำสำคัญที่เกี่ยวข้อง""",
        description="ตัวอย่างการ scrape หน้าเว็บ students",
        use_selenium=True,
        wait_time=3,
        chunk_size=500,
        chunk_overlap=50,
        metadata_filter={"category": "students", "source": "example"},
        hash_keys=["url"]
    )
    
    # Execute job
    result = orchestrator.execute_job(config)
    
    # Print results
    print(f"\n✅ Job completed!")
    print(f"   Status: {result.status}")
    print(f"   Documents processed: {result.documents_processed}")
    print(f"   Documents inserted: {result.documents_inserted}")
    print(f"   Documents skipped: {result.documents_skipped}")
    print(f"   Execution time: {result.execution_time:.2f}s")


def example_api_scraping():
    """ตัวอย่างการดึงข้อมูลจาก API"""
    print("\n" + "="*60)
    print("ตัวอย่าง 2: API Scraping")
    print("="*60)
    
    # Initialize orchestrator
    orchestrator = DataIngestionOrchestrator()
    orchestrator.initialize_astradb()
    
    # Create job config for API
    config = ScrapingJobConfig(
        job_id="example-002",
        name="API Example",
        url="https://api.example.com/data",  # ตัวอย่าง URL
        collection_name="example_api_embedding",
        extraction_prompt="""ดึงข้อมูลจาก JSON response
สำหรับแต่ละ item ให้เก็บ:
- title
- description
- url (ถ้ามี)
- published_date (ถ้ามี)""",
        description="ตัวอย่างการดึงข้อมูลจาก API",
        use_selenium=False,  # API ไม่ต้องใช้ Selenium
        metadata_filter={"source": "api", "type": "example"}
    )
    
    print("⚠️  Note: This is an example. Replace URL with actual API endpoint.")
    # Uncomment to run:
    # result = orchestrator.execute_job(config)
    # print(f"Status: {result.status}")


def example_list_collections():
    """ตัวอย่างการดู collections"""
    print("\n" + "="*60)
    print("ตัวอย่าง 3: List Collections")
    print("="*60)
    
    orchestrator = DataIngestionOrchestrator()
    orchestrator.initialize_astradb()
    
    collections = orchestrator.list_collections()
    print(f"\n📦 Found {len(collections)} collections:")
    
    for coll_name in collections:
        info = orchestrator.get_collection_info(coll_name)
        print(f"   • {coll_name}: {info.get('document_count', 0)} documents")


if __name__ == "__main__":
    print("\n🤖 Automated Data Ingestion - Example Usage\n")
    
    # Run examples
    try:
        example_basic_scraping()
        # example_api_scraping()  # Uncomment if you have API endpoint
        example_list_collections()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Examples completed!")

