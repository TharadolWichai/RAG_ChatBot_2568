"""
Streamlit Dashboard สำหรับ Automated Data Ingestion
"""
import streamlit as st
import sys
import os
import json
import uuid
from datetime import datetime
from pathlib import Path

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, current_dir)

from automated_data_ingestion.core.orchestrator import DataIngestionOrchestrator
from automated_data_ingestion.models.job_config import ScrapingJobConfig, ScrapingJobResult
from automated_data_ingestion.utils.config import Config

# Page config
st.set_page_config(
    page_title="Automated Data Ingestion",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
if 'jobs_history' not in st.session_state:
    st.session_state.jobs_history = []
if 'collections' not in st.session_state:
    st.session_state.collections = []


def initialize_orchestrator():
    """Initialize orchestrator"""
    if st.session_state.orchestrator is None:
        try:
            orchestrator = DataIngestionOrchestrator()
            orchestrator.initialize_astradb()
            st.session_state.orchestrator = orchestrator
            return True
        except Exception as e:
            st.error(f"❌ Failed to initialize: {e}")
            return False
    return True


def load_jobs_history():
    """Load jobs history from file"""
    Config.ensure_jobs_dir()
    jobs_file = os.path.join(Config.JOBS_DIR, "history.json")
    if os.path.exists(jobs_file):
        try:
            with open(jobs_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []


def save_job_to_history(job_config: dict, result: dict):
    """Save job to history"""
    Config.ensure_jobs_dir()
    jobs_file = os.path.join(Config.JOBS_DIR, "history.json")
    
    history = load_jobs_history()
    history.append({
        "job_config": job_config,
        "result": result,
        "timestamp": datetime.now().isoformat()
    })
    
    with open(jobs_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def main():
    """Main dashboard"""
    st.title("🤖 Automated Data Ingestion Dashboard")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check connection
        if st.button("🔌 Test Connection"):
            if initialize_orchestrator():
                st.success("✅ Connected to AstraDB!")
                try:
                    collections = st.session_state.orchestrator.list_collections()
                    st.info(f"📦 Found {len(collections)} collections")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        st.markdown("---")
        st.header("📋 Quick Actions")
        
        if st.button("📦 List Collections"):
            if initialize_orchestrator():
                try:
                    collections = st.session_state.orchestrator.list_collections()
                    st.session_state.collections = collections
                    st.success(f"Found {len(collections)} collections")
                    for coll in collections:
                        info = st.session_state.orchestrator.get_collection_info(coll)
                        st.text(f"  • {coll} ({info.get('document_count', 0)} docs)")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if st.button("🔄 Refresh"):
            st.rerun()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📝 Create New Job", "📊 Job History", "📦 Collections"])
    
    # Tab 1: Create New Job
    with tab1:
        st.header("สร้าง Job ใหม่")
        st.markdown("กรอกข้อมูลเพื่อสร้าง scraping job ใหม่")
        
        # Batch Processing Options (outside form for immediate rendering)
        with st.expander("🔄 Batch Processing (สำหรับ List API)", expanded=False):
            st.markdown("**ใช้เมื่อต้องการดึงข้อมูลจาก List API และสร้าง jobs สำหรับแต่ละรายการ**")
            
            # Use session state for checkbox to work properly
            batch_mode_key = "batch_mode_checkbox"
            if batch_mode_key not in st.session_state:
                st.session_state[batch_mode_key] = False
            
            # Streamlit manages session state automatically when using key
            batch_mode = st.checkbox(
                "เปิดโหมด Batch Processing", 
                value=st.session_state[batch_mode_key],
                help="เปิดเพื่อใช้ batch processing",
                key=batch_mode_key
            )
            
            # Read batch_mode from session state (Streamlit updates it automatically)
            if batch_mode_key in st.session_state:
                batch_mode = st.session_state[batch_mode_key]
            
            if batch_mode:
                # Streamlit manages session state automatically when using key
                st.text_input(
                    "List API URL *",
                    value=st.session_state.get("batch_list_api", ""),
                    placeholder="https://api.computing.kku.ac.th/api/v1/page/getPageMappingBySlug/research",
                    help="API URL ที่มี list ของรายการ (เช่น list กลุ่มวิจัย)",
                    key="batch_list_api"
                )
                st.text_input(
                    "Detail URL Pattern",
                    value=st.session_state.get("detail_url_pattern", ""),
                    placeholder="https://computing.kku.ac.th/{slug}",
                    help="Pattern สำหรับสร้าง URL รายละเอียด (ใช้ {slug} เป็น placeholder)",
                    key="detail_url_pattern"
                )
                st.text_input(
                    "Detail API Pattern",
                    value=st.session_state.get("detail_api_pattern", ""),
                    placeholder="https://api.computing.kku.ac.th/api/v1/page/getPageMappingBySlug/{slug}",
                    help="Pattern สำหรับสร้าง API URL รายละเอียด (ใช้ {slug} เป็น placeholder)",
                    key="detail_api_pattern"
                )
                st.info("💡 ระบบจะดึง list จาก API, extract slugs/URLs, แล้วสร้าง jobs สำหรับแต่ละรายการอัตโนมัติ")
        
        with st.form("new_job_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                job_name = st.text_input("ชื่อ Job *", placeholder="ตัวอย่าง: Scrape Students Page")
                url = st.text_input("URL ของหน้าเว็บ (optional)", placeholder="https://computing.kku.ac.th/students", help="กรอก URL เพื่อดึง HTML จากหน้าเว็บ (ต้องมี URL หรือ API อย่างน้อย 1 อย่าง)")
                api_url = st.text_input("API Endpoint (optional)", placeholder="https://api.computing.kku.ac.th/api/v1/...", help="กรอก API URL เพื่อดึง JSON จาก API (ต้องมี URL หรือ API อย่างน้อย 1 อย่าง)")
                collection_name = st.text_input("Collection Name *", placeholder="students_embedding")
            
            with col2:
                use_selenium = st.checkbox("ใช้ Selenium (สำหรับ JavaScript)", value=True)
                wait_time = st.number_input("Wait Time (วินาที)", min_value=1, max_value=10, value=3)
                chunk_size = st.number_input("Chunk Size", min_value=100, max_value=2000, value=500)
            
            extraction_prompt = st.text_area(
                "Extraction Prompt *",
                placeholder="""ตัวอย่าง:
- ดึงข้อมูลลิงก์ทั้งหมดที่มีอยู่ในหน้า
- แยกชื่อลิงก์และ URL
- เก็บคำสำคัญที่เกี่ยวข้อง""",
                height=150
            )
            
            description = st.text_input("คำอธิบาย (optional)", placeholder="คำอธิบายเกี่ยวกับ job นี้")
            
            # Advanced options (collapsible)
            with st.expander("⚙️ Advanced Options"):
                col3, col4 = st.columns(2)
                
                with col3:
                    metadata_category = st.text_input("Metadata Category", placeholder="students")
                    hash_keys_input = st.text_input("Hash Keys (คั่นด้วย comma)", placeholder="url,title")
                
                with col4:
                    chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=200, value=50)
                    delete_missing = st.checkbox("Delete Missing Documents", value=False)
            
            submitted = st.form_submit_button("🚀 Create & Run Job", type="primary")
            
            if submitted:
                # Get batch mode and values from session state
                batch_mode = st.session_state.get("batch_mode_checkbox", False)
                batch_list_api = st.session_state.get("batch_list_api", "") if batch_mode else None
                detail_url_pattern = st.session_state.get("detail_url_pattern", "") if batch_mode else None
                detail_api_pattern = st.session_state.get("detail_api_pattern", "") if batch_mode else None
                
                # Validation
                if not job_name or not collection_name or not extraction_prompt:
                    st.error("❌ กรุณากรอกข้อมูลที่จำเป็นให้ครบถ้วน")
                elif batch_mode:
                    if not batch_list_api:
                        st.error("❌ กรุณากรอก List API URL เมื่อเปิด Batch Processing")
                    else:
                        # Batch mode - proceed with batch processing
                        url = url or "batch_mode"  # Placeholder
                elif not url and not api_url:
                    st.error("❌ กรุณากรอก URL หรือ API Endpoint อย่างน้อย 1 อย่าง")
                
                # Only proceed if validation passes
                if job_name and collection_name and extraction_prompt:
                    if (batch_mode and batch_list_api) or (not batch_mode and (url or api_url)):
                        # Create job config
                        job_id = str(uuid.uuid4())
                        
                        hash_keys = []
                        if hash_keys_input:
                            hash_keys = [k.strip() for k in hash_keys_input.split(",")]
                        
                        metadata_filter = {}
                        if metadata_category:
                            metadata_filter["category"] = metadata_category
                        
                        config = ScrapingJobConfig(
                            job_id=job_id,
                            name=job_name,
                            url=url if not batch_mode else "batch_mode",
                            api_url=api_url.strip() if api_url and api_url.strip() else None,
                            batch_mode=batch_mode if batch_mode else False,
                            batch_list_api=batch_list_api.strip() if batch_mode and batch_list_api else None,
                            detail_url_pattern=detail_url_pattern.strip() if batch_mode and detail_url_pattern else None,
                            detail_api_pattern=detail_api_pattern.strip() if batch_mode and detail_api_pattern else None,
                            collection_name=collection_name,
                            extraction_prompt=extraction_prompt,
                            description=description,
                            use_selenium=use_selenium,
                            wait_time=wait_time,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            metadata_filter=metadata_filter,
                            hash_keys=hash_keys,
                            delete_missing=delete_missing
                        )
                        
                        # Execute job
                        with st.spinner("🔄 กำลังรัน job..."):
                            if initialize_orchestrator():
                                try:
                                    result = st.session_state.orchestrator.execute_job(config)
                                    
                                    # Save to history
                                    save_job_to_history(config.to_dict(), result.to_dict())
                                    
                                    # Display results
                                    st.success("✅ Job completed!")
                                    
                                    col_result1, col_result2 = st.columns(2)
                                    
                                    with col_result1:
                                        st.metric("Status", result.status)
                                        st.metric("Documents Processed", result.documents_processed)
                                        st.metric("Documents Inserted", result.documents_inserted)
                                    
                                    with col_result2:
                                        st.metric("Execution Time", f"{result.execution_time:.2f}s")
                                        st.metric("Documents Skipped", result.documents_skipped)
                                        if result.documents_updated > 0:
                                            st.metric("Documents Updated", result.documents_updated)
                                    
                                    # Show LLM Content Preview
                                    if hasattr(result, 'llm_content_preview') and result.llm_content_preview:
                                        with st.expander("🔍 ดู JSON/Content ที่ส่งไปให้ LLM", expanded=False):
                                            st.markdown("**Content Preview ที่ส่งไปให้ LLM:**")
                                            if result.content_type == "json":
                                                try:
                                                    import json
                                                    # Try to parse as JSON for pretty display
                                                    preview_data = json.loads(result.llm_content_preview) if (result.llm_content_preview.strip().startswith('{') or result.llm_content_preview.strip().startswith('[')) else result.llm_content_preview
                                                    if isinstance(preview_data, (dict, list)):
                                                        st.json(preview_data)
                                                    else:
                                                        st.code(result.llm_content_preview[:5000], language="json")
                                                        if len(result.llm_content_preview) > 5000:
                                                            st.info(f"⚠️ Content ถูกตัดแสดง (แสดง 5000 ตัวแรกจากทั้งหมด {len(result.llm_content_preview)} ตัว)")
                                                except:
                                                    st.code(result.llm_content_preview[:5000], language="json")
                                                    if len(result.llm_content_preview) > 5000:
                                                        st.info(f"⚠️ Content ถูกตัดแสดง (แสดง 5000 ตัวแรกจากทั้งหมด {len(result.llm_content_preview)} ตัว)")
                                            else:
                                                st.code(result.llm_content_preview[:5000], language="html")
                                                if len(result.llm_content_preview) > 5000:
                                                    st.info(f"⚠️ Content ถูกตัดแสดง (แสดง 5000 ตัวแรกจากทั้งหมด {len(result.llm_content_preview)} ตัว)")
                                            
                                            if hasattr(result, 'llm_prompt_preview') and result.llm_prompt_preview:
                                                st.markdown("---")
                                                st.markdown("**Full Prompt ที่ส่งไปให้ LLM:**")
                                                st.text_area("LLM Prompt", result.llm_prompt_preview, height=300, key=f"llm_prompt_{result.job_id}", disabled=True, label_visibility="collapsed")
                                    
                                    if result.error_message:
                                        st.error(f"Error: {result.error_message}")
                                    
                                except Exception as e:
                                    st.error(f"❌ Job failed: {e}")
                                    import traceback
                                    st.code(traceback.format_exc())
                            else:
                                st.error("❌ Failed to initialize orchestrator. Please check configuration.")
    
    # Tab 2: Job History
    with tab2:
        st.header("ประวัติ Jobs")
        
        history = load_jobs_history()
        
        if not history:
            st.info("📭 ยังไม่มีประวัติ job")
        else:
            st.write(f"พบ {len(history)} jobs")
            
            for i, job_record in enumerate(reversed(history[-10:])):  # Show last 10
                with st.expander(f"Job #{len(history)-i}: {job_record['job_config'].get('name', 'Unknown')} - {job_record['result'].get('status', 'unknown')}"):
                    col_h1, col_h2 = st.columns(2)
                    
                    with col_h1:
                        st.write("**Configuration:**")
                        st.json(job_record['job_config'])
                    
                    with col_h2:
                        st.write("**Result:**")
                        st.json(job_record['result'])
    
    # Tab 3: Collections
    with tab3:
        st.header("Collections Management")
        
        if st.button("🔄 Refresh Collections"):
            if initialize_orchestrator():
                try:
                    collections = st.session_state.orchestrator.list_collections()
                    st.session_state.collections = collections
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if st.session_state.collections:
            st.write(f"**Found {len(st.session_state.collections)} collections:**")
            
            for collection_name in st.session_state.collections:
                with st.expander(f"📦 {collection_name}"):
                    if initialize_orchestrator():
                        info = st.session_state.orchestrator.get_collection_info(collection_name)
                        st.metric("Document Count", info.get("document_count", 0))
                        st.json(info)
        else:
            st.info("👆 Click 'Refresh Collections' to load collections")


if __name__ == "__main__":
    main()

