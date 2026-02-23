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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, current_dir)

from automated_data_ingestion.core.orchestrator import DataIngestionOrchestrator
from automated_data_ingestion.models.job_config import ScrapingJobConfig, ScrapingJobResult
from automated_data_ingestion.utils.config import Config
from automated_data_ingestion.utils.model_manager import get_model_manager

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
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = os.getenv("OPENAI_MODEL", "gemini-2.5-flash-lite")
if 'available_models' not in st.session_state:
    st.session_state.available_models = []
if 'available_collections' not in st.session_state:
    st.session_state.available_collections = []
if 'selected_collection' not in st.session_state:
    st.session_state.selected_collection = None
if 'create_new_collection' not in st.session_state:
    st.session_state.create_new_collection = False


def initialize_orchestrator():
    """Initialize orchestrator"""
    if st.session_state.orchestrator is None:
        try:
            orchestrator = DataIngestionOrchestrator()
            orchestrator.initialize_astradb()
            st.session_state.orchestrator = orchestrator
            
            # Load available collections
            try:
                collections = orchestrator.list_collections()
                st.session_state.available_collections = collections
                print(f"✅ Loaded {len(collections)} collections")
            except Exception as e:
                print(f"⚠️ Could not load collections: {e}")
            
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
        
        # Model Selection Section
        with st.expander("🤖 Model Selection", expanded=True):
            st.markdown("**เลือกโมเดล LLM สำหรับ Extraction:**")
            
            # Fetch models button
            col_model1, col_model2 = st.columns([2, 1])
            with col_model1:
                if st.button("🔄 Refresh Models", use_container_width=True):
                    with st.spinner("กำลังดึงรายการโมเดล..."):
                        model_manager = get_model_manager()
                        # Force refresh (don't use cache)
                        models = model_manager.fetch_available_models()
                        if models:
                            st.session_state.available_models = models
                            st.success(f"✅ พบ {len(models)} โมเดล")
                        else:
                            st.warning("⚠️ ใช้โมเดล default")
                            st.session_state.available_models = model_manager.default_models
            
            with col_model2:
                # Show cached status
                if 'models_cache' in st.session_state:
                    st.caption("📦 Cached")
                else:
                    st.caption("🔄 Not cached")
            
            # Load models if not already loaded
            if not st.session_state.available_models:
                model_manager = get_model_manager()
                st.session_state.available_models = model_manager.get_models_with_fallback()
            
            # Categorize models
            model_manager = get_model_manager()
            categorized = model_manager.categorize_models(st.session_state.available_models)
            
            # Create selectbox with categories
            all_models_flat = []
            model_display_names = {}
            
            for category, models in categorized.items():
                all_models_flat.append(f"─── {category} ───")
                for model in models:
                    all_models_flat.append(model)
                    # Add display name with info
                    info = model_manager.get_model_recommendations().get(model)
                    if info:
                        model_display_names[model] = f"{model} ({info['speed']} {info['quality']})"
                    else:
                        model_display_names[model] = model
            
            # Find current selection index
            try:
                current_index = all_models_flat.index(st.session_state.selected_model)
            except ValueError:
                # If selected model not in list, add it
                all_models_flat.insert(0, st.session_state.selected_model)
                current_index = 0
            
            # Model selector
            selected_display = st.selectbox(
                "เลือกโมเดล:",
                options=all_models_flat,
                index=current_index,
                format_func=lambda x: x if x.startswith("───") else model_display_names.get(x, x),
                key="model_selector"
            )
            
            # Update selected model (skip category headers)
            if selected_display and not selected_display.startswith("───"):
                st.session_state.selected_model = selected_display
                # Update environment variable for this session
                os.environ["OPENAI_MODEL"] = selected_display
            
            # Show selected model info
            if st.session_state.selected_model:
                st.success(f"✅ ใช้โมเดล: **{st.session_state.selected_model}**")
                
                # Show model info if available
                recommendations = model_manager.get_model_recommendations()
                if st.session_state.selected_model in recommendations:
                    info = recommendations[st.session_state.selected_model]
                    st.caption(f"📊 {info['description']}")
        
        st.markdown("---")
        
        # Collection Selection Section
        with st.expander("📦 Collection Selection", expanded=False):
            st.markdown("**เลือก Collection สำหรับเก็บข้อมูล:**")
            
            # Refresh collections button
            col_coll1, col_coll2 = st.columns([2, 1])
            with col_coll1:
                if st.button("🔄 Refresh Collections", use_container_width=True):
                    if initialize_orchestrator():
                        try:
                            collections = st.session_state.orchestrator.list_collections()
                            st.session_state.available_collections = collections
                            st.success(f"✅ พบ {len(collections)} collections")
                        except Exception as e:
                            st.error(f"Error: {e}")
            
            with col_coll2:
                # Show cached status
                if st.session_state.available_collections:
                    st.caption(f"📦 {len(st.session_state.available_collections)} colls")
                else:
                    st.caption("🔄 Not loaded")
            
            # Load collections if not already loaded
            if not st.session_state.available_collections:
                if initialize_orchestrator():
                    pass  # Collections loaded in initialize_orchestrator
            
            # Create collection options
            collection_options = ["➕ สร้าง Collection ใหม่"] + st.session_state.available_collections
            
            # Collection selector
            selected_option = st.selectbox(
                "เลือก Collection:",
                options=collection_options,
                index=0,
                key="collection_selector_sidebar",
                help="เลือก collection ที่มีอยู่ หรือสร้างใหม่"
            )
            
            # Update selected collection
            if selected_option == "➕ สร้าง Collection ใหม่":
                st.session_state.create_new_collection = True
                st.session_state.selected_collection = None
                st.info("💡 กรอกชื่อ collection ใหม่ในฟอร์ม")
            else:
                st.session_state.create_new_collection = False
                st.session_state.selected_collection = selected_option
                st.success(f"✅ เลือก: **{selected_option}**")
                
                # Show collection info
                if st.session_state.orchestrator:
                    try:
                        info = st.session_state.orchestrator.get_collection_info(selected_option)
                        doc_count = info.get('document_count', 0)
                        st.caption(f"📊 มี {doc_count} documents")
                    except Exception as e:
                        st.caption(f"⚠️ ไม่สามารถดึงข้อมูลได้: {e}")
        
        st.markdown("---")
        
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
        
        # Quick action buttons
        if st.button("📦 Show All Collections", use_container_width=True):
            if initialize_orchestrator():
                try:
                    collections = st.session_state.orchestrator.list_collections()
                    st.session_state.available_collections = collections
                    st.success(f"Found {len(collections)} collections")
                    for coll in collections:
                        info = st.session_state.orchestrator.get_collection_info(coll)
                        st.text(f"  • {coll} ({info.get('document_count', 0)} docs)")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Quick switch for collection mode
        st.markdown("**Collection Mode:**")
        col_mode1, col_mode2 = st.columns(2)
        with col_mode1:
            if st.button("📂 Use Existing", use_container_width=True, help="เลือกจาก collection ที่มีอยู่"):
                st.session_state.create_new_collection = False
                st.rerun()
        with col_mode2:
            if st.button("➕ Create New", use_container_width=True, help="สร้าง collection ใหม่"):
                st.session_state.create_new_collection = True
                st.session_state.selected_collection = None
                st.rerun()
        
        if st.button("🔄 Refresh"):
            st.rerun()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Create New Job", "📊 Job History", "📦 Collections", "💬 Chatbot"])
    
    # Tab 1: Create New Job
    with tab1:
        st.header("สร้าง Job ใหม่")
        st.markdown("กรอกข้อมูลเพื่อสร้าง scraping job ใหม่")
        
        # Check if config was loaded from history
        loaded_config = st.session_state.get('load_config', None)
        if loaded_config:
            st.info(f"📋 โหลด config จาก: **{loaded_config.get('name')}**")
            col_clear1, col_clear2 = st.columns([1, 3])
            with col_clear1:
                if st.button("🗑️ Clear Loaded Config"):
                    st.session_state['load_config'] = None
                    st.rerun()
        
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
            
            # Get default values from loaded config
            default_name = loaded_config.get('name', '') if loaded_config else ''
            default_url = loaded_config.get('url', '') if loaded_config else ''
            default_api_url = loaded_config.get('api_url', '') if loaded_config else ''
            default_collection = loaded_config.get('collection_name', '') if loaded_config else ''
            default_extraction_prompt = loaded_config.get('extraction_prompt', '') if loaded_config else ''
            default_description = loaded_config.get('description', '') if loaded_config else ''
            default_use_selenium = loaded_config.get('use_selenium', True) if loaded_config else True
            default_wait_time = loaded_config.get('wait_time', 3) if loaded_config else 3
            default_chunk_size = loaded_config.get('chunk_size', 500) if loaded_config else 500
            default_chunk_overlap = loaded_config.get('chunk_overlap', 50) if loaded_config else 50
            
            with col1:
                job_name = st.text_input("ชื่อ Job *", value=default_name, placeholder="ตัวอย่าง: Scrape Students Page")
                url = st.text_input("URL ของหน้าเว็บ (optional)", value=default_url, placeholder="https://computing.kku.ac.th/students", help="กรอก URL เพื่อดึง HTML จากหน้าเว็บ (ต้องมี URL หรือ API อย่างน้อย 1 อย่าง)")
                api_url = st.text_input("API Endpoint (optional)", value=default_api_url, placeholder="https://api.computing.kku.ac.th/api/v1/...", help="กรอก API URL เพื่อดึง JSON จาก API (ต้องมี URL หรือ API อย่างน้อย 1 อย่าง)")
                
                # Collection input - show selector or text input based on selection
                if st.session_state.create_new_collection:
                    # Create new collection - show text input
                    collection_name = st.text_input(
                        "Collection Name * (ใหม่)",
                        value=default_collection,
                        placeholder="my_new_collection_embedding",
                        help="กรอกชื่อ collection ใหม่ (ระบบจะสร้างให้อัตโนมัติ)"
                    )
                    st.caption("💡 ชื่อควรลงท้ายด้วย '_embedding' เพื่อให้เข้าใจง่าย")
                else:
                    # Use existing collection - show selected
                    if st.session_state.selected_collection:
                        collection_name = st.session_state.selected_collection
                        st.text_input(
                            "Collection Name * (ที่เลือก)",
                            value=collection_name,
                            disabled=True,
                            help="เลือกจาก sidebar ด้านซ้าย"
                        )
                    else:
                        # No selection - allow manual input
                        collection_name = st.text_input(
                            "Collection Name *",
                            value=default_collection,
                            placeholder="students_embedding",
                            help="กรอกชื่อ collection หรือเลือกจาก sidebar"
                        )
            
            with col2:
                use_selenium = st.checkbox("ใช้ Selenium (สำหรับ JavaScript)", value=default_use_selenium)
                wait_time = st.number_input("Wait Time (วินาที)", min_value=1, max_value=10, value=default_wait_time)
                chunk_size = st.number_input("Chunk Size", min_value=100, max_value=2000, value=default_chunk_size)
            
            extraction_prompt = st.text_area(
                "Extraction Prompt *",
                value=default_extraction_prompt,
                placeholder="""ตัวอย่าง:
- ดึงข้อมูลลิงก์ทั้งหมดที่มีอยู่ในหน้า
- แยกชื่อลิงก์และ URL
- เก็บคำสำคัญที่เกี่ยวข้อง""",
                height=150
            )
            
            description = st.text_input("คำอธิบาย (optional)", value=default_description, placeholder="คำอธิบายเกี่ยวกับ job นี้")
            
            # Advanced options (collapsible)
            with st.expander("⚙️ Advanced Options"):
                col3, col4 = st.columns(2)
                
                # Get advanced defaults from loaded config
                default_category = ''
                default_hash_keys = ''
                default_delete_missing = False
                
                if loaded_config:
                    metadata_filter = loaded_config.get('metadata_filter', {})
                    default_category = metadata_filter.get('category', '')
                    hash_keys_list = loaded_config.get('hash_keys', [])
                    default_hash_keys = ','.join(hash_keys_list) if hash_keys_list else ''
                    default_delete_missing = loaded_config.get('delete_missing', False)
                
                with col3:
                    metadata_category = st.text_input("Metadata Category", value=default_category, placeholder="students")
                    hash_keys_input = st.text_input("Hash Keys (คั่นด้วย comma)", value=default_hash_keys, placeholder="url,title")
                
                with col4:
                    chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=200, value=default_chunk_overlap)
                    delete_missing = st.checkbox("Delete Missing Documents", value=default_delete_missing)
            
            submitted = st.form_submit_button("🚀 Create & Run Job", type="primary")
            
            if submitted:
                # Get batch mode and values from session state
                batch_mode = st.session_state.get("batch_mode_checkbox", False)
                batch_list_api = st.session_state.get("batch_list_api", "") if batch_mode else None
                detail_url_pattern = st.session_state.get("detail_url_pattern", "") if batch_mode else None
                detail_api_pattern = st.session_state.get("detail_api_pattern", "") if batch_mode else None
                
                # Validation
                if not job_name:
                    st.error("❌ กรุณากรอกชื่อ Job")
                elif not collection_name or (isinstance(collection_name, str) and not collection_name.strip()):
                    st.error("❌ กรุณาเลือกหรือกรอกชื่อ Collection")
                elif not extraction_prompt or (isinstance(extraction_prompt, str) and not extraction_prompt.strip()):
                    st.error("❌ กรุณากรอก Extraction Prompt")
                elif batch_mode:
                    if not batch_list_api:
                        st.error("❌ กรุณากรอก List API URL เมื่อเปิด Batch Processing")
                    else:
                        # Batch mode - proceed with batch processing
                        url = url or "batch_mode"  # Placeholder
                elif not url and not api_url:
                    st.error("❌ กรุณากรอก URL หรือ API Endpoint อย่างน้อย 1 อย่าง")
                
                # Only proceed if validation passes
                if (job_name and collection_name and extraction_prompt and 
                    (isinstance(collection_name, str) and collection_name.strip())):
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
                                        with st.expander("🔍 ดู JSON/Content ที่ส่งไปให้ LLM", expanded=True):
                                            st.markdown("**Content Preview ที่ส่งไปให้ LLM:**")
                                            if result.content_type == "json":
                                                try:
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
            
            # Add filter options
            col_filter1, col_filter2 = st.columns([3, 1])
            with col_filter1:
                filter_status = st.selectbox(
                    "กรอง Status:",
                    ["ทั้งหมด", "success", "failed"],
                    key="history_filter"
                )
            with col_filter2:
                show_count = st.number_input(
                    "แสดงจำนวน:",
                    min_value=5,
                    max_value=50,
                    value=10,
                    key="history_count"
                )
            
            # Filter history
            filtered_history = history
            if filter_status != "ทั้งหมด":
                filtered_history = [
                    job for job in history 
                    if job['result'].get('status') == filter_status
                ]
            
            # Show jobs
            for i, job_record in enumerate(reversed(filtered_history[-int(show_count):])):
                job_config = job_record['job_config']
                job_result = job_record['result']
                job_timestamp = job_record.get('timestamp', 'Unknown')
                
                # Format timestamp
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(job_timestamp)
                    display_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    display_time = job_timestamp
                
                # Status emoji
                status_emoji = "✅" if job_result.get('status') == 'success' else "❌"
                
                # Create expander with enhanced title
                with st.expander(
                    f"{status_emoji} Job #{len(filtered_history)-i}: {job_config.get('name', 'Unknown')} "
                    f"({job_result.get('documents_inserted', 0)} docs) - {display_time}"
                ):
                    # Action buttons
                    col_action1, col_action2, col_action3 = st.columns([1, 1, 2])
                    
                    with col_action1:
                        rerun_key = f"rerun_{i}_{job_config.get('job_id', i)}"
                        if st.button("🔄 Run Again", key=rerun_key, use_container_width=True):
                            # Create new job from config
                            with st.spinner("🔄 กำลังรัน job ซ้ำ..."):
                                if initialize_orchestrator():
                                    try:
                                        # Create new config with new job_id
                                        new_config = ScrapingJobConfig(
                                            job_id=str(uuid.uuid4()),
                                            name=job_config.get('name'),
                                            url=job_config.get('url'),
                                            api_url=job_config.get('api_url'),
                                            batch_mode=job_config.get('batch_mode', False),
                                            batch_list_api=job_config.get('batch_list_api'),
                                            detail_url_pattern=job_config.get('detail_url_pattern'),
                                            detail_api_pattern=job_config.get('detail_api_pattern'),
                                            collection_name=job_config.get('collection_name'),
                                            extraction_prompt=job_config.get('extraction_prompt'),
                                            description=job_config.get('description', ''),
                                            use_selenium=job_config.get('use_selenium', True),
                                            wait_time=job_config.get('wait_time', 3),
                                            chunk_size=job_config.get('chunk_size', 500),
                                            chunk_overlap=job_config.get('chunk_overlap', 50),
                                            metadata_filter=job_config.get('metadata_filter', {}),
                                            hash_keys=job_config.get('hash_keys', []),
                                            delete_missing=job_config.get('delete_missing', False)
                                        )
                                        
                                        # Execute job
                                        result = st.session_state.orchestrator.execute_job(new_config)
                                        
                                        # Save to history
                                        save_job_to_history(new_config.to_dict(), result.to_dict())
                                        
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
                                        
                                        # Compare with previous run
                                        st.markdown("---")
                                        st.subheader("📊 เปรียบเทียบกับครั้งก่อน:")
                                        
                                        prev_inserted = job_result.get('documents_inserted', 0)
                                        new_inserted = result.documents_inserted
                                        diff = new_inserted - prev_inserted
                                        
                                        if diff > 0:
                                            st.success(f"✅ เพิ่มขึ้น {diff} documents")
                                        elif diff < 0:
                                            st.warning(f"⚠️ ลดลง {abs(diff)} documents")
                                        else:
                                            st.info("📊 จำนวน documents เท่าเดิม")
                                        
                                        # Reload history
                                        st.rerun()
                                        
                                    except Exception as e:
                                        st.error(f"❌ Job failed: {e}")
                                        import traceback
                                        st.code(traceback.format_exc())
                    
                    with col_action2:
                        load_key = f"load_{i}_{job_config.get('job_id', i)}"
                        if st.button("📋 Load to Form", key=load_key, use_container_width=True):
                            # Load config to session state for form
                            st.session_state['load_config'] = job_config
                            st.info("✅ Config loaded! Switch to 'Create New Job' tab")
                    
                    with col_action3:
                        st.caption(f"⏱️ รันเมื่อ: {display_time}")
                    
                    st.markdown("---")
                    
                    # Show details in columns
                    col_h1, col_h2 = st.columns(2)
                    
                    with col_h1:
                        st.write("**📝 Configuration:**")
                        
                        # Display key info in readable format
                        st.write(f"**Name:** {job_config.get('name')}")
                        st.write(f"**Collection:** {job_config.get('collection_name')}")
                        st.write(f"**URL:** {job_config.get('url', 'N/A')}")
                        if job_config.get('api_url'):
                            st.write(f"**API URL:** {job_config.get('api_url')}")
                        
                        # Batch mode info
                        if job_config.get('batch_mode'):
                            st.write("**Batch Mode:** ✅ Enabled")
                            st.write(f"**List API:** {job_config.get('batch_list_api', 'N/A')}")
                        
                        with st.expander("🔍 ดู Full Config"):
                            st.json(job_config)
                    
                    with col_h2:
                        st.write("**📊 Result:**")
                        
                        # Display metrics
                        st.metric("Status", job_result.get('status', 'unknown'))
                        st.metric("Documents Processed", job_result.get('documents_processed', 0))
                        st.metric("Documents Inserted", job_result.get('documents_inserted', 0))
                        st.metric("Documents Skipped", job_result.get('documents_skipped', 0))
                        
                        if job_result.get('documents_updated', 0) > 0:
                            st.metric("Documents Updated", job_result.get('documents_updated', 0))
                        
                        st.write(f"**⏱️ Execution Time:** {job_result.get('execution_time', 0):.2f}s")
                        
                        if job_result.get('error_message'):
                            st.error(f"Error: {job_result.get('error_message')}")
                        
                        with st.expander("🔍 ดู Full Result"):
                            st.json(job_result)
    
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
    
    # Tab 4: Chatbot
    with tab4:
        st.header("💬 RAG Chatbot")
        st.markdown("ถามคำถามเกี่ยวกับข้อมูลที่เก็บในระบบ")
        
        # Initialize chatbot
        def initialize_chatbot():
            """Initialize chatbot"""
            if st.session_state.chatbot is None:
                try:
                    # Import chatbot
                    sys.path.insert(0, os.path.join(project_root, "main_app"))
                    from main_unified_chatbot_automated import UnifiedChatbotAutomated
                    
                    with st.spinner("🤖 กำลังโหลด Chatbot..."):
                        chatbot = UnifiedChatbotAutomated()
                        st.session_state.chatbot = chatbot
                        return True
                except Exception as e:
                    st.error(f"❌ Failed to initialize chatbot: {e}")
                    import traceback
                    with st.expander("🔍 ดู Error Details"):
                        st.code(traceback.format_exc())
                    return False
            return True
        
        # Chatbot initialization and controls
        col_init1, col_init2, col_init3 = st.columns([2, 1, 1])
        with col_init1:
            if st.button("🚀 Initialize Chatbot", type="primary", use_container_width=True):
                if initialize_chatbot():
                    st.success("✅ Chatbot initialized successfully!")
                    st.rerun()
        
        with col_init2:
            if st.button("🔄 Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        with col_init3:
            if st.button("🔄 Reload", use_container_width=True):
                st.session_state.chatbot = None
                st.session_state.chat_history = []
                st.rerun()
        
        # Check if chatbot is initialized
        if st.session_state.chatbot is None:
            st.info("👆 Click 'Initialize Chatbot' to start chatting")
            
            with st.expander("📋 ข้อมูลเพิ่มเติม", expanded=True):
                st.markdown("""
                ### ✨ Features:
                - **Hybrid Intent Classification**: ใช้ Rule-Based + LLM
                - **Auto-Discovery**: ค้นหา collections อัตโนมัติ
                - **Dynamic Retrievers**: สร้าง retrievers แบบ dynamic
                - **Multi-Agent Search**: ค้นหาจากทุก collections เมื่อไม่แน่ใจ
                
                ### 🎯 ตัวอย่างคำถาม:
                - "อาจารย์สมชาย" → อาจารย์และบุคลากร
                - "ติดต่อวิทยาลัย" → ข้อมูลติดต่อ
                - "ลิงก์จองห้องประชุม" → ลิงก์และระบบ
                - "ทุนการศึกษา" → ทุนการศึกษา
                """)
        else:
            # Show chatbot status
            if hasattr(st.session_state.chatbot, 'chatbot_map'):
                num_agents = len(st.session_state.chatbot.chatbot_map)
                st.success(f"✅ Chatbot ready with {num_agents} agents")
            
            # Display chat history
            st.markdown("---")
            
            # Chat container - display all messages
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.markdown(message["content"])
                        
                        # Show metadata if available
                        if "metadata" in message and message["metadata"].get("debug_output"):
                            with st.expander("🔍 Debug Info"):
                                st.code(message["metadata"]["debug_output"], language="text")
            
            # Chat input
            user_question = st.chat_input("พิมพ์คำถามของคุณ...")
            
            if user_question:
                # Add user message to history immediately
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_question
                })
                
                # Display user message immediately
                with st.chat_message("user"):
                    st.write(user_question)
                
                # Get answer from chatbot
                with st.chat_message("assistant"):
                    with st.spinner("🤔 กำลังคิด..."):
                        try:
                            # Capture output for display (optional)
                            import io
                            from contextlib import redirect_stdout, redirect_stderr
                            
                            # Create string buffer to capture output
                            output_buffer = io.StringIO()
                            
                            # Get answer (with output capture)
                            with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
                                answer = st.session_state.chatbot.answer(user_question)
                            
                            # Get captured output
                            debug_output = output_buffer.getvalue()
                            
                            # Display answer
                            st.markdown(answer)
                            
                            # Add assistant message to history
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": answer,
                                "metadata": {
                                    "debug_output": debug_output if debug_output else None
                                }
                            })
                            
                        except Exception as e:
                            error_msg = f"❌ เกิดข้อผิดพลาด: {str(e)}"
                            st.error(error_msg)
                            
                            # Add error to history
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": error_msg
                            })
                            
                            # Show traceback in expander
                            import traceback
                            with st.expander("🔍 ดู Error Details"):
                                st.code(traceback.format_exc())
            
            # Sidebar info for chatbot tab
            with st.sidebar:
                if st.session_state.chatbot:
                    st.markdown("---")
                    st.header("🤖 Chatbot Info")
                    
                    # Show available agents
                    if hasattr(st.session_state.chatbot, 'chatbot_map'):
                        st.write(f"**Available Agents:** {len(st.session_state.chatbot.chatbot_map)}")
                        for intent, config in list(st.session_state.chatbot.chatbot_map.items())[:5]:  # Show first 5
                            st.text(f"  {config['icon']} {config['name']}")
                            st.caption(f"    📦 {config['collection']}")
                        
                        if len(st.session_state.chatbot.chatbot_map) > 5:
                            st.caption(f"... and {len(st.session_state.chatbot.chatbot_map) - 5} more")
                    
                    # Show chat stats
                    st.markdown("---")
                    st.metric("Chat Messages", len(st.session_state.chat_history))
                    
                    # Export chat history
                    if st.session_state.chat_history:
                        chat_json = json.dumps(st.session_state.chat_history, ensure_ascii=False, indent=2)
                        st.download_button(
                            label="📥 Export Chat History",
                            data=chat_json,
                            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )


if __name__ == "__main__":
    main()

