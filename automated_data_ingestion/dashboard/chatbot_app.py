"""
Standalone Streamlit App: RAG Chatbot (separate port)
Run: streamlit run chatbot_app.py --server.port 8502
"""

import json
import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import streamlit as st

# Add parent directories to path for importing model_manager
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "main_app"))

# Import model manager
try:
    from automated_data_ingestion.utils.model_manager import get_model_manager
    MODEL_MANAGER_AVAILABLE = True
except ImportError:
    MODEL_MANAGER_AVAILABLE = False
    print("⚠️ Model manager not available")

# -----------------------------
# API Config
# -----------------------------
API_URL = os.getenv("CHATBOT_API_URL", "http://localhost:8000/api/v1/chat/completions")

# Timeout: (connect timeout, read timeout) วินาที
API_CONNECT_TIMEOUT = int(os.getenv("API_CONNECT_TIMEOUT", "15"))
API_READ_TIMEOUT = int(os.getenv("API_READ_TIMEOUT", "180"))

def make_session() -> requests.Session:
    """Session พร้อม retry สำหรับ network error (ไม่ retry ReadTimeout เพื่อไม่ให้ซ้ำคำถาม)"""
    session = requests.Session()
    retry = Retry(
        total=2,
        backoff_factor=1.5,
        status_forcelist=[502, 503, 504],  # retry เฉพาะ server error ชั่วคราว
        allowed_methods=["POST", "GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session
HEALTH_URL = os.getenv("CHATBOT_HEALTH_URL", "http://localhost:8000/api/v1/health")
META_URL = os.getenv("CHATBOT_META_URL", "http://localhost:8000/api/v1/chatbot/meta")
FEEDBACK_API_URL = os.getenv("FEEDBACK_API_URL", "http://localhost:8000/api/v1/feedback")
FEEDBACK_STATS_URL = os.getenv("FEEDBACK_STATS_URL", "http://localhost:8000/api/v1/feedback/stats")

# -----------------------------
# Page config (ต้องมาก่อน st.* อื่น ๆ)
# -----------------------------
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"  # ✅ ลดปัญหาไม่สมส่วน (sidebar ไม่กินพื้นที่)
)

# -----------------------------
# CSS (UI polish)
# -----------------------------
def inject_css():
    st.markdown(
        """
        <style>
          :root {
            --primary: #60a5fa;
            --primary-2: #3b82f6;
            --bg: #071224;
            --bg-soft: #0b1730;
            --card: rgba(255,255,255,0.08);
            --card-strong: rgba(255,255,255,0.12);
            --border: rgba(255,255,255,0.14);
            --text: #eaf2ff;
            --muted: #9fb3d9;
            --sidebar-text: #f8fbff;
        }

        /* ===== App Background ===== */
        .stApp {
            background: #ffffff !important;
            color: #0f172a;
        }

        /* ===== Layout ===== */
        .block-container {
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 1350px;
        }

        header {
            background: transparent !important;
        }

        /* ===== Sidebar ===== */
        section[data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                rgba(17, 43, 138, 0.95) 0%,
                rgba(30, 64, 175, 0.94) 100%
            ) !important;
            border-right: 1px solid rgba(255,255,255,0.08);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
        }
        .quick-prompts-wrap {
            margin-bottom: 28px;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.35rem;
        }

        /* ===== Sidebar text ===== */
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] small,
        section[data-testid="stSidebar"] div {
            color: var(--sidebar-text) !important;
        }

        /* ===== Sidebar card ===== */
        .sidebar-card {
            background: rgba(255,255,255,0.08);
            padding: 14px 16px;
            border-radius: 16px;
            margin-bottom: 12px;
            border: 1px solid rgba(255,255,255,0.14);
            box-shadow:
                0 8px 24px rgba(0,0,0,0.16),
                inset 0 1px 0 rgba(255,255,255,0.04);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
        }

        /* ===== Sidebar buttons ===== */
        section[data-testid="stSidebar"] .stButton > button {
            border-radius: 14px;
            min-height: 42px;
            font-weight: 600;
            border: 1px solid rgba(255,255,255,0.16) !important;
            background: rgba(255,255,255,0.14) !important;
            color: #ffffff !important;
            box-shadow:
                0 8px 20px rgba(0,0,0,0.12),
                inset 0 1px 0 rgba(255,255,255,0.05);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            transition: all 0.2s ease;
        }

        section[data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(255,255,255,0.20) !important;
            border-color: rgba(255,255,255,0.28) !important;
            transform: translateY(-1px);
            box-shadow: 0 10px 24px rgba(0,0,0,0.16);
        }

        section[data-testid="stSidebar"] .stButton > button p,
        section[data-testid="stSidebar"] .stButton > button span,
        section[data-testid="stSidebar"] .stButton > button div {
            color: #ffffff !important;
        }

        /* ===== Label ===== */
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stSelectbox label {
            color: #dbeafe !important;
            background: transparent !important;
            padding: 0 !important;
            margin-bottom: 6px !important;
            box-shadow: none !important;
            font-weight: 600;
        }

        /* ===== Selectbox: kill white background ===== */
        section[data-testid="stSidebar"] .stSelectbox,
        section[data-testid="stSidebar"] .stSelectbox > div,
        section[data-testid="stSidebar"] .stSelectbox > div > div {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }

        section[data-testid="stSidebar"] div[data-baseweb="select"] {
            background: transparent !important;
        }

        section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
            background: rgba(255,255,255,0.10) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255,255,255,0.18) !important;
            border-radius: 14px !important;
            min-height: 44px !important;
            box-shadow:
                0 8px 20px rgba(0,0,0,0.10),
                inset 0 1px 0 rgba(255,255,255,0.04) !important;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }

        section[data-testid="stSidebar"] div[data-baseweb="select"] > div:hover,
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div:focus-within {
            background: rgba(255,255,255,0.14) !important;
            border-color: rgba(255,255,255,0.28) !important;
            box-shadow:
                0 10px 24px rgba(0,0,0,0.14),
                inset 0 1px 0 rgba(255,255,255,0.05) !important;
        }

        section[data-testid="stSidebar"] div[data-baseweb="select"] span,
        section[data-testid="stSidebar"] div[data-baseweb="select"] div,
        section[data-testid="stSidebar"] div[data-baseweb="select"] input {
            color: #ffffff !important;
            background: transparent !important;
        }

        section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
            fill: #ffffff !important;
        }

        /* ===== Dropdown panel ===== */
        ul[role="listbox"] {
            background: rgba(12, 18, 35, 0.96) !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
            border-radius: 14px !important;
            box-shadow: 0 18px 40px rgba(0,0,0,0.28) !important;
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
        }

        ul[role="listbox"] li,
        ul[role="listbox"] div {
            color: #ffffff !important;
            background: transparent !important;
        }

        ul[role="listbox"] li:hover,
        ul[role="listbox"] [aria-selected="true"] {
            background: rgba(255,255,255,0.10) !important;
            border-radius: 10px;
        }

        /* ===== Sidebar inputs / textarea ===== */
        section[data-testid="stSidebar"] .stTextInput input,
        section[data-testid="stSidebar"] textarea,
        section[data-testid="stSidebar"] input {
            color: #ffffff !important;
            background: rgba(255,255,255,0.10) !important;
            border: 1px solid rgba(255,255,255,0.16) !important;
            border-radius: 14px !important;
            box-shadow: none !important;
        }

        /* ===== Sidebar metric ===== */
        section[data-testid="stSidebar"] div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.10) !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
            padding: 16px;
            border-radius: 16px;
            box-shadow:
                0 8px 24px rgba(0,0,0,0.12),
                inset 0 1px 0 rgba(255,255,255,0.04);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
        }

        /* ===== Global Buttons ===== */
        .stButton > button {
            border-radius: 12px;
            min-height: 42px;
            font-weight: 500;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.90);
            color: #0f172a;
            transition: all 0.2s ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 18px rgba(37,99,235,0.12);
        }

        /* ===== Topbar ===== */
        .topbar {
            background: #ffffff;
            border-radius: 20px;
            padding: 18px 22px;
            box-shadow: 0 10px 30px rgba(37,99,235,0.08);
            border: 1px solid #e2e8f0;
            margin-bottom: 14px;
            backdrop-filter: none;
            -webkit-backdrop-filter: none;
        }


        /* ===== Feature Section ===== */
        .feature-section {
            background: #ffffff;
            border: 1px solid #dbe4f0;
            border-radius: 22px;
            padding: 28px 28px 22px 28px;
            box-shadow: 0 10px 30px rgba(37, 99, 235, 0.06);
            margin-bottom: 22px;
            backdrop-filter: none;
            -webkit-backdrop-filter: none;
        }

        .feature-section h3 {
            color: #0f172a;
        }


        .info-list {
            display: flex;
            flex-direction: column;
            gap: 12px;
            margin-bottom: 28px;
            margin-top: 8px;
        }

        .info-item {
            background: #f8fbff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 12px 14px;
            color: #0f172a;
            font-size: 15px;
            line-height: 1.5;
            backdrop-filter: none;
            -webkit-backdrop-filter: none;
        }


        /* ===== Metrics ===== */
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #dbe4f0;
            padding: 16px;
            border-radius: 16px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
            color: #0f172a !important;
            backdrop-filter: none;
            -webkit-backdrop-filter: none;
        }

        /* ===== Chat (dark grey bubble) ===== */
        [data-testid="stChatMessageContent"] {
            border-radius: 16px;
            padding: 0.95rem 1.15rem;
            border: 1px solid #e2e8f0;
            background: #f8fafc;   /* เทาเข้มแบบนุ่ม */
            color: #0f172a !important;
        }

        /* ===== Badge ===== */
        .badge {
            display: inline-block;
            background: rgba(255,255,255,0.10);
            color: #e0edff;
            padding: 6px 10px;
            border-radius: 999px;
            font-size: 12px;
            border: 1px solid rgba(255,255,255,0.14);
            font-weight: 600;
            backdrop-filter: blur(8px);
        }

        /* ===== Input ===== */
        textarea {
            border-radius: 14px !important;
        }

        /* ===== Expander ===== */
        .streamlit-expanderHeader {
            font-weight: 600;
            color: #0f172a !important;
        }
        /* ข้อความทั่วไปฝั่ง main */
        .stMarkdown,
        .stCaption,
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] {
            color: #0f172a !important;
        }
        
        /* ===== Feedback section ===== */
        [data-testid="stFeedback"],
        [data-testid="stTextInput"] input {
            color: #0f172a !important;
            background: #ffffff !important;
        }

        /* label */
        [data-testid="stFeedback"] label {
            color: #0f172a !important;
        }
        
        /* ปุ่มใน main area เท่านั้น */
        section[data-testid="stSidebar"] ~ div .stButton > button,
        div[data-testid="stAppViewContainer"] .main .stButton > button {
            border: 1px solid #cbd5e1 !important;
            background: #ffffff !important;
            color: #0f172a !important;
        }

        /* hover เรืองแสง */
        section[data-testid="stSidebar"] ~ div .stButton > button:hover,
        div[data-testid="stAppViewContainer"] .main .stButton > button:hover {
            border-color: #60a5fa !important;
            box-shadow:
                0 0 0 1px rgba(96,165,250,0.45),
                0 0 10px rgba(96,165,250,0.28),
                0 4px 14px rgba(37,99,235,0.12) !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
inject_css()

# -----------------------------
# Path setup (ยังเก็บไว้เพื่อไม่ให้พังโครงสร้างเดิม)
# -----------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))      # .../dashboard
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))  # project root

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "main_app"))

# -----------------------------
# Session state
# -----------------------------
# สร้าง Session ID สำหรับแต่ละผู้ใช้
if 'session_id' not in st.session_state:
    import uuid
    st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"
    st.session_state.session_created_at = datetime.now().isoformat()
    print(f"✅ New user session: {st.session_state.session_id}")

if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "chatbot_meta" not in st.session_state:
    st.session_state.chatbot_meta = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "strict_mode" not in st.session_state:
    st.session_state.strict_mode = False
if "return_contexts" not in st.session_state:
    st.session_state.return_contexts = False
if "return_debug" not in st.session_state:
    st.session_state.return_debug = True
if 'chatbot_model' not in st.session_state:
    st.session_state.chatbot_model = os.getenv("CHATBOT_MODEL", "gemini-2.5-pro")
if 'available_models' not in st.session_state:
    st.session_state.available_models = []

# -----------------------------
# Helpers
# -----------------------------
def initialize_chatbot():
    """Initialize chatbot by checking FastAPI health (instead of importing chatbot class)."""
    if st.session_state.chatbot is None:
        try:
            with st.spinner("🔌 กำลังเชื่อมต่อ API..."):
                r = requests.get(HEALTH_URL, timeout=10)
            r.raise_for_status()

            # Fetch meta for sidebar (agents/collections)
            try:
                meta = requests.get(META_URL, timeout=10)
                meta.raise_for_status()
                st.session_state.chatbot_meta = meta.json()
            except Exception:
                st.session_state.chatbot_meta = None

            st.session_state.chatbot = "api_ready"
            return True
        except Exception as e:
            st.error(f"❌ Failed to connect API: {e}")
            import traceback
            with st.expander("🔍 ดู Error Details"):
                st.code(traceback.format_exc())
            return False
    return True

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

def render_topbar():
    left, b1, b2, b3 = st.columns([7, 1.2, 1.2, 1.2])

    with left:
        logo_col, title_col = st.columns([2, 10], vertical_alignment="center")
        with logo_col:
            # ใช้ path ที่ relative กับไฟล์นี้
            logo_path = os.path.join(os.path.dirname(__file__), "assets", "cp-kku-logo.png")
            st.image(
                logo_path,
                width=90,              # ปรับตรงนี้ได้ (40–52 กำลังสวย)
            )
        with title_col:
            st.markdown(
                """
                <div style="line-height:1.15">
                    <div style="font-size:20px; font-weight:700;">
                    ระบบถามตอบข้อมูลวิทยาลัยการคอมพิวเตอร์
                    </div>
                    <div style="font-size:14px; color:#475569;">
                    มหาวิทยาลัยขอนแก่น
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        # status badge
        if st.session_state.chatbot is None:
            st.markdown('<span class="badge">Status: Not Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge">Status: Ready</span>', unsafe_allow_html=True)

    with b1:
        if st.button("🚀 Start", type="primary", use_container_width=True):
            if initialize_chatbot():
                st.success("✅ Initialized!")
                st.rerun()

    with b2:
        if st.button("🧹 Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    with b3:
        if st.button("🔄 Reload", use_container_width=True):
            st.session_state.chatbot = None
            st.session_state.chatbot_meta = None
            st.session_state.chat_history = []
            st.rerun()
# -----------------------------

def ask(question: str):
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("🤔 กำลังคิด..."):
            try:
                payload = {
                    "question": question,
                    "strict_mode": st.session_state.get("strict_mode", False),
                    "return_contexts": st.session_state.get("return_contexts", False),
                    "return_debug": st.session_state.get("return_debug", True),
                    "model": st.session_state.get("chatbot_model"),
                }
                session = make_session()
                res = session.post(
                    API_URL,
                    json=payload,
                    timeout=(API_CONNECT_TIMEOUT, API_READ_TIMEOUT),
                )
                res.raise_for_status()
                data = res.json()

                answer = data.get("answer", "")
                intent = data.get("intent")
                confidence = data.get("confidence")
                contexts = data.get("contexts") or []
                debug_output = data.get("debug_output")

                st.markdown(answer)

                meta_bits = []
                if intent:
                    meta_bits.append(f"**Intent:** `{intent}`")
                if confidence is not None:
                    if isinstance(confidence, (int, float)):
                        meta_bits.append(f"**Confidence:** `{confidence:.2f}`")
                    else:
                        meta_bits.append(f"**Confidence:** `{confidence}`")
                # แสดงโมเดลที่ใช้
                if st.session_state.get("chatbot_model"):
                    meta_bits.append(f"**Model:** `{st.session_state.chatbot_model}`")
                if meta_bits:
                    st.caption(" • ".join(meta_bits))

                if contexts:
                    with st.expander("📚 Contexts ที่ใช้ตอบ"):
                        st.markdown(f"**📄 ใช้ context ทั้งหมด {len(contexts)} รายการ**")
                        for i, c in enumerate(contexts, 1):
                            st.markdown(f"**{i}.** {c}")

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "metadata": {
                        "intent": intent,
                        "confidence": confidence,
                        "contexts": contexts if contexts else None,
                        "debug_output": debug_output if debug_output else None,
                        "model": st.session_state.get("chatbot_model")  # ✅ บันทึกโมเดล
                    }
                })
                
                # ✅ Rerun เพื่อให้ chat history loop แสดง feedback form
                st.rerun()

                if debug_output:
                    with st.expander("🔍 Debug Info"):
                        st.code(debug_output, language="text")

            except requests.exceptions.ReadTimeout:
                error_msg = (
                    f"⏱️ เซิร์ฟเวอร์ใช้เวลานานเกินไป ({API_READ_TIMEOUT} วินาที)\n\n"
                    "ลองทำสิ่งต่อไปนี้:\n"
                    "- ถามคำถามสั้นลง หรือเจาะจงมากขึ้น\n"
                    "- รอสักครู่แล้วลองใหม่ (เซิร์ฟเวอร์อาจโหลดสูง)\n"
                    "- กด **Reload** แล้วเริ่มใหม่"
                )
                st.warning(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

            except requests.exceptions.ConnectionError:
                error_msg = "🔌 ไม่สามารถเชื่อมต่อ API ได้ กรุณาตรวจสอบว่าเซิร์ฟเวอร์ทำงานอยู่"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

            except requests.exceptions.HTTPError as e:
                code = e.response.status_code if e.response is not None else "?"
                error_msg = f"❌ API ตอบกลับ error {code}: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                import traceback
                with st.expander("🔍 ดู Error Details"):
                    st.code(traceback.format_exc())

            except Exception as e:
                error_msg = f"❌ เกิดข้อผิดพลาด: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                import traceback
                with st.expander("🔍 ดู Error Details"):
                    st.code(traceback.format_exc())


# -----------------------------
# UI
# -----------------------------
render_topbar()
st.divider()

with st.container(border=True):
    colA, colB = st.columns([3, 2])  # ✅ สมดุลขึ้น

    with colA:
        st.markdown("### ✨ ใช้งานเร็ว")
        st.markdown(
            "- กด **Start** เพื่อเริ่มใช้งาน\n"
            "- พิมพ์คำถามด้านล่าง หรือกดปุ่มตัวอย่าง\n"
            "- สามารถ Export ประวัติแชทได้จาก Sidebar"
        )

    with colB:
        st.markdown("### ⚡ Quick Prompts")
        qp1, qp2 = st.columns(2)
        with qp1:
            if st.button("📞 ติดต่อวิทยาลัย", use_container_width=True):
                st.session_state.pending_question = "ติดต่อวิทยาลัยได้ช่องทางไหนบ้าง"
        with qp2:
            if st.button("🎓 ทุนการศึกษา", use_container_width=True):
                st.session_state.pending_question = "มีทุนการศึกษาอะไรบ้าง และสมัครอย่างไร"

        qp3, qp4 = st.columns(2)
        with qp3:
            if st.button("🏢 จองห้องประชุม", use_container_width=True):
                st.session_state.pending_question = "มีลิงก์หรือขั้นตอนจองห้องประชุมไหม"
        with qp4:
            if st.button("👩‍🏫 รายชื่ออาจารย์", use_container_width=True):
                st.session_state.pending_question = "ขอรายชื่ออาจารย์และข้อมูลติดต่อ"

# Not ready state
if st.session_state.chatbot is None:
    # ✅ ย้าย Tip มา main กัน sidebar โล่ง
    st.info("💡 Tip: ถ้าเจอ 503 จาก AstraDB ให้ลองรอสักครู่แล้วกด Init ใหม่")

    with st.container(border=True):
        st.warning("👆 กด **Init** (ปุ่มด้านบน) เพื่อเริ่มใช้งาน Chatbot")
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
    st.stop()

# Status bar (metrics)
agents = 0
if st.session_state.chatbot_meta and isinstance(st.session_state.chatbot_meta, dict):
    agents = int(st.session_state.chatbot_meta.get("total", 0) or 0)

m1, m2, m3 = st.columns(3)
m1.metric("Status", "Ready")
m2.metric("Agents", agents)
m3.metric("Messages", len(st.session_state.chat_history))

st.divider()

# Chat history
for idx, message in enumerate(st.session_state.chat_history):
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])
            if "metadata" in message:
                md = message["metadata"] or {}
                meta_bits = []
                if md.get("intent"):
                    meta_bits.append(f"**Intent:** `{md.get('intent')}`")
                if md.get("confidence") is not None:
                    conf = md.get("confidence")
                    if isinstance(conf, (int, float)):
                        meta_bits.append(f"**Confidence:** `{conf:.2f}`")
                    else:
                        meta_bits.append(f"**Confidence:** `{conf}`")
                # แสดงโมเดลที่ใช้
                if md.get("model"):
                    meta_bits.append(f"**Model:** `{md.get('model')}`")
                if meta_bits:
                    st.caption(" • ".join(meta_bits))

                if md.get("contexts"):
                    with st.expander("📚 Contexts ที่ใช้ตอบ"):
                        st.markdown(f"**📄 ใช้ context ทั้งหมด {len(md['contexts'])} รายการ**")
                        for i, c in enumerate(md["contexts"], 1):
                            st.markdown(f"**{i}.** {c}")

                if md.get("debug_output"):
                    with st.expander("🔍 Debug Info"):
                        st.code(md["debug_output"], language="text")
            
            # ⭐ Feedback Section - แสดงเฉพาะข้อความล่าสุด
            is_last_message = (idx == len(st.session_state.chat_history) - 1)
            
            if is_last_message and message["role"] == "assistant":
                st.markdown("---")
                
                feedback_key = f"feedback_{idx}"
                
                # Initialize session state
                if f"submitted_{feedback_key}" not in st.session_state:
                    st.session_state[f"submitted_{feedback_key}"] = False
                if f"rating_{feedback_key}" not in st.session_state:
                    st.session_state[f"rating_{feedback_key}"] = None
                if f"comment_{feedback_key}" not in st.session_state:
                    st.session_state[f"comment_{feedback_key}"] = ""
                
                if not st.session_state[f"submitted_{feedback_key}"]:
                    # ใช้ Expander เพื่อให้กดแล้วฟอร์มแสดง
                    with st.expander("📊 **ประเมินคำตอบนี้** (กดเพื่อเปิดฟอร์ม)", expanded=True):
                        col1, col2 = st.columns([2, 3])
                        
                        with col1:
                            st.markdown("**ให้คะแนน:**")
                            # ใช้ callback เพื่อเก็บค่า rating
                            def save_rating():
                                st.session_state[f"rating_{feedback_key}"] = st.session_state[f"stars_{feedback_key}"]
                            
                            rating = st.feedback(
                                "stars",
                                key=f"stars_{feedback_key}",
                                on_change=save_rating
                            )
                        
                        with col2:
                            comment = st.text_input(
                                "💬 ความคิดเห็น (Optional):",
                                placeholder="ตอบได้ดีหรือยัง? มีข้อเสนอแนะไหม?",
                                key=f"comment_input_{feedback_key}",
                                value=st.session_state[f"comment_{feedback_key}"]
                            )
                            # เก็บ comment
                            st.session_state[f"comment_{feedback_key}"] = comment
                        
                        if st.button("✅ ส่งคะแนน", key=f"submit_{feedback_key}", type="primary"):
                            final_rating = st.session_state.get(f"rating_{feedback_key}")
                            final_comment = st.session_state.get(f"comment_{feedback_key}", "")
                            
                            if final_rating is not None:
                                try:
                                    # ดึง question จาก history
                                    question = ""
                                    if idx > 0 and st.session_state.chat_history[idx-1]["role"] == "user":
                                        question = st.session_state.chat_history[idx-1]["content"]
                                    
                                    md = message.get("metadata", {})
                                    feedback_payload = {
                                        "session_id": st.session_state.session_id,
                                        "question": question,
                                        "answer": message["content"],
                                        "model": md.get("model"),
                                        "intent": md.get("intent"),
                                        "rating": final_rating + 1,  # 0-4 → 1-5
                                        "comment": final_comment
                                    }
                                    
                                    feedback_res = requests.post(
                                        FEEDBACK_API_URL,
                                        json=feedback_payload,
                                        timeout=10
                                    )
                                    
                                    if feedback_res.status_code == 200:
                                        st.session_state[f"submitted_{feedback_key}"] = True
                                        st.success("✅ ขอบคุณสำหรับการให้คะแนน!")
                                        st.balloons()
                                        time.sleep(0.5)
                                        st.rerun()
                                    else:
                                        st.error(f"❌ ไม่สามารถบันทึกได้: {feedback_res.text}")
                                
                                except Exception as e:
                                    st.error(f"❌ เกิดข้อผิดพลาด: {e}")
                                    import traceback
                                    st.code(traceback.format_exc())
                            else:
                                st.warning("⚠️ กรุณาให้คะแนนก่อนส่ง")
                else:
                    st.success("✅ ขอบคุณสำหรับการให้คะแนนแล้ว!")

# Quick prompt
if st.session_state.pending_question:
    q = st.session_state.pending_question
    st.session_state.pending_question = None
    ask(q)

# Chat input
user_question = st.chat_input("พิมพ์คำถามของคุณ...")
if user_question:
    ask(user_question)

# Sidebar info (ตอน ready ค่อยมีเนื้อหา)
with st.sidebar:
    st.markdown('<div class="sidebar-card"><b>🤖 Chatbot Info</b></div>', unsafe_allow_html=True)
    st.metric("Chat Messages", len(st.session_state.chat_history))
    
    # ✅ Model Selection Section
    st.markdown("---")
    st.markdown('<div class="sidebar-card"><b>💬 Model Selection</b></div>', unsafe_allow_html=True)
    
    # Load available models from KKU API
    if not st.session_state.available_models:
        if MODEL_MANAGER_AVAILABLE:
            try:
                model_manager = get_model_manager()
                models = model_manager.get_models_with_fallback()
                st.session_state.available_models = models if models else []
                st.caption(f"✅ โหลด {len(models)} โมเดลจาก KKU API")
            except Exception as e:
                st.caption(f"⚠️ ไม่สามารถดึงโมเดล: {e}")
                # Fallback to default list
                st.session_state.available_models = [
                    "gemini-2.5-pro",
                    "gemini-2.5-flash",
                    "gemini-2.5-flash-lite",
                    "gpt-5-mini",
                    "claude-3-5-sonnet",
                ]
        else:
            # Fallback to default list
            st.session_state.available_models = [
                "gemini-2.5-pro",
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite",
                "gpt-5-mini",
                "claude-3-5-sonnet",
            ]
    
    # Refresh models button
    if st.button("🔄 Refresh Models", use_container_width=True):
        if MODEL_MANAGER_AVAILABLE:
            with st.spinner("กำลังดึงรายการโมเดล..."):
                try:
                    model_manager = get_model_manager()
                    models = model_manager.fetch_available_models()
                    if models:
                        st.session_state.available_models = models
                        st.success(f"✅ พบ {len(models)} โมเดล")
                    else:
                        st.warning("⚠️ ไม่พบโมเดล ใช้ default")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    # Model selector
    if st.session_state.available_models:
        try:
            current_index = st.session_state.available_models.index(st.session_state.chatbot_model)
        except ValueError:
            current_index = 0
            
        selected_model = st.selectbox(
            "เลือกโมเดล:",
            options=st.session_state.available_models,
            index=current_index,
            key="model_selector_sidebar"
        )
        
        # Update selected model
        if selected_model != st.session_state.chatbot_model:
            st.session_state.chatbot_model = selected_model
            st.success(f"✅ เปลี่ยนเป็น: **{selected_model}**")
            st.info("💡 โมเดลจะถูกใช้ในคำถามถัดไป")
        
        st.caption(f"🤖 ใช้โมเดล: **{st.session_state.chatbot_model}**")
    else:
        st.warning("⚠️ ไม่พบรายการโมเดล")
    
    st.markdown("---")

    # ✅ คืน sidebar collection/agent list เหมือนเดิม
    if st.session_state.chatbot_meta and isinstance(st.session_state.chatbot_meta, dict):
        agents_list = st.session_state.chatbot_meta.get("agents", []) or []
        if agents_list:
            st.markdown('<div class="sidebar-card"><b>🧩 Available Agents</b><br>', unsafe_allow_html=True)
            for a in agents_list[:6]:
                st.write(f"{a.get('icon','📦')} {a.get('name','-')}")
                st.caption(f"📦 {a.get('collection','-')}")
            st.markdown("</div>", unsafe_allow_html=True)

    st.session_state.strict_mode = st.toggle("Strict mode", value=st.session_state.strict_mode)
    st.session_state.return_contexts = st.toggle("Return contexts", value=st.session_state.return_contexts)
    st.session_state.return_debug = st.toggle("Return debug", value=st.session_state.return_debug)
    st.caption(f"API: {API_URL}")

    if st.session_state.chat_history:
        chat_json = json.dumps(st.session_state.chat_history, ensure_ascii=False, indent=2)
        st.download_button(
            label="📥 Export Chat History",
            data=chat_json,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

    st.markdown('<div class="sidebar-card"><b>🔗 Instant Link</b><br>พอร์ตนี้คือเว็บแยกสำหรับแชร์ลิงก์ให้คนอื่นเข้าใช้งาน</div>', unsafe_allow_html=True)
 