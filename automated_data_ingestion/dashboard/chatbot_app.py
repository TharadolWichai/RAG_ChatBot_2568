"""
Standalone Streamlit App: RAG Chatbot (separate port)
Run: streamlit run chatbot_app.py --server.port 8502
"""

import json
import os
import sys
from datetime import datetime
import requests
import streamlit as st

# -----------------------------
# API Config
# -----------------------------
API_URL = os.getenv("CHATBOT_API_URL", "http://localhost:8000/api/v1/chat/completions")
HEALTH_URL = os.getenv("CHATBOT_HEALTH_URL", "http://localhost:8000/api/v1/health")
META_URL = os.getenv("CHATBOT_META_URL", "http://localhost:8000/api/v1/chatbot/meta")

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
    st.markdown("""
    <style>

    /* ----------- Layout ----------- */

    .block-container {
        padding-top: 2.8rem !important;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1050px;
    }

    h1,h2,h3{
        letter-spacing:-0.02em;
    }

    /* ----------- Buttons ----------- */

    .stButton>button{
        border-radius:12px;
        height:40px;
        border:1px solid #e5e7eb;
        background:white;
        font-weight:500;
        transition:all .15s ease;
    }

    .stButton>button:hover{
        border-color:#2563eb;
        color:#2563eb;
    }

    /* ----------- Chat Layout ----------- */

    [data-testid="stChatMessage"]{
        padding:10px 0;
    }

    /* avatar / emoji */

    [data-testid="stChatMessageAvatar"]{
        width:32px;
        height:32px;
        font-size:16px;
    }

    /* bubble base */

    [data-testid="stChatMessageContent"]{
        border-radius:16px;
        padding:12px 16px;
        font-size:15px;
        line-height:1.65;
        box-shadow:none;
        border:none;
        max-width:720px;
    }

    /* assistant bubble */

    [data-testid="stChatMessage"][data-testid*="assistant"]
    [data-testid="stChatMessageContent"]{
        background:#f8fafc;
        color:#0f172a;
    }

    /* user bubble */

    [data-testid="stChatMessage"][data-testid*="user"]
    [data-testid="stChatMessageContent"]{
        background:#2563eb;
        color:white;
    }

    /* ----------- Chat Input ----------- */

    textarea{
        border-radius:14px !important;
        border:1px solid #e5e7eb !important;
        font-size:15px !important;
        padding:10px !important;
    }

    textarea:focus{
        border-color:#2563eb !important;
        box-shadow:none !important;
    }

    /* ----------- Metrics ----------- */

    [data-testid="stMetric"]{
        border-radius:12px;
        padding:12px;
        background:#ffffff;
        border:1px solid #f1f5f9;
    }

    /* ----------- Containers ----------- */

    div[data-testid="stContainer"]{
        border-radius:16px;
    }

    /* ----------- Sidebar ----------- */

    section[data-testid="stSidebar"] .block-container{
        padding-top:1.5rem;
    }

    /* ----------- Badge ----------- */

    .badge{
        background:#eff6ff;
        border:1px solid #bfdbfe;
        color:#1d4ed8;
        padding:4px 10px;
        border-radius:8px;
        font-size:13px;
        font-weight:500;
    }

    /* ----------- Cleaner spacing ----------- */

    hr{
        margin-top:1.2rem;
        margin-bottom:1.2rem;
    }

    </style>
    """, unsafe_allow_html=True)

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
    left, b1, b2 = st.columns([8, 1, 1])

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
                }
                res = requests.post(API_URL, json=payload, timeout=120)
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
                        "debug_output": debug_output if debug_output else None
                    }
                })

                if debug_output:
                    with st.expander("🔍 Debug Info"):
                        st.code(debug_output, language="text")

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
        st.markdown("""
            <div style="display:flex; align-items:center; gap:10px; font-size:24px; font-weight:700;">
            <svg xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 -960 960 960"
            fill="#327ffb">
            <path d="m381-240 424-424-57-56-368 367-169-170-57 57 227 226Zm0 113L42-466l169-170 170 170 366-367 172 168-538 538Z"/>
            </svg>ใช้งานเร็ว</div>
            """, unsafe_allow_html=True)
        st.markdown(
            "- กด **Start** เพื่อเริ่มใช้งาน\n"
            "- พิมพ์คำถามด้านล่าง หรือกดปุ่มตัวอย่าง\n"
            "- สามารถ Export ประวัติแชทได้จาก Sidebar"
        )

    with colB:
        st.markdown("""
            <div style="display:flex; align-items:center; gap:10px; font-size:24px; font-weight:700;">
            <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#327ffb"><path d="m720-120-56-56 63-64H560v-80h167l-63-64 56-56 160 160-160 160Zm-600 0v-600q0-33 23.5-56.5T200-800h480q33 0 56.5 23.5T760-720v203q-10-2-20-2.5t-20-.5q-10 0-20 .5t-20 2.5v-203H200v400h283q-2 10-2.5 20t-.5 20q0 1₀ .5 2₀t2.5 2₀H24₀L1₂₀-1₂₀Zm16₀-44₀h3₂₀v-8₀H28₀v8₀Zm0 16₀h2₀₀v-8₀H28₀v8₀Zm-8₀ 8₀v-4₀₀ 4₀₀Z"/></svg>Quick Prompts</div>
            
            """, unsafe_allow_html=True)
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
        st.warning("👆 กด **Start** (ปุ่มด้านบน) เพื่อเริ่มใช้งาน Chatbot")
        with st.expander("📋 ข้อมูลเพิ่มเติม", expanded=True):
            st.markdown("""
            ### Features:
            - **Hybrid Intent Classification**: ใช้ Rule-Based + LLM
            - **Auto-Discovery**: ค้นหา collections อัตโนมัติ
            - **Dynamic Retrievers**: สร้าง retrievers แบบ dynamic
            - **Multi-Agent Search**: ค้นหาจากทุก collections เมื่อไม่แน่ใจ

            ### ตัวอย่างคำถาม:
            - "ฉันจะติดต่อวิทยาลัยการคอมพิวเตอร์ได้อย่างไร?"
            - "มีทุนการศึกษาอะไรบ้าง และสมัครอย่างไร?"
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

    st.markdown("### ⚙️ Settings")

    st.session_state.strict_mode = st.toggle(
        "Strict mode", value=st.session_state.strict_mode
    )

    st.session_state.return_contexts = st.toggle(
        "Return contexts", value=st.session_state.return_contexts
    )

    st.session_state.return_debug = st.toggle(
        "Debug", value=st.session_state.return_debug
    )

    st.divider()

    st.metric("Messages", len(st.session_state.chat_history))

    if st.session_state.chat_history:
        chat_json = json.dumps(
            st.session_state.chat_history,
            ensure_ascii=False,
            indent=2
        )

        st.download_button(
            "📥 Export Chat",
            data=chat_json,
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

