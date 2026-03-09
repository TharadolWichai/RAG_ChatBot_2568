"""
Standalone Streamlit App: RAG Chatbot (separate port)
Run: streamlit run chatbot_app.py --server.port 8502
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
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
    st.markdown(
        """
        <style>
          /* FIX: กันหัวขาด */
        .block-container {
            padding-top: 3.2rem !important;
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 1400px;
        }

        h1, h2, h3 { letter-spacing: -0.02em; }

        /* buttons / inputs */
        .stButton>button {
        border-radius: 14px;
        padding: 0.6rem 1rem;
        height: 42px;                     /* ⬅ กันปุ่มโดนตัด */
        }

          /* chat spacing */
        [data-testid="stChatMessage"] { padding: 0.35rem 0; }
        [data-testid="stChatMessageContent"]{
            border-radius: 18px;
            padding: 0.9rem 1.1rem;
        }

        /* sidebar */
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.2rem;
        }
        section[data-testid="stSidebar"] {
            border-right: 1px solid #e2e8f0;
        }

        /* card / container */
        div[data-testid="stMetric"],
        .sidebar-card {
        background: #ffffff;
        }

        /* chat bubble assistant */
        [data-testid="stChatMessageContent"] {
        background: #f8fafc;
        }

        /* badge */
        .badge {
        background: #eff6ff;
        border-color: #93c5fd;
        color: #1d4ed8;
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
    left, b1, b2, b3 = st.columns([7, 1.2, 1.2, 1.2])

    with left:
        logo_col, title_col = st.columns([2, 10], vertical_alignment="center")
        with logo_col:
            APP_DIR = Path(__file__).resolve().parent        # โฟลเดอร์ที่มี chatbot_app.py
            LOGO_PATH = APP_DIR / "assets" / "cp-kku-logo.png"

            if LOGO_PATH.exists():
                st.image(str(LOGO_PATH), width=90)
            else:
                st.warning(f"Logo not found: {LOGO_PATH}")
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
    st.markdown("## 💬 RAG Chatbot")

    # --- Status Card ---
    with st.container(border=True):
        st.markdown("### Status")
        status = "Ready ✅" if st.session_state.chatbot is not None else "Not Ready ⚠️"
        st.write(status)
        st.caption(f"API: {API_URL}")

        cols = st.columns(2)
        cols[0].metric("Messages", len(st.session_state.chat_history))

        agents_count = 0
        if st.session_state.chatbot_meta and isinstance(st.session_state.chatbot_meta, dict):
            agents_count = int(st.session_state.chatbot_meta.get("total", 0) or 0)
        cols[1].metric("Agents", agents_count)

    # --- Settings Card ---
    with st.container(border=True):
        st.markdown("### ⚙️ Settings")
        st.caption("ปรับพฤติกรรมการตอบของบอท")

        st.session_state.strict_mode = st.toggle(
            "Strict mode",
            value=st.session_state.strict_mode,
            help="เข้มงวดกับการตอบจาก context มากขึ้น ลดการเดา"
        )
        st.session_state.return_contexts = st.toggle(
            "Show contexts",
            value=st.session_state.return_contexts,
            help="แสดง context ที่ใช้ตอบ"
        )
        st.session_state.return_debug = st.toggle(
            "Show debug",
            value=st.session_state.return_debug,
            help="แสดง debug output สำหรับ dev"
        )

    # --- Tools / Actions ---
    with st.container(border=True):
        st.markdown("### 🧰 Tools")
        c1, c2 = st.columns(2)

        if c1.button("🧹 Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        if c2.button("🔄 Reset API", use_container_width=True):
            st.session_state.chatbot = None
            st.session_state.chatbot_meta = None
            st.session_state.chat_history = []
            st.rerun()

        st.caption("Tip: ถ้าเจอ 503/เชื่อมต่อหลุด ให้ Reset แล้วกด Start ใหม่")

    # --- Export ---
    with st.container(border=True):
        st.markdown("### 📤 Export")
        if st.session_state.chat_history:
            chat_json = json.dumps(st.session_state.chat_history, ensure_ascii=False, indent=2)
            st.download_button(
                label="📥 Download chat history (.json)",
                data=chat_json,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.caption("ยังไม่มีประวัติแชทให้ดาวน์โหลด")

    # --- Share / About ---
    with st.container(border=True):
        st.markdown("### 🔗 Share")
        st.caption("พอร์ตนี้คือเว็บแยกสำหรับแชร์ลิงก์ให้คนอื่นเข้าใช้งาน")
        st.code("streamlit run chatbot_app.py --server.port 8502", language="bash")
