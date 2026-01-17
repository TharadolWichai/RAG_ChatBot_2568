"""
Standalone Streamlit App: RAG Chatbot (separate port)
Run: streamlit run chatbot_app.py --server.port 8502
"""

import json
import os
import sys
from datetime import datetime

import streamlit as st

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
            padding-top: 1.5rem;
        }

        /* badge */
        .badge {
            display:inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            border: 1px solid rgba(229,231,235,0.7);
            font-size: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
inject_css()

# -----------------------------
# Path setup
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
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# -----------------------------
# Helpers
# -----------------------------
def initialize_chatbot():
    """Initialize chatbot"""
    if st.session_state.chatbot is None:
        try:
            from main_unified_chatbot_automated import UnifiedChatbotAutomated
            with st.spinner("🤖 กำลังโหลด Chatbot..."):
                st.session_state.chatbot = UnifiedChatbotAutomated()
                return True
        except Exception as e:
            st.error(f"❌ Failed to initialize chatbot: {e}")
            import traceback
            with st.expander("🔍 ดู Error Details"):
                st.code(traceback.format_exc())
            return False
    return True

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

def render_topbar():
    left, b1, b2, b3 = st.columns([7, 1.2, 1.2, 1.2])

    with left:
        st.markdown("## 💬 RAG Chatbot")
        st.caption("ถามคำถามเกี่ยวกับข้อมูลที่เก็บในระบบ (AstraDB + RAG)")
        if st.session_state.chatbot is None:
            st.markdown('<span class="badge">Status: Not Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge">Status: Ready</span>', unsafe_allow_html=True)

    with b1:
        if st.button("🚀 Init", type="primary", use_container_width=True):
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
                import io
                from contextlib import redirect_stderr, redirect_stdout

                output_buffer = io.StringIO()
                with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
                    answer = st.session_state.chatbot.answer(question)

                debug_output = output_buffer.getvalue()
                st.markdown(answer)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "metadata": {"debug_output": debug_output if debug_output else None}
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
            "- กด **Init** เพื่อเริ่มใช้งาน\n"
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
agents = len(getattr(st.session_state.chatbot, "chatbot_map", {}) or {})
m1, m2, m3 = st.columns(3)
m1.metric("Status", "Ready")
m2.metric("Agents", agents)
m3.metric("Messages", len(st.session_state.chat_history))

st.divider()

# Chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])
            if "metadata" in message and message["metadata"].get("debug_output"):
                with st.expander("🔍 Debug Info"):
                    st.code(message["metadata"]["debug_output"], language="text")

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

    if hasattr(st.session_state.chatbot, "chatbot_map"):
        st.markdown('<div class="sidebar-card"><b>🧩 Available Agents</b><br>', unsafe_allow_html=True)
        for _, config in list(st.session_state.chatbot.chatbot_map.items())[:6]:
            st.write(f"{config.get('icon','📦')} {config.get('name','-')}")
            st.caption(f"📦 {config.get('collection','-')}")
        st.markdown("</div>", unsafe_allow_html=True)

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
