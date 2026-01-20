import os
import sys

# ทำให้ import main_app ได้แน่ ๆ (กรณีรันจาก root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAIN_APP_DIR = os.path.join(PROJECT_ROOT, "main_app")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if MAIN_APP_DIR not in sys.path:
    sys.path.insert(0, MAIN_APP_DIR)

from main_app.main_unified_chatbot_automated import UnifiedChatbotAutomated

# สร้างครั้งเดียว (Singleton)
_chatbot = None

def get_chatbot():
    global _chatbot
    if _chatbot is None:
        _chatbot = UnifiedChatbotAutomated()  # automated ไม่มี strict_mode
    return _chatbot