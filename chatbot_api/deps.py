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

# สร้างครั้งเดียว (Singleton) แต่รองรับ dynamic model
_chatbot = None
_current_model = None

def get_chatbot(model: str = None):
    """
    Get or create chatbot instance.
    If model is specified and different from current, recreate chatbot.
    """
    global _chatbot, _current_model
    
    # Get model from parameter or environment
    requested_model = model or os.getenv("CHATBOT_MODEL", "gemini-2.5-pro")
    
    # Recreate if model changed or chatbot doesn't exist
    if _chatbot is None or (requested_model != _current_model):
        # Set model in environment before creating chatbot
        if requested_model:
            os.environ["CHATBOT_MODEL"] = requested_model
            print(f"🤖 Creating chatbot with model: {requested_model}")
        
        _chatbot = UnifiedChatbotAutomated()
        _current_model = requested_model
    
    return _chatbot