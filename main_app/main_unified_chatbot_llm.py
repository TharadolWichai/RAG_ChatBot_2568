# main_unified_chatbot_llm.py - Unified Multi-Agent RAG Chatbot with LLM-Based Intent Classification
# ใช้ GPT-4o-mini สำหรับ Intent Classification

import sys
import os
import re
import json
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

# PyThaiNLP for Thai text processing
try:
    from pythainlp import word_tokenize
    PYTHAINLP_AVAILABLE = True
except ImportError:
    PYTHAINLP_AVAILABLE = False

# OpenAI for Intent Classification
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI library not available. Install with: pip install openai")

# Import all chatbot modules
try:
    from main_allpeople import retriever as allpeople_retriever, manual_qa_chain as allpeople_qa
    ALLPEOPLE_AVAILABLE = True
except Exception as e:
    print(f"⚠️ AllPeople chatbot not available: {e}")
    ALLPEOPLE_AVAILABLE = False

try:
    from main_contact import retriever as contact_retriever, manual_qa_chain as contact_qa
    CONTACT_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Contact chatbot not available: {e}")
    CONTACT_AVAILABLE = False

try:
    from main_links import retriever as links_retriever, manual_qa_chain as links_qa
    LINKS_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Links chatbot not available: {e}")
    LINKS_AVAILABLE = False

try:
    from main_scholarship import retriever as scholarship_retriever, manual_qa_chain as scholarship_qa
    SCHOLARSHIP_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Scholarship chatbot not available: {e}")
    SCHOLARSHIP_AVAILABLE = False

try:
    from main_student_club import retriever as club_retriever, manual_qa_chain as club_qa
    CLUB_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Student Club chatbot not available: {e}")
    CLUB_AVAILABLE = False

try:
    from main_students import retriever as students_retriever, manual_qa_chain as students_qa
    STUDENTS_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Students chatbot not available: {e}")
    STUDENTS_AVAILABLE = False

load_dotenv()

# Debug: Show which agents are available
print("\n🔍 Agent Availability Status:")
print(f"   AllPeople: {ALLPEOPLE_AVAILABLE}")
print(f"   Contact: {CONTACT_AVAILABLE}")
print(f"   Links: {LINKS_AVAILABLE}")
print(f"   Scholarship: {SCHOLARSHIP_AVAILABLE}")
print(f"   Student Club: {CLUB_AVAILABLE}")
print(f"   Students: {STUDENTS_AVAILABLE}")
print(f"   OpenAI: {OPENAI_AVAILABLE}")
print()

# ==========================================
# LLM-Based Intent Classification System
# ==========================================

class LLMIntentClassifier:
    """ระบบจำแนกประเภทคำถามด้วย LLM (GPT-4o-mini)"""
    
    def __init__(self):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library is required. Install with: pip install openai")
        
        # Get API key from environment (support both OPENAI_API_KEY and OPENROUTER_API_KEY)
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY or OPENROUTER_API_KEY not found in environment variables")
        
        # Initialize OpenAI client (works with OpenRouter too)
        base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
        # Model configuration
        self.model = "openai/gpt-4o-mini-2024-07-18"  # Fast and cheap model
        self.temperature = 0.1  # Low temperature for consistent results
        
        # Intent descriptions for the prompt
        self.intent_descriptions = {
            "allpeople": {
                "name": "อาจารย์และบุคลากร",
                "description": "ข้อมูลเกี่ยวกับอาจารย์, ผู้ช่วยศาสตราจารย์, รองศาสตราจารย์, ศาสตราจารย์, บุคลากร, คณาจารย์, หัวหน้าภาควิชา, ประวัติอาจารย์, ผลงานวิจัยของอาจารย์",
                "examples": ["อาจารย์สมชาย", "ผศ.ดร.สมหญิง", "หัวหน้าภาควิชา", "อาจารย์ที่สอนวิชา AI"]
            },
            "contact": {
                "name": "ข้อมูลติดต่อ",
                "description": "ข้อมูลติดต่อหน่วยงาน, เบอร์โทรศัพท์, อีเมล, ที่อยู่, แฟกซ์, Hot Line, การติดต่อวิทยาลัย",
                "examples": ["ติดต่อวิทยาลัย", "เบอร์โทรศัพท์", "อีเมล", "ที่อยู่คณะ"]
            },
            "links": {
                "name": "ลิงก์และระบบ",
                "description": "ลิงก์ระบบต่างๆ, การจองห้องประชุม, จองห้องแล็บ, แบบฟอร์มต่างๆ, ระบบจัดการเอกสาร, ระบบวิทยานิพนธ์, ระบบโครงงาน, ดาวน์โหลดเอกสาร",
                "examples": ["ลิงก์จองห้องประชุม", "แบบฟอร์มลาพักผ่อน", "ระบบจัดการโครงงาน", "ดาวน์โหลดแบบฟอร์ม"]
            },
            "scholarship": {
                "name": "ทุนการศึกษา",
                "description": "ทุนการศึกษา, ทุนวิจัย, ทุนนานาชาติ, ทุน ASEAN, ทุน GMS, คุณสมบัติของทุน, เงื่อนไขทุน, การสมัครทุน, ประเภทของทุน",
                "examples": ["ทุนการศึกษา", "ทุนวิจัย", "ทุนนานาชาติ", "สมัครทุน", "คุณสมบัติทุน"]
            },
            "student_club": {
                "name": "สโมสรนักศึกษา",
                "description": "สโมสรนักศึกษา, คณะกรรมการสโมสร, ประธานสโมสร, รองประธานสโมสร, เลขานุการ, เหรัญญิก, ความเป็นมาของสโมสร, กิจกรรมสโมสร",
                "examples": ["ประธานสโมสร", "คณะกรรมการสโมสร", "กิจกรรมสโมสร", "ประวัติสโมสร"]
            },
            "students": {
                "name": "ลิงก์บริการนักศึกษา",
                "description": "บริการสำหรับนักศึกษา, ลิงก์โครงงาน, ลิงก์วิทยานิพนธ์, ลิงก์ลงทะเบียน, ลิงก์ตารางสอน, ลิงก์ผลการเรียน, ลิงก์สหกิจศึกษา",
                "examples": ["ลิงก์โครงงานนักศึกษา", "ลิงก์ลงทะเบียน", "ลิงก์ตารางสอน", "บริการนักศึกษา"]
            }
        }
    
    def classify(self, query: str) -> Tuple[str, float, str]:
        """
        จำแนกประเภทคำถามด้วย LLM
        Returns: (intent_name, confidence_score, reason)
        """
        
        # Build the prompt
        prompt = self._build_classification_prompt(query)
        
        try:
            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert intent classifier for a university chatbot system. You must respond ONLY with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=200,
                response_format={"type": "json_object"}  # Force JSON response
            )
            
            # Parse response
            result_text = response.choices[0].message.content.strip()
            result = json.loads(result_text)
            
            intent = result.get("intent", "unknown")
            confidence = float(result.get("confidence", 0.0))
            reason = result.get("reason", "No reason provided")
            
            # Validate intent
            if intent not in self.intent_descriptions and intent != "unknown":
                print(f"⚠️ Invalid intent '{intent}' from LLM, falling back to unknown")
                intent = "unknown"
                confidence = 0.0
            
            return intent, confidence, reason
            
        except Exception as e:
            print(f"❌ Error calling LLM: {e}")
            # Fallback to unknown
            return "unknown", 0.0, f"Error: {str(e)}"
    
    def _build_classification_prompt(self, query: str) -> str:
        """สร้าง prompt สำหรับ LLM"""
        
        # Build intent list
        intent_list = []
        for intent_key, intent_info in self.intent_descriptions.items():
            intent_list.append(
                f"- **{intent_key}** ({intent_info['name']}): {intent_info['description']}\n"
                f"  ตัวอย่าง: {', '.join(intent_info['examples'])}"
            )
        
        intents_text = "\n".join(intent_list)
        
        prompt = f"""คุณเป็น Intent Classifier สำหรับระบบ Chatbot ของวิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น

มี 6 categories ดังนี้:

{intents_text}

คำถามจากผู้ใช้: "{query}"

งานของคุณ:
1. วิเคราะห์คำถามและเลือก intent ที่เหมาะสมที่สุด
2. ถ้าไม่แน่ใจหรือคำถามไม่ตรงกับ intent ใดเลย ให้ตอบ "unknown"
3. ประเมินความมั่นใจ (confidence) จาก 0.0 ถึง 1.0
4. อธิบายเหตุผลสั้นๆ

ตอบเป็น JSON รูปแบบนี้เท่านั้น:
{{
  "intent": "intent_key หรือ unknown",
  "confidence": 0.0-1.0,
  "reason": "เหตุผลสั้นๆ ว่าทำไมเลือก intent นี้"
}}

ตัวอย่างการตอบ:
- คำถาม "อาจารย์สมชาย" → {{"intent": "allpeople", "confidence": 0.95, "reason": "ถามเกี่ยวกับอาจารย์โดยตรง"}}
- คำถาม "ลิงก์จองห้องประชุม" → {{"intent": "links", "confidence": 0.98, "reason": "ขอลิงก์สำหรับจองห้องประชุม"}}
- คำถาม "ทุนการศึกษา" → {{"intent": "scholarship", "confidence": 0.97, "reason": "ถามเกี่ยวกับทุนการศึกษา"}}
- คำถาม "สวัสดี" → {{"intent": "unknown", "confidence": 0.0, "reason": "เป็นการทักทายทั่วไป ไม่เกี่ยวกับ intent ใดๆ"}}

**สำคัญ:** ตอบเป็น JSON เท่านั้น ห้ามมีข้อความอื่น"""
        
        return prompt

# ==========================================
# Unified Chatbot Router with LLM Classification
# ==========================================

class UnifiedChatbotLLM:
    """Unified Chatbot ที่ใช้ LLM ในการจำแนก Intent"""
    
    def __init__(self):
        # Initialize LLM classifier
        try:
            self.classifier = LLMIntentClassifier()
            self.llm_available = True
            print("✅ LLM Intent Classifier initialized successfully!")
        except Exception as e:
            print(f"❌ Failed to initialize LLM classifier: {e}")
            self.llm_available = False
            return
        
        # Map intents to chatbot functions
        self.chatbot_map = {}
        
        if ALLPEOPLE_AVAILABLE:
            self.chatbot_map["allpeople"] = {
                "name": "อาจารย์และบุคลากร",
                "qa_function": allpeople_qa,
                "icon": "👨‍🏫"
            }
        
        if CONTACT_AVAILABLE:
            self.chatbot_map["contact"] = {
                "name": "ข้อมูลติดต่อ",
                "qa_function": contact_qa,
                "icon": "📞"
            }
        
        if LINKS_AVAILABLE:
            self.chatbot_map["links"] = {
                "name": "ลิงก์และระบบ",
                "qa_function": links_qa,
                "icon": "🔗"
            }
        
        if SCHOLARSHIP_AVAILABLE:
            self.chatbot_map["scholarship"] = {
                "name": "ทุนการศึกษา",
                "qa_function": scholarship_qa,
                "icon": "🎓"
            }
        
        if CLUB_AVAILABLE:
            self.chatbot_map["student_club"] = {
                "name": "สโมสรนักศึกษา",
                "qa_function": club_qa,
                "icon": "🎭"
            }
        
        if STUDENTS_AVAILABLE:
            self.chatbot_map["students"] = {
                "name": "ลิงก์นักศึกษา",
                "qa_function": students_qa,
                "icon": "📚"
            }
        
        print(f"✅ Unified Chatbot (LLM) initialized with {len(self.chatbot_map)} agents")
        for intent, config in self.chatbot_map.items():
            print(f"   {config['icon']} {config['name']}")
    
    def answer(self, question: str) -> str:
        """ตอบคำถามโดยใช้ LLM เลือก Agent"""
        
        if not self.llm_available:
            return "❌ LLM Intent Classifier ไม่พร้อมใช้งาน กรุณาตรวจสอบ API key"
        
        # Step 1: LLM Intent Classification
        print(f"\n🤖 กำลังใช้ GPT-4o-mini วิเคราะห์คำถาม...")
        intent, confidence, reason = self.classifier.classify(question)
        
        print(f"\n🎯 LLM Intent Classification:")
        print(f"   ประเภท: {intent}")
        print(f"   ความมั่นใจ: {confidence:.2f} (0.0-1.0)")
        print(f"   เหตุผล: {reason}")
        
        # Step 2: Route to appropriate chatbot
        if intent == "unknown" or confidence < 0.5:
            # Low confidence or unknown - use multi-agent search
            print(f"⚠️ ความมั่นใจต่ำหรือไม่แน่ใจประเภท - จะค้นหาจากทุก Agent")
            return self._multi_agent_search(question)
        
        if intent not in self.chatbot_map:
            # Agent not available
            print(f"⚠️ Agent '{intent}' ไม่พร้อมใช้งาน - จะค้นหาจากทุก Agent")
            return self._multi_agent_search(question)
        
        # Step 3: Use specific chatbot
        chatbot_config = self.chatbot_map[intent]
        print(f"   ➡️  เลือก Agent: {chatbot_config['icon']} {chatbot_config['name']}")
        print(f"{'='*60}\n")
        
        try:
            answer = chatbot_config["qa_function"](question)
            return f"{chatbot_config['icon']} [{chatbot_config['name']}]\n\n{answer}"
        except Exception as e:
            print(f"❌ Error from {chatbot_config['name']}: {e}")
            return f"ขอโทษ เกิดข้อผิดพลาดจาก Agent {chatbot_config['name']}"
    
    def _multi_agent_search(self, question: str) -> str:
        """ค้นหาจากทุก Agent และรวมผลลัพธ์"""
        print("🔍 กำลังค้นหาจากทุก Agent...\n")
        print(f"📊 Total agents to search: {len(self.chatbot_map)}")
        print(f"📋 Agents: {list(self.chatbot_map.keys())}\n")
        
        results = []
        
        for i, (intent, config) in enumerate(self.chatbot_map.items(), 1):
            print(f"\n{'='*60}")
            print(f"🔸 Agent {i}/{len(self.chatbot_map)}: {config['icon']} {config['name']} ({intent})")
            print(f"{'='*60}")
            try:
                print(f"   🚀 Calling QA function for {config['name']}...")
                answer = config["qa_function"](question)
                print(f"\n   ✅ Got response from {config['name']}")
                print(f"   📏 Answer length: {len(answer) if answer else 0} chars")
                
                # Check if answer is meaningful
                answer_lines = answer.strip().split('\n')
                last_line = answer_lines[-1].lower() if answer_lines else ""
                
                is_not_found = any(phrase in last_line for phrase in [
                    "ไม่พบข้อมูล", "ไม่มีข้อมูล", "no data", "not found", 
                    "ขอโทษ", "sorry", "ไม่สามารถ"
                ])
                
                is_meaningful = len(answer.strip()) > 50 and not is_not_found
                
                if answer and is_meaningful:
                    results.append({
                        "agent": config["name"],
                        "icon": config["icon"],
                        "answer": answer
                    })
                    print(f"   ✅ พบข้อมูลที่มีความหมาย! ({len(answer)} chars)")
                else:
                    reason = "not found" if is_not_found else "too short"
                    print(f"   ⚪ ไม่พบข้อมูลที่มีความหมาย ({reason})")
                    
            except Exception as e:
                print(f"   ❌ Error from {config['name']}: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"   ⏭️  Moving to next agent... ({i}/{len(self.chatbot_map)} done)")
        
        # Combine results
        if not results:
            return "ขอโทษ ไม่พบข้อมูลที่ตรงกับคำถามของคุณในระบบ"
        
        if len(results) == 1:
            result = results[0]
            return f"{result['icon']} [{result['agent']}]\n\n{result['answer']}"
        
        # Multiple results
        combined = "พบข้อมูลจากหลาย Agent:\n\n"
        for i, result in enumerate(results, 1):
            combined += f"{result['icon']} **{result['agent']}**\n"
            combined += f"{result['answer']}\n\n"
            if i < len(results):
                combined += f"{'-'*60}\n\n"
        
        return combined
    
    def show_help(self):
        """แสดงคำแนะนำการใช้งาน"""
        print("\n" + "="*60)
        print("📖 คำแนะนำการใช้งาน Unified Chatbot (LLM Version)")
        print("="*60)
        print("\n🤖 ระบบใช้ GPT-4o-mini วิเคราะห์คำถามอัตโนมัติ!")
        print("\nระบบสามารถตอบคำถามในหัวข้อต่อไปนี้:\n")
        
        for intent, config in self.chatbot_map.items():
            print(f"{config['icon']} {config['name']}")
            desc = self.classifier.intent_descriptions[intent]["description"]
            print(f"   {desc[:80]}...")
            print()
        
        print("\n💡 ตัวอย่างคำถาม:")
        print("   - อาจารย์สมชาย → LLM จะเลือก: อาจารย์และบุคลากร")
        print("   - ติดต่อวิทยาลัย → LLM จะเลือก: ข้อมูลติดต่อ")
        print("   - ลิงก์จองห้องประชุม → LLM จะเลือก: ลิงก์และระบบ")
        print("   - ทุนการศึกษา → LLM จะเลือก: ทุนการศึกษา")
        print("   - ประธานสโมสร → LLM จะเลือก: สโมสรนักศึกษา")
        print("   - ลิงก์โครงงาน → LLM จะเลือก: ลิงก์นักศึกษา")
        print()
        print("📝 คำสั่งพิเศษ:")
        print("   - 'help' หรือ 'ช่วยเหลือ' = แสดงคำแนะนำ")
        print("   - 'agents' หรือ 'รายการ' = แสดง Agent ทั้งหมด")
        print("   - 'exit' หรือ 'ออก' = ออกจากโปรแกรม")
        print("="*60)

# ==========================================
# Main Program
# ==========================================

def main():
    """Main function สำหรับ Unified Chatbot (LLM Version)"""
    
    # Fix encoding for Windows terminal (only when running as main script)
    if sys.platform == "win32":
        import codecs
        try:
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
        except:
            pass  # Already detached
    
    print("\n" + "="*60)
    print("🤖 Unified RAG Chatbot (LLM Version) - วิทยาลัยการคอมพิวเตอร์ มข.")
    print("="*60)
    print("🧠 ใช้ GPT-4o-mini วิเคราะห์คำถามอัตโนมัติ!")
    print()
    
    # Initialize unified chatbot
    try:
        chatbot = UnifiedChatbotLLM()
        if not chatbot.llm_available:
            print("\n❌ ไม่สามารถเริ่มระบบได้ - LLM ไม่พร้อมใช้งาน")
            print("กรุณาตรวจสอบ:")
            print("1. ติดตั้ง openai: pip install openai")
            print("2. ตั้งค่า OPENAI_API_KEY ใน .env")
            return
    except Exception as e:
        print(f"\n❌ Error initializing chatbot: {e}")
        return
    
    # Show initial help
    chatbot.show_help()
    
    # Main loop
    while True:
        print("\n" + "-"*60)
        try:
            question = input("\n❓ ถามมาเลย LLM Version (หรือพิมพ์ 'help'): ").strip()
        except EOFError:
            print("\n👋 ออกจากโปรแกรม")
            break
        
        if not question:
            continue
        
        # Handle special commands
        if question.lower() in ['exit', 'quit', 'ออก', 'จบ']:
            print("\n👋 ขอบคุณที่ใช้บริการ!")
            break
        
        if question.lower() in ['help', 'ช่วยเหลือ', 'คำแนะนำ']:
            chatbot.show_help()
            continue
        
        if question.lower() in ['agents', 'รายการ', 'agent']:
            print("\n📋 รายการ Agent ทั้งหมด:")
            for intent, config in chatbot.chatbot_map.items():
                print(f"   {config['icon']} {config['name']}")
            continue
        
        # Get answer
        try:
            print()
            answer = chatbot.answer(question)
            print("\n" + "="*60)
            print("🤖 คำตอบ:")
            print("="*60)
            print(answer)
        except Exception as e:
            print(f"\n❌ เกิดข้อผิดพลาด: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

