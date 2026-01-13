# main_unified_chatbot_hybrid.py - Unified Multi-Agent RAG Chatbot with Hybrid Intent Classification
# รวม Rule-Based + LLM-Based เข้าด้วยกัน (Best of Both Worlds!)

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

# OpenAI for Intent Classification (fallback)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI library not available. LLM fallback will be disabled.")

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

try:
    from main_researchgroup import retriever as research_retriever, manual_qa_chain as research_qa
    RESEARCH_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Research Group chatbot not available: {e}")
    RESEARCH_AVAILABLE = False

try:
    from main_bsc_entrance import retriever as bsc_retriever, manual_qa_chain as bsc_qa
    BSC_AVAILABLE = True
except Exception as e:
    print(f"⚠️ BSC Entrance chatbot not available: {e}")
    BSC_AVAILABLE = False

try:
    from main_digital_services import retriever as digital_retriever, manual_qa_chain as digital_qa
    DIGITAL_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Digital Services chatbot not available: {e}")
    DIGITAL_AVAILABLE = False

try:
    from main_graduate import retriever as graduate_retriever, manual_qa_chain as graduate_qa
    GRADUATE_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Graduate Programs chatbot not available: {e}")
    GRADUATE_AVAILABLE = False

load_dotenv()

# Debug: Show which agents are available
print("\n🔍 Agent Availability Status:")
print(f"   AllPeople: {ALLPEOPLE_AVAILABLE}")
print(f"   Contact: {CONTACT_AVAILABLE}")
print(f"   Links: {LINKS_AVAILABLE}")
print(f"   Scholarship: {SCHOLARSHIP_AVAILABLE}")
print(f"   Student Club: {CLUB_AVAILABLE}")
print(f"   Students: {STUDENTS_AVAILABLE}")
print(f"   Research Group: {RESEARCH_AVAILABLE}")
print(f"   BSC Entrance: {BSC_AVAILABLE}")
print(f"   Digital Services: {DIGITAL_AVAILABLE}")
print(f"   Graduate Programs: {GRADUATE_AVAILABLE}")
print(f"   OpenAI (LLM): {OPENAI_AVAILABLE}")
print()

# ==========================================
# Hybrid Intent Classification System
# ==========================================

class HybridIntentClassifier:
    """
    ระบบจำแนกประเภทคำถามแบบ Hybrid
    - ลอง Rule-Based ก่อน (เร็ว, ไม่เสียค่าใช้จ่าย)
    - ถ้าความมั่นใจต่ำ → ใช้ LLM ช่วย (แม่นยำ แต่เสียค่าใช้จ่าย)
    - Fallback → Multi-agent search
    """
    
    def __init__(self):
        # Rule-based patterns
        self.intent_patterns = {
            "allpeople": {
                "keywords": [
                    "อาจารย์", "ผู้ช่วย", "รอง", "ศาสตราจารย์", "อ.", "ดร.", "หัวหน้า",
                    "บุคลากร", "คณาจารย์", "ผู้สอน", "professor", "lecturer", "faculty",
                    "staff", "teacher", "อาจารย์ประจำ", "สายวิชาการ"
                ],
                "patterns": [
                    r'อาจารย์.*',
                    r'ผู้ช่วยศาสตราจารย์.*',
                    r'รองศาสตราจารย์.*',
                    r'ศาสตราจารย์.*',
                    r'.*หัวหน้า.*',
                    r'.*บุคลากร.*',
                    r'.*คณาจารย์.*'
                ]
            },
            "contact": {
                "keywords": [
                    "ติดต่อ", "โทร", "อีเมล", "email", "เบอร์", "โทรศัพท์", "ที่อยู่",
                    "contact", "address", "phone", "hotline", "แฟกซ์", "fax",
                    "ต่อ", "เบอร์โทร"
                ],
                "patterns": [
                    r'.*ติดต่อ.*',
                    r'.*โทร.*',
                    r'.*อีเมล.*',
                    r'.*เบอร์.*',
                    r'.*ที่อยู่.*',
                    r'\d{3}-\d+',
                    r'.*@.*\..*'
                ]
            },
            "links": {
                "keywords": [
                    "ลิงก์", "ระบบ", "link", "url", "เว็บไซต์", "website", "หน้าเว็บ",
                    "จอง", "booking", "reservation", "ห้องประชุม", "ห้องแล็บ",
                    "แบบฟอร์ม", "form", "ดาวน์โหลด", "download", "อัปโหลด", "upload"
                ],
                "patterns": [
                    r'.*ลิงก์.*',
                    r'.*ระบบ.*',
                    r'.*จอง.*',
                    r'.*แบบฟอร์ม.*',
                    r'https?://.*'
                ]
            },
            "scholarship": {
                "keywords": [
                    "ทุน", "ทุนการศึกษา", "ทุนวิจัย", "scholarship", "grant", "funding",
                    "ทุนนานาชาติ", "ทุน asean", "ทุน gms", "ทุนส่งเสริม",
                    "คุณสมบัติทุน", "เงื่อนไขทุน", "ผลประโยชน์ทุน"
                ],
                "patterns": [
                    r'.*ทุน.*',
                    r'.*scholarship.*',
                    r'.*grant.*',
                    r'.*funding.*'
                ]
            },
            "student_club": {
                "keywords": [
                    "สโมสร", "สโมสรนักศึกษา", "คณะกรรมการ", "ประธาน", "รองประธาน",
                    "student club", "club", "กรรมการ", "เลขานุการ", "เหรัญญิก",
                    "ความเป็นมา", "ประวัติสโมสร"
                ],
                "patterns": [
                    r'.*สโมสร.*',
                    r'.*คณะกรรมการ.*',
                    r'.*ประธาน.*',
                    r'.*student.*club.*'
                ]
            },
            "students": {
                "keywords": [
                    "นักศึกษา", "student", "โครงงาน", "project", "ฝึกงาน", "internship",
                    "สหกิจ", "co-op", "วิทยานิพนธ์", "thesis", "ตารางสอน", "schedule",
                    "ลงทะเบียน", "registration", "เกรด", "grade", "ผลการเรียน"
                ],
                "patterns": [
                    r'.*นักศึกษา.*',
                    r'.*โครงงาน.*',
                    r'.*ฝึกงาน.*',
                    r'.*สหกิจ.*',
                    r'.*วิทยานิพนธ์.*',
                    r'.*ลงทะเบียน.*'
                ]
            },
            "research": {
                "keywords": [
                    "กลุ่มวิจัย", "research group", "lab", "laboratory", "ห้องแล็บ",
                    "นักวิจัย", "researcher", "งานวิจัย", "research", "ผลงานวิจัย",
                    "AIDA", "AIII", "AGT", "ASC", "NLSP", "I-SERG", "MLISLAB"
                ],
                "patterns": [
                    r'.*กลุ่มวิจัย.*',
                    r'.*research.*group.*',
                    r'.*lab.*',
                    r'.*ห้องแล็บ.*',
                    r'.*งานวิจัย.*'
                ]
            },
            "bsc_entrance": {
                "keywords": [
                    "รับเข้า", "สมัคร", "admission", "entrance", "รอบ", "โควตา",
                    "tcas", "portfolio", "เกณฑ์", "คะแนน", "หลักสูตร", "ปริญญาตรี",
                    "undergraduate", "รับสมัคร", "สอบเข้า"
                ],
                "patterns": [
                    r'.*รับเข้า.*',
                    r'.*สมัคร.*',
                    r'.*admission.*',
                    r'.*รอบ.*\d+.*',
                    r'.*tcas.*',
                    r'.*portfolio.*',
                    r'.*โควตา.*'
                ]
            },
            "digital_services": {
                "keywords": [
                    "บริการดิจิตอล", "digital service", "web hosting", "โฮสติ้ง",
                    "virtual machine", "vm", "เครื่องเสมือน", "apple store", "google play",
                    "grammarly", "chatgpt plus", "ai server", "snapdrop", "แชร์ไฟล์",
                    "บริการ", "ส่วนที่", "ข้อตกลง", "เทคโนโลยี"
                ],
                "patterns": [
                    r'.*บริการ.*ดิจิตอล.*',
                    r'.*digital.*service.*',
                    r'.*web.*hosting.*',
                    r'.*virtual.*machine.*',
                    r'.*apple.*store.*',
                    r'.*google.*play.*',
                    r'.*grammarly.*',
                    r'.*chatgpt.*plus.*',
                    r'.*snapdrop.*',
                    r'.*ส่วนที่.*\d+.*'
                ]
            },
            "graduate": {
                "keywords": [
                    "บัณฑิตศึกษา", "graduate", "ปริญญาโท", "master", "มหาบัณฑิต", "ป.โท",
                    "ปริญญาเอก", "phd", "ph.d", "ดุษฎีบัณฑิต", "ป.เอก", "doctoral",
                    "หลักสูตรโท", "หลักสูตรเอก", "สมัครโท", "สมัครเอก", "คุณสมบัติโท",
                    "คุณสมบัติเอก", "ค่าเทอมโท", "ค่าเทอมเอก", "อาจารย์ที่ปรึกษา"
                ],
                "patterns": [
                    r'.*บัณฑิตศึกษา.*',
                    r'.*ปริญญาโท.*',
                    r'.*ป\.โท.*',
                    r'.*master.*',
                    r'.*ปริญญาเอก.*',
                    r'.*ป\.เอก.*',
                    r'.*phd.*',
                    r'.*ph\.d.*',
                    r'.*doctoral.*',
                    r'.*หลักสูตร.*โท.*',
                    r'.*หลักสูตร.*เอก.*'
                ]
            }
        }
        
        # LLM descriptions (for fallback)
        self.llm_intent_descriptions = {
            "allpeople": {
                "name": "อาจารย์และบุคลากร",
                "description": "ข้อมูลเกี่ยวกับอาจารย์, ผู้ช่วยศาสตราจารย์, รองศาสตราจารย์, ศาสตราจารย์, บุคลากร, คณาจารย์, หัวหน้าภาควิชา",
                "examples": ["อาจารย์สมชาย", "ผศ.ดร.สมหญิง", "หัวหน้าภาควิชา"]
            },
            "contact": {
                "name": "ข้อมูลติดต่อ",
                "description": "ข้อมูลติดต่อหน่วยงาน, เบอร์โทรศัพท์, อีเมล, ที่อยู่, แฟกซ์, Hot Line",
                "examples": ["ติดต่อวิทยาลัย", "เบอร์โทรศัพท์", "อีเมล"]
            },
            "links": {
                "name": "ลิงก์และระบบ",
                "description": "ลิงก์ระบบต่างๆ, การจองห้องประชุม, จองห้องแล็บ, แบบฟอร์ม, ระบบจัดการเอกสาร",
                "examples": ["ลิงก์จองห้องประชุม", "แบบฟอร์มลาพักผ่อน", "ดาวน์โหลดแบบฟอร์ม"]
            },
            "scholarship": {
                "name": "ทุนการศึกษา",
                "description": "ทุนการศึกษา, ทุนวิจัย, ทุนนานาชาติ, ทุน ASEAN, ทุน GMS, คุณสมบัติทุน",
                "examples": ["ทุนการศึกษา", "ทุนวิจัย", "ทุนนานาชาติ"]
            },
            "student_club": {
                "name": "สโมสรนักศึกษา",
                "description": "สโมสรนักศึกษา, คณะกรรมการสโมสร, ประธานสโมสร, กิจกรรมสโมสร",
                "examples": ["ประธานสโมสร", "คณะกรรมการสโมสร", "กิจกรรมสโมสร"]
            },
            "students": {
                "name": "ลิงก์บริการนักศึกษา",
                "description": "บริการสำหรับนักศึกษา, ลิงก์โครงงาน, วิทยานิพนธ์, ลงทะเบียน, ตารางสอน",
                "examples": ["ลิงก์โครงงานนักศึกษา", "ลิงก์ลงทะเบียน", "ตารางสอน"]
            },
            "research": {
                "name": "กลุ่มวิจัย",
                "description": "ข้อมูลกลุ่มวิจัย, ห้องแล็บ, นักวิจัย, AIDA Lab, AIII Lab, AGT Lab",
                "examples": ["กลุ่มวิจัย AIDA", "ห้องแล็บ AI", "รายชื่อกลุ่มวิจัย"]
            },
            "bsc_entrance": {
                "name": "การรับเข้าศึกษา",
                "description": "การรับเข้าศึกษาระดับปริญญาตรี, รอบ Portfolio, TCAS, โควตา, เกณฑ์คะแนน",
                "examples": ["รอบ Portfolio", "เกณฑ์รับเข้า", "TCAS รอบ 3"]
            },
            "digital_services": {
                "name": "บริการดิจิตอล",
                "description": "บริการดิจิตอล, Web Hosting, Virtual Machine, Apple Store, Google Play, Grammarly, ChatGPT Plus",
                "examples": ["Web Hosting", "Virtual Machine", "Apple Store", "Grammarly"]
            },
            "graduate": {
                "name": "หลักสูตรบัณฑิตศึกษา",
                "description": "ข้อมูลหลักสูตรบัณฑิตศึกษา, ปริญญาโท (Master), ปริญญาเอก (Ph.D.), คุณสมบัติผู้สมัคร, ค่าใช้จ่าย, อาจารย์ที่ปรึกษา",
                "examples": ["ปริญญาโท", "ปริญญาเอก", "หลักสูตรโท"]
            }
        }
        
        # Initialize LLM if available
        self.llm_client = None
        if OPENAI_AVAILABLE:
            try:
                api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
                if api_key:
                    base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
                    self.llm_client = OpenAI(api_key=api_key, base_url=base_url)
                    self.llm_model = "openai/gpt-4o-mini"
                    print("✅ LLM fallback initialized successfully!")
            except Exception as e:
                print(f"⚠️ LLM fallback initialization failed: {e}")
    
    def classify(self, query: str) -> Tuple[str, float, str, str]:
        """
        จำแนกประเภทคำถามแบบ Hybrid (Improved version)
        Returns: (intent_name, confidence_score, method_used, reason)
        """
        
        # Step 1: Try Rule-Based first (fast & free!)
        rule_intent, rule_confidence = self._rule_based_classify(query)
        
        # Adjusted thresholds for better performance
        HIGH_CONFIDENCE_THRESHOLD = 7.0  # ลดลงจาก 8.0 → ใช้ rule-based ได้ง่ายขึ้น
        
        if rule_confidence >= HIGH_CONFIDENCE_THRESHOLD:
            # Rule-based is confident enough!
            print(f"   ✅ Rule-Based มั่นใจสูง (คะแนน: {rule_confidence:.2f})")
            return rule_intent, rule_confidence, "rule_based", "High confidence from keyword/pattern matching"
        
        # Step 2: Low confidence - Use LLM fallback if available
        if self.llm_client and rule_confidence < HIGH_CONFIDENCE_THRESHOLD:
            print(f"   ⚠️ Rule-Based มั่นใจต่ำ (คะแนน: {rule_confidence:.2f}) → ใช้ LLM ช่วย")
            llm_intent, llm_confidence, llm_reason = self._llm_classify(query)
            
            # Use LLM result if it gives a valid intent (even with lower confidence)
            if llm_intent != "unknown" and llm_confidence >= 0.4:  # ลดจาก 0.6 → 0.4
                print(f"   ✅ LLM ให้คำแนะนำ: {llm_intent} (มั่นใจ: {llm_confidence:.2f})")
                return llm_intent, llm_confidence, "llm_fallback", llm_reason
            elif llm_intent != "unknown":
                # LLM ให้ intent แต่ confidence ต่ำ - ยังดีกว่า unknown
                print(f"   🔄 LLM มั่นใจต่ำแต่ยังใช้ได้: {llm_intent} (มั่นใจ: {llm_confidence:.2f})")
                return llm_intent, max(llm_confidence, 0.3), "llm_low_conf", llm_reason
            else:
                # LLM ไม่แน่ใจ - ใช้ rule-based แทน (ดีกว่า unknown)
                print(f"   🔄 LLM ไม่แน่ใจ → ใช้ Rule-Based แทน: {rule_intent} (คะแนน: {rule_confidence:.2f})")
                return rule_intent, rule_confidence, "rule_fallback", "LLM uncertain, using rule-based result"
        
        # Step 3: No LLM available or rule-based result is ok
        if rule_intent != "unknown":
            return rule_intent, rule_confidence, "rule_based", "LLM not available, using rule-based"
        
        return "unknown", 0.0, "rule_based", "No match found"
    
    def _rule_based_classify(self, query: str) -> Tuple[str, float]:
        """Rule-based classification (same as main_unified_chatbot.py)"""
        query_lower = query.lower()
        
        # Tokenize with PyThaiNLP if available
        if PYTHAINLP_AVAILABLE:
            tokens = word_tokenize(query_lower, engine='newmm')
        else:
            tokens = query_lower.split()
        
        # Calculate scores for each intent
        intent_scores = {}
        
        for intent, config in self.intent_patterns.items():
            score = 0.0
            matched_keywords = 0
            
            # 1. Keyword matching
            for keyword in config["keywords"]:
                if keyword.lower() in query_lower:
                    matched_keywords += 1
                    score += 3.0
                elif any(keyword.lower() in token.lower() for token in tokens):
                    matched_keywords += 1
                    score += 2.0
            
            # 2. Pattern matching
            pattern_matches = 0
            for pattern in config["patterns"]:
                if re.search(pattern, query_lower):
                    pattern_matches += 1
                    score += 2.0
            
            if matched_keywords > 0 or pattern_matches > 0:
                # Bonus for multiple matches
                if matched_keywords >= 2:
                    score *= 1.5
                intent_scores[intent] = score
            else:
                intent_scores[intent] = 0.0
        
        # Get best match
        if not intent_scores:
            return "unknown", 0.0
        
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        
        if best_intent[1] < 5.0:
            return "unknown", best_intent[1]
        
        return best_intent
    
    def _llm_classify(self, query: str) -> Tuple[str, float, str]:
        """LLM-based classification (fallback)"""
        if not self.llm_client:
            return "unknown", 0.0, "LLM not available"
        
        try:
            # Build prompt
            prompt = self._build_llm_prompt(query)
            
            # Call LLM
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an intent classifier. Respond ONLY with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=150,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            result_text = response.choices[0].message.content.strip()
            result = json.loads(result_text)
            
            intent = result.get("intent", "unknown")
            confidence = float(result.get("confidence", 0.0))
            reason = result.get("reason", "No reason provided")
            
            # Validate intent
            if intent not in self.llm_intent_descriptions and intent != "unknown":
                intent = "unknown"
                confidence = 0.0
            
            return intent, confidence, reason
            
        except Exception as e:
            print(f"   ❌ LLM Error: {e}")
            return "unknown", 0.0, f"Error: {str(e)}"
    
    def _build_llm_prompt(self, query: str) -> str:
        """Build prompt for LLM"""
        intent_list = []
        for intent_key, intent_info in self.llm_intent_descriptions.items():
            intent_list.append(
                f"- **{intent_key}** ({intent_info['name']}): {intent_info['description']}"
            )
        
        intents_text = "\n".join(intent_list)
        
        prompt = f"""คุณเป็น Intent Classifier สำหรับระบบ Chatbot ของวิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น

มี 9 categories ดังนี้:

{intents_text}

คำถามจากผู้ใช้: "{query}"

วิเคราะห์และตอบเป็น JSON:
{{
  "intent": "intent_key หรือ unknown",
  "confidence": 0.0-1.0,
  "reason": "เหตุผลสั้นๆ"
}}

**สำคัญ:** ตอบเป็น JSON เท่านั้น"""
        
        return prompt

# ==========================================
# Unified Chatbot with Hybrid Classification
# ==========================================

class UnifiedChatbotHybrid:
    """Unified Chatbot ที่ใช้ Hybrid Intent Classification"""
    
    def __init__(self):
        # Initialize hybrid classifier
        self.classifier = HybridIntentClassifier()
        
        # Map intents to chatbot functions
        self.chatbot_map = {}
        
        if ALLPEOPLE_AVAILABLE:
            self.chatbot_map["allpeople"] = {
                "name": "อาจารย์และบุคลากร",
                "qa_function": allpeople_qa,
                "retriever": allpeople_retriever,
                "icon": "👨‍🏫"
            }
        
        if CONTACT_AVAILABLE:
            self.chatbot_map["contact"] = {
                "name": "ข้อมูลติดต่อ",
                "qa_function": contact_qa,
                "retriever": contact_retriever,
                "icon": "📞"
            }
        
        if LINKS_AVAILABLE:
            self.chatbot_map["links"] = {
                "name": "ลิงก์และระบบ",
                "qa_function": links_qa,
                "retriever": links_retriever,
                "icon": "🔗"
            }
        
        if SCHOLARSHIP_AVAILABLE:
            self.chatbot_map["scholarship"] = {
                "name": "ทุนการศึกษา",
                "qa_function": scholarship_qa,
                "retriever": scholarship_retriever,
                "icon": "🎓"
            }
        
        if CLUB_AVAILABLE:
            self.chatbot_map["student_club"] = {
                "name": "สโมสรนักศึกษา",
                "qa_function": club_qa,
                "retriever": club_retriever,
                "icon": "🎭"
            }
        
        if STUDENTS_AVAILABLE:
            self.chatbot_map["students"] = {
                "name": "ลิงก์นักศึกษา",
                "qa_function": students_qa,
                "retriever": students_retriever,
                "icon": "📚"
            }
        
        if RESEARCH_AVAILABLE:
            self.chatbot_map["research"] = {
                "name": "กลุ่มวิจัย",
                "qa_function": research_qa,
                "retriever": research_retriever,
                "icon": "🔬"
            }
        
        if BSC_AVAILABLE:
            self.chatbot_map["bsc_entrance"] = {
                "name": "การรับเข้าศึกษา",
                "qa_function": bsc_qa,
                "retriever": bsc_retriever,
                "icon": "🎓"
            }
        
        if DIGITAL_AVAILABLE:
            self.chatbot_map["digital_services"] = {
                "name": "บริการดิจิตอล",
                "qa_function": digital_qa,
                "retriever": digital_retriever,
                "icon": "💻"
            }
        
        if GRADUATE_AVAILABLE:
            self.chatbot_map["graduate"] = {
                "name": "หลักสูตรบัณฑิตศึกษา",
                "qa_function": graduate_qa,
                "retriever": graduate_retriever,
                "icon": "🎓"
            }
        
        print(f"✅ Unified Chatbot (Hybrid) initialized with {len(self.chatbot_map)} agents")
        for intent, config in self.chatbot_map.items():
            print(f"   {config['icon']} {config['name']}")
    
    def answer(self, question: str) -> str:
        """ตอบคำถามโดยใช้ Hybrid Classification"""
        
        # Step 1: Hybrid Intent Classification
        print(f"\n🔀 กำลังวิเคราะห์คำถามด้วย Hybrid Classification...")
        intent, confidence, method, reason = self.classifier.classify(question)
        
        print(f"\n🎯 Hybrid Intent Classification:")
        print(f"   ประเภท: {intent}")
        print(f"   ความมั่นใจ: {confidence:.2f}")
        print(f"   วิธีการ: {method}")
        print(f"   เหตุผล: {reason}")
        
        # Step 2: Route to appropriate chatbot
        if intent == "unknown":
            # Use multi-agent search
            print(f"❓ ไม่แน่ใจประเภทคำถาม - จะค้นหาจากทุก Agent")
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
    
    def answer_with_contexts(self, question: str) -> tuple:
        """
        ตอบคำถามและ return contexts สำหรับ RAGAS evaluation
        
        Returns:
            tuple: (answer: str, contexts: List[str])
        """
        # Step 1: Hybrid Intent Classification
        intent, confidence, method, reason = self.classifier.classify(question)
        
        contexts = []
        
        # Step 2: Route to appropriate chatbot and get contexts
        if intent == "unknown" or intent not in self.chatbot_map:
            # Multi-agent search - collect contexts from all agents
            for intent_key, config in self.chatbot_map.items():
                try:
                    if "retriever" in config and config["retriever"]:
                        docs = config["retriever"].get_relevant_documents(question)
                        contexts.extend([doc.page_content for doc in docs[:3]])
                except:
                    pass
            
            answer = self.answer(question)
        else:
            # Specific agent - get contexts from that agent
            chatbot_config = self.chatbot_map[intent]
            
            try:
                # Get contexts if retriever available
                if "retriever" in chatbot_config and chatbot_config["retriever"]:
                    docs = chatbot_config["retriever"].get_relevant_documents(question)
                    contexts = [doc.page_content for doc in docs]
                
                # Get answer
                answer = self.answer(question)
            except Exception as e:
                answer = f"Error: {str(e)}"
                contexts = []
        
        # Ensure we have at least some context
        if not contexts:
            contexts = [f"No specific contexts retrieved for: {question}"]
        
        return answer, contexts
    
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
        print("📖 คำแนะนำการใช้งาน Unified Chatbot (Hybrid Version)")
        print("="*60)
        print("\n🔀 ระบบใช้ Hybrid Classification (Rule-Based + LLM)!")
        print("   - ลอง Rule-Based ก่อน (เร็ว, ฟรี)")
        print("   - ถ้าไม่มั่นใจ → ใช้ LLM ช่วย (แม่นยำ)")
        print("\nระบบสามารถตอบคำถามในหัวข้อต่อไปนี้:\n")
        
        for intent, config in self.chatbot_map.items():
            print(f"{config['icon']} {config['name']}")
        
        print("\n💡 ตัวอย่างคำถาม:")
        print("   - อาจารย์สมชาย → อาจารย์และบุคลากร")
        print("   - ติดต่อวิทยาลัย → ข้อมูลติดต่อ")
        print("   - ลิงก์จองห้องประชุม → ลิงก์และระบบ")
        print("   - ทุนการศึกษา → ทุนการศึกษา")
        print("   - ประธานสโมสร → สโมสรนักศึกษา")
        print("   - ลิงก์โครงงาน → ลิงก์นักศึกษา")
        print("   - กลุ่มวิจัย AIDA → กลุ่มวิจัย")
        print("   - รอบ Portfolio → การรับเข้าศึกษา")
        print("   - Web Hosting → บริการดิจิตอล")
        print("   - ปริญญาโท → หลักสูตรบัณฑิตศึกษา")
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
    """Main function สำหรับ Unified Chatbot (Hybrid Version)"""
    
    # Fix encoding for Windows terminal (only when running as main script)
    if sys.platform == "win32":
        import codecs
        try:
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
        except:
            pass  # Already detached
    
    print("\n" + "="*60)
    print("🤖 Unified RAG Chatbot (Hybrid Version) - วิทยาลัยการคอมพิวเตอร์ มข.")
    print("="*60)
    print("🔀 ใช้ Hybrid Classification (Rule-Based + LLM)!")
    print()
    
    # Initialize unified chatbot
    try:
        chatbot = UnifiedChatbotHybrid()
    except Exception as e:
        print(f"\n❌ Error initializing chatbot: {e}")
        return
    
    # Show initial help
    chatbot.show_help()
    
    # Main loop
    while True:
        print("\n" + "-"*60)
        try:
            question = input("\n❓ ถามมาเลย Hybrid Version (หรือพิมพ์ 'help'): ").strip()
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

