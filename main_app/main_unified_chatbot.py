# main_unified_chatbot.py - Unified Multi-Agent RAG Chatbot
# รวมแชทบอททั้งหมดไว้ที่เดียว พร้อม Intent Classification

import sys
import os
import re
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

# PyThaiNLP for Thai text processing
try:
    from pythainlp import word_tokenize
    PYTHAINLP_AVAILABLE = True
except ImportError:
    PYTHAINLP_AVAILABLE = False

# Import all chatbot modules
try:
    from main_allpeople import retriever as allpeople_retriever, manual_qa_chain as allpeople_qa
    ALLPEOPLE_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] AllPeople chatbot not available: {e}")
    ALLPEOPLE_AVAILABLE = False

try:
    from main_contact import retriever as contact_retriever, manual_qa_chain as contact_qa
    CONTACT_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] Contact chatbot not available: {e}")
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
print("\n[INFO] Agent Availability Status:")
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
print()

# ==========================================
# Intent Classification System
# ==========================================

class IntentClassifier:
    """ระบบจำแนกประเภทคำถามแบบ Hybrid (Rule-based + LLM fallback)"""
    
    def __init__(self):
        # Define keyword patterns for each chatbot
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
                    r'\d{3}-\d+',  # Phone pattern
                    r'.*@.*\..*'    # Email pattern
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
    
    def classify(self, query: str) -> Tuple[str, float]:
        """
        จำแนกประเภทคำถาม
        Returns: (intent_name, confidence_score)
        """
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
            
            # 1. Keyword matching (with higher weight)
            for keyword in config["keywords"]:
                if keyword.lower() in query_lower:
                    matched_keywords += 1
                    score += 3.0  # Increased from 1.0
                # Also check in tokens
                elif any(keyword.lower() in token.lower() for token in tokens):
                    matched_keywords += 1
                    score += 2.0  # Increased from 0.5
            
            # 2. Pattern matching
            pattern_matches = 0
            for pattern in config["patterns"]:
                if re.search(pattern, query_lower):
                    pattern_matches += 1
                    score += 2.0  # Increased from 0.8
            
            # Don't normalize if we have matches (to avoid dilution)
            # Only use raw score when matches are found
            if matched_keywords > 0 or pattern_matches > 0:
                # Bonus for multiple matches
                if matched_keywords >= 2:
                    score *= 1.5  # 50% bonus for 2+ keyword matches
                intent_scores[intent] = score
            else:
                intent_scores[intent] = 0.0
        
        # Get best match
        if not intent_scores:
            return "unknown", 0.0
        
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        
        # Return unknown if confidence is too low
        # New scoring: 3 points per keyword, 2 per pattern
        # Threshold: Need at least 5 points (1-2 strong matches)
        if best_intent[1] < 5.0:  # Adjusted for new scoring system
            return "unknown", best_intent[1]
        
        return best_intent

# ==========================================
# Unified Chatbot Router
# ==========================================

class UnifiedChatbot:
    """Unified Chatbot ที่รวมทุก Agent เข้าด้วยกัน"""
    
    def __init__(self, strict_mode=False):
        """
        Initialize Unified Chatbot
        
        Args:
            strict_mode (bool): 
                - False (default): ใช้ multi-agent search เมื่อไม่เจอ keyword (Production Mode)
                - True: ไม่ fallback, ตอบไม่ได้เมื่อไม่เจอ keyword (Evaluation/Strict Mode)
        """
        self.strict_mode = strict_mode
        self.classifier = IntentClassifier()
        
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
        
        print(f"✅ Unified Chatbot initialized with {len(self.chatbot_map)} agents")
        print(f"   Mode: {'🔒 Strict (No Fallback)' if self.strict_mode else '🔓 Normal (With Fallback)'}")
        for intent, config in self.chatbot_map.items():
            print(f"   {config['icon']} {config['name']}")
    
    def answer(self, question: str) -> str:
        """ตอบคำถามโดยเลือก Agent ที่เหมาะสม"""
        
        # Step 1: Classify intent
        intent, confidence = self.classifier.classify(question)
        
        print(f"\n🎯 Intent Classification:")
        print(f"   ประเภท: {intent}")
        print(f"   คะแนน: {confidence:.2f} (threshold: 5.0)")
        print(f"   โหมด: {'🔒 Strict Mode (No Fallback)' if self.strict_mode else '🔓 Normal Mode (With Fallback)'}")
        
        # Step 2: Route to appropriate chatbot
        if intent == "unknown":
            if self.strict_mode:
                # Strict Mode: ไม่ fallback, ตอบไม่ได้
                print(f"❌ [STRICT MODE] Cannot classify intent: No keyword match found")
                return "❌ [Error] Cannot answer: No matching keyword found in question. Unable to classify intent."
            else:
                # Normal Mode: Try all chatbots and return best answer
                print(f"❓ ไม่แน่ใจประเภทคำถาม - จะค้นหาจากทุก Agent")
                return self._multi_agent_search(question)
        
        if intent not in self.chatbot_map:
            if self.strict_mode:
                # Strict Mode: ไม่ fallback
                print(f"❌ [STRICT MODE] Agent '{intent}' ไม่พร้อมใช้งาน")
                return f"❌ [Error] Agent '{intent}' is not available"
            else:
                # Normal Mode: Agent not available - try multi-agent search
                print(f"⚠️ Agent '{intent}' ไม่พร้อมใช้งาน - จะค้นหาจากทุก Agent")
                print(f"   Available agents: {list(self.chatbot_map.keys())}")
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
        # Step 1: Classify intent
        intent, confidence = self.classifier.classify(question)
        
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
                
                # Enhanced check: ดูที่ส่วนท้ายของคำตอบ (บรรทัดสุดท้าย)
                answer_lines = answer.strip().split('\n')
                last_line = answer_lines[-1].lower() if answer_lines else ""
                
                # Check if answer is meaningful (not "ไม่พบข้อมูล")
                is_not_found = any(phrase in last_line for phrase in [
                    "ไม่พบข้อมูล", "ไม่มีข้อมูล", "no data", "not found", 
                    "ขอโทษ", "sorry", "ไม่สามารถ"
                ])
                
                # Also check if answer is too short (likely an error message)
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
            # Single result
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
        print("📖 คำแนะนำการใช้งาน Unified Chatbot")
        print("="*60)
        print("\nระบบสามารถตอบคำถามในหัวข้อต่อไปนี้:\n")
        
        for intent, config in self.chatbot_map.items():
            print(f"{config['icon']} {config['name']}")
            keywords = self.classifier.intent_patterns[intent]["keywords"][:5]
            print(f"   คำสำคัญ: {', '.join(keywords)}")
            print()
        
        print("\n💡 ตัวอย่างคำถาม:")
        print("   - อาจารย์สมชาย (จะเลือก Agent: อาจารย์และบุคลากร)")
        print("   - ติดต่อวิทยาลัย (จะเลือก Agent: ข้อมูลติดต่อ)")
        print("   - ลิงก์จองห้องประชุม (จะเลือก Agent: ลิงก์และระบบ)")
        print("   - ทุนการศึกษา (จะเลือก Agent: ทุนการศึกษา)")
        print("   - ประธานสโมสร (จะเลือก Agent: สโมสรนักศึกษา)")
        print("   - ลิงก์โครงงาน (จะเลือก Agent: ลิงก์นักศึกษา)")
        print("   - กลุ่มวิจัย AIDA (จะเลือก Agent: กลุ่มวิจัย)")
        print("   - รอบ Portfolio (จะเลือก Agent: การรับเข้าศึกษา)")
        print("   - Web Hosting (จะเลือก Agent: บริการดิจิตอล)")
        print("   - ปริญญาโท (จะเลือก Agent: หลักสูตรบัณฑิตศึกษา)")
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
    """Main function สำหรับ Unified Chatbot"""
    
    print("\n" + "="*60)
    print("🤖 Unified RAG Chatbot - วิทยาลัยการคอมพิวเตอร์ มข.")
    print("="*60)
    print("รวมแชทบอททั้งหมดไว้ในที่เดียว พร้อม Auto-Routing!")
    print()
    
    # Initialize unified chatbot
    chatbot = UnifiedChatbot()
    
    # Show initial help
    chatbot.show_help()
    
    # Main loop
    while True:
        print("\n" + "-"*60)
        try:
            question = input("\n❓ ถามมาเลย (หรือพิมพ์ 'help'): ").strip()
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
            print()  # New line for better formatting
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

