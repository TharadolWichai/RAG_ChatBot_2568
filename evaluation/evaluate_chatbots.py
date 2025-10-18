# evaluate_chatbots.py - RAGAS Evaluation Script
# เปรียบเทียบประสิทธิภาพของ 3 Chatbot Versions: Rule-Based, LLM-Based, Hybrid

import sys
import os
import json
import time
from typing import Dict, List, Any
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'main_app'))

from dotenv import load_dotenv

# RAGAS imports
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
    print("[SUCCESS] RAGAS loaded successfully")
except ImportError as e:
    RAGAS_AVAILABLE = False
    print(f"[WARNING] RAGAS import error: {e}")
    print("   Install with: pip install ragas datasets langchain-openai")
except Exception as e:
    RAGAS_AVAILABLE = False
    print(f"[WARNING] RAGAS error: {e}")

load_dotenv()

# Import the 3 chatbot versions
print("[INFO] Loading Chatbot versions...")

try:
    from main_unified_chatbot import UnifiedChatbot
    RULE_BASED_AVAILABLE = True
    print("[SUCCESS] Rule-Based Chatbot loaded")
except Exception as e:
    RULE_BASED_AVAILABLE = False
    print(f"[WARNING] Rule-Based Chatbot not available: {e}")

try:
    from main_unified_chatbot_llm import UnifiedChatbotLLM
    LLM_BASED_AVAILABLE = True
    print("[SUCCESS] LLM-Based Chatbot loaded")
except Exception as e:
    LLM_BASED_AVAILABLE = False
    print(f"[WARNING] LLM-Based Chatbot not available: {e}")

try:
    from main_unified_chatbot_hybrid import UnifiedChatbotHybrid
    HYBRID_AVAILABLE = True
    print("[SUCCESS] Hybrid Chatbot loaded")
except Exception as e:
    HYBRID_AVAILABLE = False
    print(f"[WARNING] Hybrid Chatbot not available: {e}")

print()

# ==========================================
# Test Dataset
# ==========================================

TEST_QUESTIONS = [
    {
        "question": "อาจารย์พุธษดี",
        "expected_intent": "allpeople",
        "ground_truth": "ควรตอบข้อมูลเกี่ยวกับอาจารย์ที่มีชื่อว่าพุธษดี รวมถึงตำแหน่ง คณะ และข้อมูลติดต่อ",
        "category": "simple"
    },
    {
        "question": "ติดต่อวิทยาลัย",
        "expected_intent": "contact",
        "ground_truth": "ควรตอบเบอร์โทรศัพท์ อีเมล ที่อยู่ของวิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น",
        "category": "simple"
    },
    {
        "question": "ลิงก์จองห้องประชุม",
        "expected_intent": "links",
        "ground_truth": "ควรให้ลิงก์สำหรับจองห้องประชุมของวิทยาลัยการคอมพิวเตอร์",
        "category": "simple"
    },
    {
        "question": "ทุนการศึกษามีอะไรบ้าง",
        "expected_intent": "scholarship",
        "ground_truth": "ควรแสดงรายการทุนการศึกษาทั้งหมด เช่น ทุนวิจัย ทุนนานาชาติ ทุน ASEAN & GMS",
        "category": "simple"
    },
    {
        "question": "ประธานสโมสรนักศึกษา",
        "expected_intent": "student_club",
        "ground_truth": "ควรตอบชื่อและข้อมูลของประธานสโมสรนักศึกษาปัจจุบัน",
        "category": "simple"
    },
    {
        "question": "กลุ่มวิจัย AIDA",
        "expected_intent": "research",
        "ground_truth": "ควรตอบข้อมูลเกี่ยวกับกลุ่มวิจัย AIDA (Applied Intelligence and Data Analytics) รวมถึงหัวหน้ากลุ่ม สมาชิก และลิงก์เว็บไซต์",
        "category": "simple"
    },
    {
        "question": "รอบ Portfolio คืออะไร",
        "expected_intent": "bsc_entrance",
        "ground_truth": "ควรอธิบายรอบ Portfolio ของการรับเข้าศึกษาระดับปริญญาตรี รวมถึงเกณฑ์และวิธีสมัคร",
        "category": "simple"
    },
    {
        "question": "Web Hosting",
        "expected_intent": "digital_services",
        "ground_truth": "ควรอธิบายบริการ Web Hosting ของวิทยาลัยการคอมพิวเตอร์ รวมถึงวิธีการใช้งานและข้อตกลง",
        "category": "simple"
    },
    {
        "question": "ขอข้อมูลเกี่ยวกับการสมัคร",
        "expected_intent": "bsc_entrance",
        "ground_truth": "ควรให้ข้อมูลเกี่ยวกับการสมัครเข้าศึกษา รวมถึงรอบต่างๆ และเกณฑ์การรับเข้า",
        "category": "ambiguous"
    },
    {
        "question": "มีบริการอะไรบ้าง",
        "expected_intent": "unknown",
        "ground_truth": "ควรแสดงรายการบริการต่างๆ ของวิทยาลัย เช่น ลิงก์ระบบ บริการดิจิตอล หรือบริการนักศึกษา",
        "category": "ambiguous"
    },
    {
        "question": "Virtual Machine คืออะไร และใช้ยังไง",
        "expected_intent": "digital_services",
        "ground_truth": "ควรอธิบายบริการ Virtual Machine พร้อมวิธีการใช้งานและขั้นตอนการขอใช้บริการ",
        "category": "complex"
    },
    {
        "question": "ทุน ASEAN & GMS มีคุณสมบัติอะไรบ้าง",
        "expected_intent": "scholarship",
        "ground_truth": "ควรระบุคุณสมบัติของผู้สมัครทุน ASEAN & GMS รวมถึงเงื่อนไขและผลประโยชน์ที่ได้รับ",
        "category": "complex"
    }
]

# ==========================================
# Chatbot Wrapper Classes
# ==========================================

class ChatbotWrapper:
    """Base wrapper class for chatbots"""
    
    def __init__(self, name: str, chatbot: Any):
        self.name = name
        self.chatbot = chatbot
    
    def answer(self, question: str) -> Dict[str, Any]:
        """Get answer and metadata from chatbot"""
        start_time = time.time()
        
        try:
            answer = self.chatbot.answer(question)
            elapsed_time = time.time() - start_time
            
            return {
                "answer": answer,
                "time": elapsed_time,
                "success": True,
                "error": None
            }
        except Exception as e:
            elapsed_time = time.time() - start_time
            return {
                "answer": f"Error: {str(e)}",
                "time": elapsed_time,
                "success": False,
                "error": str(e)
            }

# ==========================================
# Evaluation Functions
# ==========================================

def get_contexts_from_chatbot(chatbot: Any, question: str) -> List[str]:
    """
    ดึง contexts (retrieved documents) จาก chatbot
    สำหรับ RAGAS evaluation
    
    Args:
        chatbot: Chatbot instance ที่มี method answer_with_contexts()
        question: คำถาม
        
    Returns:
        List[str]: รายการ contexts ที่ retrieve มาได้
    """
    try:
        # ใช้ method answer_with_contexts() เพื่อดึง contexts จริงๆ
        if hasattr(chatbot, 'answer_with_contexts'):
            _, contexts = chatbot.answer_with_contexts(question)
            return contexts if contexts else [f"No contexts found for: {question}"]
        else:
            # Fallback: chatbot ไม่มี method answer_with_contexts()
            return [f"Chatbot does not support context retrieval for: {question}"]
    except Exception as e:
        print(f"   [WARNING] Failed to get contexts: {e}")
        return [f"Error retrieving contexts: {str(e)}"]

def run_evaluation(test_size: int = None) -> Dict[str, Any]:
    """
    รัน evaluation สำหรับ chatbot ทั้ง 3 versions
    
    Args:
        test_size: จำนวนคำถามที่จะทดสอบ (None = ทั้งหมด)
    
    Returns:
        Dictionary ของผลลัพธ์การ evaluation
    """
    
    if not RAGAS_AVAILABLE:
        print("❌ Cannot run evaluation without RAGAS")
        return None
    
    # เลือกคำถามทดสอบ
    questions = TEST_QUESTIONS[:test_size] if test_size else TEST_QUESTIONS
    print(f"\n📝 กำลังทดสอบด้วย {len(questions)} คำถาม")
    print("="*60)
    
    results = {}
    
    # Test each chatbot version
    chatbot_configs = []
    
    if RULE_BASED_AVAILABLE:
        print("\n🔧 กำลังเตรียม Rule-Based Chatbot...")
        rule_chatbot = UnifiedChatbot()
        chatbot_configs.append(("Rule-Based", rule_chatbot))
    
    if LLM_BASED_AVAILABLE:
        print("\n🤖 กำลังเตรียม LLM-Based Chatbot...")
        llm_chatbot = UnifiedChatbotLLM()
        chatbot_configs.append(("LLM-Based", llm_chatbot))
    
    if HYBRID_AVAILABLE:
        print("\n🔀 กำลังเตรียม Hybrid Chatbot...")
        hybrid_chatbot = UnifiedChatbotHybrid()
        chatbot_configs.append(("Hybrid", hybrid_chatbot))
    
    # Evaluate each chatbot
    for chatbot_name, chatbot in chatbot_configs:
        print(f"\n{'='*60}")
        print(f"🧪 Evaluating: {chatbot_name}")
        print(f"{'='*60}")
        
        wrapper = ChatbotWrapper(chatbot_name, chatbot)
        
        # Collect answers and metadata
        answers = []
        contexts_list = []
        response_times = []
        errors = []
        
        for i, test_case in enumerate(questions, 1):
            question = test_case["question"]
            print(f"\n[{i}/{len(questions)}] {question}")
            
            # Get answer
            result = wrapper.answer(question)
            answers.append(result["answer"])
            response_times.append(result["time"])
            
            if not result["success"]:
                errors.append({
                    "question": question,
                    "error": result["error"]
                })
            
            # Get contexts (for RAGAS)
            contexts = get_contexts_from_chatbot(chatbot, question)
            contexts_list.append(contexts)
            
            print(f"   ⏱️  Time: {result['time']:.2f}s")
            print(f"   📏 Answer length: {len(result['answer'])} chars")
        
        # Prepare dataset for RAGAS
        eval_data = {
            "question": [tc["question"] for tc in questions],
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": [tc["ground_truth"] for tc in questions]
        }
        
        dataset = Dataset.from_dict(eval_data)
        
        # Run RAGAS evaluation
        print(f"\n[INFO] Running RAGAS evaluation...")
        
        # RAGAS needs OpenAI API key - try to use OPENROUTER_API_KEY if OPENAI_API_KEY not set
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            # Try using OPENROUTER_API_KEY
            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            if openrouter_key:
                print("   [WARNING] OPENAI_API_KEY not found, using OPENROUTER_API_KEY for RAGAS...")
                print("   [INFO] Configuring RAGAS to use OpenRouter...")
                os.environ["OPENAI_API_KEY"] = openrouter_key
                os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
            else:
                print("   [ERROR] No API key found for RAGAS evaluation")
                print("   Please set OPENAI_API_KEY or OPENROUTER_API_KEY in .env")
                raise ValueError("API key required for RAGAS evaluation")
        else:
            # OPENAI_API_KEY exists, check if it's OpenRouter format
            if openai_api_key.startswith("sk-or-v1-"):
                print("   [INFO] Detected OpenRouter API key format")
                print("   [INFO] Configuring RAGAS to use OpenRouter...")
                os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
            else:
                print("   [INFO] Detected native OpenAI API key")
        
        try:
            # Use specific LLM configuration for better stability
            from langchain_openai import ChatOpenAI
            
            # Get API configuration
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
            
            # Configure LLM with explicit settings for OpenRouter
            # Using GPT-3.5-turbo for better compatibility with RAGAS
            llm = ChatOpenAI(
                model="openai/gpt-3.5-turbo",  # OpenRouter format
                temperature=0.1,
                api_key=api_key,
                base_url=base_url,
                default_headers={
                    "HTTP-Referer": "https://github.com/ChatBot_RAG_CS_KKU",
                    "X-Title": "CS_KKU_RAG_Evaluation"
                }
            )
            
            print(f"   [DEBUG] Using model: openai/gpt-3.5-turbo")
            print(f"   [DEBUG] Base URL: {base_url}")
            print(f"   [DEBUG] API key starts with: {api_key[:10]}...")
            
            # Use HuggingFace embeddings instead (local, no API needed)
            from langchain_huggingface import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            print(f"   [DEBUG] Embeddings configured: HuggingFace (local)")
            print(f"   [INFO] Starting RAGAS evaluation with GPT-3.5-turbo...")
            
            ragas_results = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall
                ],
                llm=llm,
                embeddings=embeddings  # Use our configured embeddings
            )
            
            print(f"   [SUCCESS] RAGAS evaluation completed!")
            
            # Store results - handle RAGAS results format
            def safe_float(value):
                """Safely convert RAGAS result to float"""
                if isinstance(value, list):
                    # If it's a list, take the first element or average
                    return float(value[0]) if value else 0.0
                elif isinstance(value, (int, float)):
                    return float(value)
                else:
                    return float(str(value)) if str(value) != 'nan' else 0.0
            
            results[chatbot_name] = {
                "ragas_scores": {
                    "faithfulness": safe_float(ragas_results["faithfulness"]),
                    "answer_relevancy": safe_float(ragas_results["answer_relevancy"]),
                    "context_precision": safe_float(ragas_results["context_precision"]),
                    "context_recall": safe_float(ragas_results["context_recall"])
                },
                "performance": {
                    "avg_response_time": sum(response_times) / len(response_times),
                    "min_response_time": min(response_times),
                    "max_response_time": max(response_times),
                    "total_questions": len(questions),
                    "errors": len(errors)
                },
                "errors": errors
            }
            
            # Print results
            print(f"\n[SUCCESS] {chatbot_name} Results:")
            print(f"   Faithfulness:       {safe_float(ragas_results['faithfulness']):.4f}")
            print(f"   Answer Relevancy:   {safe_float(ragas_results['answer_relevancy']):.4f}")
            print(f"   Context Precision:  {safe_float(ragas_results['context_precision']):.4f}")
            print(f"   Context Recall:     {safe_float(ragas_results['context_recall']):.4f}")
            print(f"   Avg Response Time:  {results[chatbot_name]['performance']['avg_response_time']:.2f}s")
            
        except Exception as e:
            print(f"   [ERROR] RAGAS evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            
            results[chatbot_name] = {
                "ragas_scores": None,
                "performance": {
                    "avg_response_time": sum(response_times) / len(response_times),
                    "total_questions": len(questions),
                    "errors": len(errors)
                },
                "error": str(e)
            }
    
    return results

def print_comparison(results: Dict[str, Any]):
    """แสดงผลการเปรียบเทียบแบบตาราง"""
    
    print("\n" + "="*80)
    print("📊 COMPARISON SUMMARY")
    print("="*80)
    
    if not results:
        print("❌ No results to compare")
        return
    
    # Table header
    print(f"\n{'Metric':<25} | {'Rule-Based':<15} | {'LLM-Based':<15} | {'Hybrid':<15}")
    print("-" * 80)
    
    # RAGAS Metrics
    metrics = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall"
    ]
    
    for metric in metrics:
        row = f"{metric.replace('_', ' ').title():<25} |"
        
        for chatbot_name in ["Rule-Based", "LLM-Based", "Hybrid"]:
            if chatbot_name in results and results[chatbot_name].get("ragas_scores"):
                score = results[chatbot_name]["ragas_scores"].get(metric, 0.0)
                row += f" {score:>13.4f} |"
            else:
                row += f" {'N/A':>13} |"
        
        print(row)
    
    print("-" * 80)
    
    # Performance Metrics
    row = f"{'Avg Response Time (s)':<25} |"
    for chatbot_name in ["Rule-Based", "LLM-Based", "Hybrid"]:
        if chatbot_name in results:
            time_val = results[chatbot_name]["performance"]["avg_response_time"]
            row += f" {time_val:>13.2f} |"
        else:
            row += f" {'N/A':>13} |"
    print(row)
    
    row = f"{'Errors':<25} |"
    for chatbot_name in ["Rule-Based", "LLM-Based", "Hybrid"]:
        if chatbot_name in results:
            errors = results[chatbot_name]["performance"]["errors"]
            row += f" {errors:>13} |"
        else:
            row += f" {'N/A':>13} |"
    print(row)
    
    print("="*80)
    
    # Find best performer
    print("\n🏆 Best Performers:")
    
    for metric in metrics:
        best_chatbot = None
        best_score = -1
        
        for chatbot_name in ["Rule-Based", "LLM-Based", "Hybrid"]:
            if chatbot_name in results and results[chatbot_name].get("ragas_scores"):
                score = results[chatbot_name]["ragas_scores"].get(metric, 0.0)
                if score > best_score:
                    best_score = score
                    best_chatbot = chatbot_name
        
        if best_chatbot:
            print(f"   {metric.replace('_', ' ').title():<25}: {best_chatbot} ({best_score:.4f})")
    
    # Fastest
    fastest_chatbot = None
    fastest_time = float('inf')
    
    for chatbot_name in ["Rule-Based", "LLM-Based", "Hybrid"]:
        if chatbot_name in results:
            time_val = results[chatbot_name]["performance"]["avg_response_time"]
            if time_val < fastest_time:
                fastest_time = time_val
                fastest_chatbot = chatbot_name
    
    if fastest_chatbot:
        print(f"   {'Fastest Response':<25}: {fastest_chatbot} ({fastest_time:.2f}s)")
    
    print()

def save_results(results: Dict[str, Any], filename: str = None):
    """บันทึกผลลัพธ์เป็น JSON"""
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}.json"
    
    filepath = os.path.join(os.path.dirname(__file__), filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to: {filepath}")

# ==========================================
# Main Program
# ==========================================

def main():
    """Main evaluation program"""
    
    print("\n" + "="*80)
    print("RAGAS Evaluation - Unified Chatbot Comparison")
    print("="*80)
    print("Comparing 3 Versions: Rule-Based vs LLM-Based vs Hybrid")
    print()
    
    if not RAGAS_AVAILABLE:
        print("\n[ERROR] RAGAS not installed")
        print("\n[INFO] Please install required packages:")
        print("   pip install ragas datasets langchain-openai")
        return
    
    # Check available chatbots
    available_count = sum([RULE_BASED_AVAILABLE, LLM_BASED_AVAILABLE, HYBRID_AVAILABLE])
    
    if available_count == 0:
        print("\n[ERROR] No chatbots available for evaluation")
        return
    
    print(f"[SUCCESS] {available_count} chatbot(s) available for evaluation")
    print()
    
    # Ask user for test size
    print("Test Dataset Options:")
    print("   1. Quick test (3 questions)")
    print("   2. Standard test (6 questions)")
    print("   3. Full test (all questions)")
    print()
    
    try:
        choice = input("Select option (1-3) [default: 2]: ").strip()
        
        if choice == "1":
            test_size = 3
        elif choice == "3":
            test_size = None
        else:
            test_size = 6
    except:
        test_size = 6
    
    # Run evaluation
    print(f"\n🚀 Starting evaluation...")
    results = run_evaluation(test_size=test_size)
    
    if results:
        # Show comparison
        print_comparison(results)
        
        # Save results
        try:
            save = input("\n💾 Save results to file? (y/n) [default: y]: ").strip().lower()
            if save != 'n':
                save_results(results)
        except:
            save_results(results)
        
        print("\n✅ Evaluation completed!")
    else:
        print("\n❌ Evaluation failed")

if __name__ == "__main__":
    main()

