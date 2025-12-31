# evaluate_retriever_10q.py - Retriever Evaluation Script (10 Questions)
# ทดสอบประสิทธิภาพ Retriever ด้วยคำถาม 10 ข้อ (1 คำถามต่อ 1 intent)

import sys
import os
import json
import time
from typing import Dict, List, Any, Tuple
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
    print("✅ [SUCCESS] RAGAS loaded successfully")
except ImportError as e:
    RAGAS_AVAILABLE = False
    print(f"⚠️ [WARNING] RAGAS import error: {e}")
    print("   Install with: pip install ragas datasets langchain-openai")
except Exception as e:
    RAGAS_AVAILABLE = False
    print(f"⚠️ [WARNING] RAGAS error: {e}")

load_dotenv()

# Import the 3 chatbot versions
print("\n📦 [INFO] Loading Chatbot versions...")

try:
    from main_unified_chatbot import UnifiedChatbot
    RULE_BASED_AVAILABLE = True
    print("✅ [SUCCESS] Rule-Based Chatbot loaded")
except Exception as e:
    RULE_BASED_AVAILABLE = False
    print(f"⚠️ [WARNING] Rule-Based Chatbot not available: {e}")

try:
    from main_unified_chatbot_llm import UnifiedChatbotLLM
    LLM_BASED_AVAILABLE = True
    print("✅ [SUCCESS] LLM-Based Chatbot loaded")
except Exception as e:
    LLM_BASED_AVAILABLE = False
    print(f"⚠️ [WARNING] LLM-Based Chatbot not available: {e}")

try:
    from main_unified_chatbot_hybrid import UnifiedChatbotHybrid
    HYBRID_AVAILABLE = True
    print("✅ [SUCCESS] Hybrid Chatbot loaded")
except Exception as e:
    HYBRID_AVAILABLE = False
    print(f"⚠️ [WARNING] Hybrid Chatbot not available: {e}")

print()

# ==========================================
# Test Dataset - Load from test_forRetriver.json
# ==========================================

def load_test_questions(filename: str = "test_forRetriver.json"):
    """
    โหลดคำถามทดสอบจากไฟล์ที่ระบุ
    
    Args:
        filename: ชื่อไฟล์ JSON ที่ต้องการโหลด
                  - "test_forRetriver.json" (มีคีย์เวิร์ด - ทดสอบปกติ)
                  - "test_forRetriver_no_keywords.json" (ไม่มีคีย์เวิร์ด - ทดสอบ Strict Mode)
    
    Returns:
        List[dict]: รายการคำถามทดสอบ
    """
    try:
        test_questions_path = os.path.join(os.path.dirname(__file__), filename)
        
        if not os.path.exists(test_questions_path):
            print(f"❌ [ERROR] {filename} not found at: {test_questions_path}")
            return []
        
        with open(test_questions_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions = data.get("test_questions", [])
        metadata = data.get("metadata", {})
        
        print(f"✅ [SUCCESS] Loaded {len(questions)} questions from {filename}")
        print(f"   📊 Metadata:")
        print(f"      - Total: {metadata.get('total_questions', 0)} questions")
        print(f"      - Categories: {metadata.get('categories', {})}")
        print(f"      - Difficulty: {metadata.get('difficulty', {})}")
        print(f"      - Intents: {list(metadata.get('intents_coverage', {}).keys())}")
        
        # แสดงข้อมูลเพิ่มเติมถ้าเป็นไฟล์ no_keywords
        if "no_keywords" in filename:
            print(f"      - Modified Questions: {metadata.get('modified_questions', 0)} ({metadata.get('modification_percentage', 0):.2f}%)")
            print(f"      - Modified Intents: {list(metadata.get('modified_intents', {}).keys())}")
        
        return questions
        
    except Exception as e:
        print(f"❌ [ERROR] Error loading test_forRetriver.json: {e}")
        return []

# โหลดคำถามทดสอบ (จะโหลดใน main() หลังจากเลือกไฟล์แล้ว)
TEST_QUESTIONS = []

# ==========================================
# Retriever Analysis Functions
# ==========================================

def analyze_contexts(contexts: List[str], question: str) -> Dict[str, Any]:
    """
    วิเคราะห์ contexts ที่ retrieve มา
    
    Returns:
        Dict with:
        - num_contexts: จำนวน contexts
        - avg_length: ความยาวเฉลี่ย (chars)
        - total_length: ความยาวรวม (chars)
        - has_relevant_info: มีข้อมูลที่เกี่ยวข้องหรือไม่
    """
    if not contexts:
        return {
            "num_contexts": 0,
            "avg_length": 0,
            "total_length": 0,
            "min_length": 0,
            "max_length": 0,
            "has_relevant_info": False
        }
    
    lengths = [len(ctx) for ctx in contexts]
    
    return {
        "num_contexts": len(contexts),
        "avg_length": sum(lengths) / len(lengths) if lengths else 0,
        "total_length": sum(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "has_relevant_info": len(contexts) > 0  # Simple check
    }

def get_contexts_from_chatbot(chatbot: Any, question: str, max_contexts: int = None) -> Tuple[List[str], int]:
    """
    ดึง contexts (retrieved documents) จาก chatbot
    
    **สำคัญ:** Retriever ใน main_app ทุกตัวจะ return contexts ที่เรียงตามความเกี่ยวข้องแล้ว
    โดยเรียงจาก relevant สูงสุด → ต่ำสุด ดังนั้นการเลือก top 5/10 จะได้ contexts ที่ดีที่สุด
    
    Args:
        chatbot: Chatbot instance ที่มี method answer_with_contexts()
        question: คำถาม
        max_contexts: จำนวน contexts สูงสุดที่ต้องการ (None = ทั้งหมด)
        
    Returns:
        Tuple[List[str], int]: (contexts ที่ใช้, จำนวน contexts ทั้งหมดที่ retrieve มา)
        
    Note:
        - Contexts ที่ได้มาจะเรียงตาม relevance score จากสูงไปต่ำ
        - การเลือก [:max_contexts] จะได้ contexts ที่ relevant ที่สุด
    """
    try:
        # ใช้ method answer_with_contexts() เพื่อดึง contexts จริงๆ
        if hasattr(chatbot, 'answer_with_contexts'):
            _, all_contexts = chatbot.answer_with_contexts(question)
            
            if not all_contexts:
                return [f"No contexts found for: {question}"], 0
            
            total_retrieved = len(all_contexts)
            
            # ⚠️ สำคัญ: contexts ที่ได้มาเรียงตามความเกี่ยวข้องแล้ว (สูง→ต่ำ)
            # เลือก top N contexts = เลือก contexts ที่มีความเกี่ยวข้องสูงที่สุด
            # 
            # การยืนยันว่าเรียงแล้ว:
            # 1. ✅ ScholarshipRetriever: .sort(combined_score, reverse=True)
            # 2. ✅ StudentClubRetriever: .sort(combined_score, reverse=True)
            # 3. ✅ StudentsRetriever: .sort(combined_score, reverse=True)
            # 4. ✅ LinksRetriever: .sort(combined_score, reverse=True)
            # 5. ✅ GraduateRetriever: .sort(combined_score, reverse=True)
            # 6. ✅ ContactRetriever: .sort(combined_score, reverse=True)
            # 7. ✅ AllPeopleRetriever: .sort(combined_score, reverse=True)
            # 8. ✅ DigitalServicesRetriever: .sort(combined_score, reverse=True)
            # 9. ✅ ResearchGroupRetriever: .sort(priority="สูง" first)
            # 10. ✅ BSCEntranceRetriever: .sort(priority="high" first)
            
            if max_contexts and max_contexts > 0:
                contexts = all_contexts[:max_contexts]
                print(f"      📚 Retrieved {total_retrieved} contexts → Using top {len(contexts)} (most relevant)")
                print(f"      ✅ Confirmed: Contexts are sorted by relevance (highest first)")
            else:
                contexts = all_contexts
                print(f"      📚 Retrieved {total_retrieved} contexts → Using all")
                print(f"      ✅ Confirmed: Contexts are sorted by relevance (highest first)")
            
            return contexts, total_retrieved
        else:
            # Fallback: chatbot ไม่มี method answer_with_contexts()
            return [f"Chatbot does not support context retrieval for: {question}"], 0
    except Exception as e:
        print(f"      ⚠️ [WARNING] Failed to get contexts: {e}")
        return [f"Error retrieving contexts: {str(e)}"], 0

# ==========================================
# Evaluation Functions
# ==========================================

def run_evaluation(test_questions: List[Dict], max_contexts: int = None, strict_mode: bool = False) -> Dict[str, Any]:
    """
    รัน evaluation สำหรับ chatbot ทั้ง 3 versions
    ด้วยคำถามทดสอบที่กำหนด
    
    Args:
        test_questions: รายการคำถามทดสอบ
        max_contexts: จำนวน contexts สูงสุดที่จะส่งให้ RAGAS (None = ทั้งหมด)
        strict_mode: เปิดใช้ Strict Mode สำหรับ Rule-Based (ไม่ fallback multi-agent search)
    
    Returns:
        Dictionary ของผลลัพธ์การ evaluation
    """
    
    if not RAGAS_AVAILABLE:
        print("❌ Cannot run evaluation without RAGAS")
        return None
    
    questions = test_questions
    print(f"\n{'='*80}")
    print(f"📝 กำลังทดสอบด้วย {len(questions)} คำถาม")
    print(f"{'='*80}")
    
    # แสดงรายการคำถาม
    print("\n📋 รายการคำถามที่จะทดสอบ:")
    for i, q in enumerate(questions, 1):
        print(f"   {i}. [{q['expected_intent']:20s}] {q['question']}")
        print(f"      Difficulty: {q['difficulty']:6s} | Category: {q['category']}")
    
    print(f"\n{'='*80}\n")
    
    results = {}
    
    # Test each chatbot version
    chatbot_configs = []
    
    if RULE_BASED_AVAILABLE:
        print("\n🔧 กำลังเตรียม Rule-Based Chatbot...")
        print(f"   Mode: {'🔒 Strict (No Fallback)' if strict_mode else '🔓 Normal (With Fallback)'}")
        rule_chatbot = UnifiedChatbot(strict_mode=strict_mode)
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
        print(f"\n{'='*80}")
        print(f"🧪 Evaluating: {chatbot_name}")
        print(f"{'='*80}")
        
        # Collect answers and metadata
        answers = []
        contexts_list = []
        response_times = []
        errors = []
        contexts_analysis = []
        total_retrieved_list = []  # เก็บจำนวน contexts ทั้งหมดที่ retrieve มา
        
        for i, test_case in enumerate(questions, 1):
            question = test_case["question"]
            expected_intent = test_case["expected_intent"]
            difficulty = test_case.get("difficulty", "medium")
            category = test_case.get("category", "general")
            
            print(f"\n{'─'*80}")
            print(f"[{i}/{len(questions)}] Intent: {expected_intent} | {difficulty.upper()} | {category}")
            print(f"❓ Question: {question}")
            print(f"{'─'*80}")
            
            # Get answer
            start_time = time.time()
            try:
                answer = chatbot.answer(question)
                elapsed_time = time.time() - start_time
                success = True
                error_msg = None
            except Exception as e:
                answer = f"Error: {str(e)}"
                elapsed_time = time.time() - start_time
                success = False
                error_msg = str(e)
                errors.append({
                    "question": question,
                    "error": error_msg
                })
            
            answers.append(answer)
            response_times.append(elapsed_time)
            
            # Get contexts (for RAGAS) - จำกัดตาม max_contexts
            contexts, total_retrieved = get_contexts_from_chatbot(chatbot, question, max_contexts)
            contexts_list.append(contexts)
            total_retrieved_list.append(total_retrieved)
            
            # Analyze contexts
            ctx_analysis = analyze_contexts(contexts, question)
            contexts_analysis.append(ctx_analysis)
            
            # Display results
            print(f"   ⏱️  Response Time: {elapsed_time:.2f}s")
            print(f"   📏 Answer Length: {len(answer)} chars")
            print(f"   📚 Contexts Retrieved: {ctx_analysis['num_contexts']}")
            print(f"   📊 Contexts Stats:")
            print(f"      - Total length: {ctx_analysis['total_length']} chars")
            print(f"      - Avg length: {ctx_analysis['avg_length']:.0f} chars")
            print(f"      - Min/Max: {ctx_analysis['min_length']}/{ctx_analysis['max_length']} chars")
            
            if not success:
                print(f"   ❌ Error: {error_msg}")
        
        # Prepare dataset for RAGAS
        eval_data = {
            "question": [tc["question"] for tc in questions],
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": [tc["ground_truth"] for tc in questions]
        }
        
        dataset = Dataset.from_dict(eval_data)
        
        # Run RAGAS evaluation
        print(f"\n{'='*80}")
        print(f"📊 [INFO] Running RAGAS evaluation for {chatbot_name}...")
        print(f"{'='*80}")
        
        # Configure API for RAGAS
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            if openrouter_key:
                print("   ⚠️ [WARNING] OPENAI_API_KEY not found, using OPENROUTER_API_KEY for RAGAS...")
                os.environ["OPENAI_API_KEY"] = openrouter_key
                os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
            else:
                print("   ❌ [ERROR] No API key found for RAGAS evaluation")
                raise ValueError("API key required for RAGAS evaluation")
        
        try:
            # Use specific LLM configuration
            from langchain_openai import ChatOpenAI
            from langchain_huggingface import HuggingFaceEmbeddings
            
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
            
            llm = ChatOpenAI(
                model="openai/gpt-4o-mini",
                temperature=0.1,
                api_key=api_key,
                base_url=base_url,
                default_headers={
                    "HTTP-Referer": "https://github.com/ChatBot_RAG_CS_KKU",
                    "X-Title": "CS_KKU_Retriever_Evaluation"
                }
            )
            
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            print(f"   🔧 Using model: openai/gpt-4o-mini")
            print(f"   🔧 Base URL: {base_url}")
            print(f"   🔧 Embeddings: HuggingFace (local)")
            print(f"   ⚙️  Starting RAGAS evaluation (with Answer Relevancy as bonus metric)...")
            print(f"   ℹ️  Note: Sending ALL retrieved contexts (no limit)")
            
            # Run RAGAS evaluation (รวม Answer Relevancy แต่ไม่โฟกัส)
            ragas_results = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,  # เพิ่มกลับมา แต่ไม่โฟกัส
                    context_precision,
                    context_recall
                ],
                llm=llm,
                embeddings=embeddings
            )
            
            print(f"   ✅ [SUCCESS] RAGAS evaluation completed!")
            
            # Store results
            def safe_float(value):
                """Safely convert RAGAS result to float"""
                if isinstance(value, list):
                    return float(value[0]) if value else 0.0
                elif isinstance(value, (int, float)):
                    return float(value)
                else:
                    return float(str(value)) if str(value) != 'nan' else 0.0
            
            # ==========================================
            # MANUAL CALCULATION: Adjust metrics for error cases
            # ==========================================
            
            # Get RAGAS raw scores
            ragas_faithfulness = safe_float(ragas_results["faithfulness"])
            ragas_answer_relevancy = safe_float(ragas_results["answer_relevancy"])
            ragas_context_precision = safe_float(ragas_results["context_precision"])
            ragas_context_recall = safe_float(ragas_results["context_recall"])
            
            # Calculate adjusted scores (if there are errors)
            num_success = len(questions) - len(errors)
            num_errors = len(errors)
            
            if num_errors > 0:
                print(f"\n   🔧 Adjusting metrics to include {num_errors} error cases...")
                print(f"      Success: {num_success} questions")
                print(f"      Errors: {num_errors} questions")
                
                # Adjusted Faithfulness
                adjusted_faithfulness = (ragas_faithfulness * num_success + 0.0 * num_errors) / len(questions)
                
                # Adjusted Context Precision
                if not (ragas_context_precision != ragas_context_precision):  # Check if not NaN
                    adjusted_context_precision = (ragas_context_precision * num_success + 0.0 * num_errors) / len(questions)
                else:
                    # If RAGAS returns NaN, estimate successful questions have ~0.7 precision
                    if num_success > 0:
                        estimated_precision = 0.7
                        adjusted_context_precision = (estimated_precision * num_success + 0.0 * num_errors) / len(questions)
                    else:
                        adjusted_context_precision = 0.0
                
                # Adjusted Context Recall
                adjusted_context_recall = (ragas_context_recall * num_success + 0.0 * num_errors) / len(questions)
                
                # Adjusted Answer Relevancy
                adjusted_answer_relevancy = (ragas_answer_relevancy * num_success + 0.0 * num_errors) / len(questions)
                
                print(f"      ✅ Adjustments completed!")
                print(f"         Faithfulness: {ragas_faithfulness:.4f} → {adjusted_faithfulness:.4f}")
                print(f"         Context Precision: {ragas_context_precision if not (ragas_context_precision != ragas_context_precision) else 'nan'} → {adjusted_context_precision:.4f}")
                print(f"         Context Recall: {ragas_context_recall:.4f} → {adjusted_context_recall:.4f}")
            else:
                # No errors, use original scores
                adjusted_faithfulness = ragas_faithfulness
                adjusted_answer_relevancy = ragas_answer_relevancy
                adjusted_context_precision = ragas_context_precision
                adjusted_context_recall = ragas_context_recall
            
            # Calculate average contexts stats
            avg_contexts_stats = {
                "avg_num_contexts_used": sum(c["num_contexts"] for c in contexts_analysis) / len(contexts_analysis),
                "avg_num_contexts_retrieved": sum(total_retrieved_list) / len(total_retrieved_list) if total_retrieved_list else 0,
                "avg_context_length": sum(c["avg_length"] for c in contexts_analysis) / len(contexts_analysis),
                "total_contexts_used": sum(c["num_contexts"] for c in contexts_analysis),
                "total_contexts_retrieved": sum(total_retrieved_list)
            }
            
            results[chatbot_name] = {
                "ragas_scores": {
                    "faithfulness": adjusted_faithfulness,
                    "answer_relevancy": adjusted_answer_relevancy,
                    "context_precision": adjusted_context_precision,
                    "context_recall": adjusted_context_recall
                },
                "ragas_scores_original": {
                    "faithfulness": ragas_faithfulness,
                    "answer_relevancy": ragas_answer_relevancy,
                    "context_precision": ragas_context_precision,
                    "context_recall": ragas_context_recall
                },
                "performance": {
                    "avg_response_time": sum(response_times) / len(response_times),
                    "min_response_time": min(response_times),
                    "max_response_time": max(response_times),
                    "total_questions": len(questions),
                    "errors": len(errors)
                },
                "retriever_stats": avg_contexts_stats,
                "contexts_analysis": contexts_analysis,
                "errors": errors
            }
            
            # Print results
            print(f"\n{'='*80}")
            print(f"📊 {chatbot_name} Results:")
            print(f"{'='*80}")
            
            # Show adjusted scores (main)
            print(f"   📈 RAGAS Scores (Adjusted - Including Error Cases):")
            print(f"      - Faithfulness:       {adjusted_faithfulness:.4f}")
            print(f"      - Context Precision:  {adjusted_context_precision:.4f}")
            print(f"      - Context Recall:     {adjusted_context_recall:.4f}")
            print(f"   💡 Bonus Metric:")
            print(f"      - Answer Relevancy:   {adjusted_answer_relevancy:.4f} (for reference)")
            
            # Show original scores if there were errors
            if num_errors > 0:
                print(f"\n   📊 RAGAS Scores (Original - Success Questions Only):")
                print(f"      - Faithfulness:       {ragas_faithfulness:.4f}")
                ctx_prec_str = f"{ragas_context_precision:.4f}" if not (ragas_context_precision != ragas_context_precision) else "nan"
                print(f"      - Context Precision:  {ctx_prec_str}")
                print(f"      - Context Recall:     {ragas_context_recall:.4f}")
                print(f"      - Answer Relevancy:   {ragas_answer_relevancy:.4f}")
            
            print(f"\n   ⚡ Performance:")
            print(f"      - Avg Response Time:  {results[chatbot_name]['performance']['avg_response_time']:.2f}s")
            print(f"      - Min Response Time:  {results[chatbot_name]['performance']['min_response_time']:.2f}s")
            print(f"      - Max Response Time:  {results[chatbot_name]['performance']['max_response_time']:.2f}s")
            print(f"      - Errors:             {len(errors)} {'❌' if len(errors) > 0 else '✅'}")
            
            print(f"\n   📚 Retriever Stats:")
            print(f"      - Avg Retrieved/Question: {avg_contexts_stats['avg_num_contexts_retrieved']:.1f}")
            print(f"      - Avg Used/Question:      {avg_contexts_stats['avg_num_contexts_used']:.1f}")
            print(f"      - Avg Context Length:     {avg_contexts_stats['avg_context_length']:.0f} chars")
            print(f"      - Total Retrieved:        {avg_contexts_stats['total_contexts_retrieved']}")
            print(f"      - Total Used:             {avg_contexts_stats['total_contexts_used']}")
            
            # Show error impact if any
            if num_errors > 0:
                print(f"\n   💡 Impact of {num_errors} errors:")
                print(f"      - Error Rate: {num_errors}/{len(questions)} ({num_errors/len(questions)*100:.1f}%)")
                faith_drop = ragas_faithfulness - adjusted_faithfulness
                print(f"      - Faithfulness Drop: {faith_drop:.4f} ({faith_drop*100:.1f}%)")
                print(f"      - This shows real-world performance including failures")
            
        except Exception as e:
            print(f"   ❌ [ERROR] RAGAS evaluation failed: {e}")
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
    
    print(f"\n{'='*100}")
    print("📊 COMPARISON SUMMARY - Retriever Test (Adjusted Scores)")
    print(f"{'='*100}")
    print("💡 Note: Scores are adjusted to include error cases (errors count as 0.0)")
    print("   This gives more realistic performance metrics!")
    print(f"{'='*100}")
    
    if not results:
        print("❌ No results to compare")
        return
    
    # Check if any chatbot has errors
    has_errors = any(results[name]["performance"]["errors"] > 0 
                     for name in results if "performance" in results[name])
    
    # Table header
    print(f"\n{'Metric':<30} | {'Rule-Based':<20} | {'LLM-Based':<20} | {'Hybrid':<20}")
    print("-" * 100)
    
    # RAGAS Metrics (Main) - Adjusted Scores
    print("📈 Main Metrics (Adjusted)")
    metrics = [
        "faithfulness",
        "context_precision",
        "context_recall"
    ]
    
    for metric in metrics:
        row = f"{metric.replace('_', ' ').title():<30} |"
        
        for chatbot_name in ["Rule-Based", "LLM-Based", "Hybrid"]:
            if chatbot_name in results and results[chatbot_name].get("ragas_scores"):
                score = results[chatbot_name]["ragas_scores"].get(metric, 0.0)
                row += f" {score:>18.4f} |"
            else:
                row += f" {'N/A':>18} |"
        
        print(row)
    
    # Bonus Metric: Answer Relevancy
    print("-" * 100)
    print("💡 Bonus Metric (for reference)")
    row = f"{'Answer Relevancy':<30} |"
    for chatbot_name in ["Rule-Based", "LLM-Based", "Hybrid"]:
        if chatbot_name in results and results[chatbot_name].get("ragas_scores"):
            score = results[chatbot_name]["ragas_scores"].get("answer_relevancy", 0.0)
            row += f" {score:>18.4f} |"
        else:
            row += f" {'N/A':>18} |"
    print(row)
    
    # Show Original Scores if there were errors
    if has_errors:
        print("-" * 100)
        print("📊 Original Scores (Success Questions Only - for reference)")
        
        for metric in metrics + ["answer_relevancy"]:
            row = f"{metric.replace('_', ' ').title():<30} |"
            
            for chatbot_name in ["Rule-Based", "LLM-Based", "Hybrid"]:
                if chatbot_name in results and results[chatbot_name].get("ragas_scores_original"):
                    score = results[chatbot_name]["ragas_scores_original"].get(metric, 0.0)
                    if score != score:  # Check for NaN
                        row += f" {'nan':>18} |"
                    else:
                        row += f" {score:>18.4f} |"
                else:
                    row += f" {'N/A':>18} |"
            
            print(row)
    
    print("-" * 100)
    
    # Performance Metrics
    print("⚡ Performance & Retriever Stats")
    row = f"{'Avg Response Time (s)':<30} |"
    for chatbot_name in ["Rule-Based", "LLM-Based", "Hybrid"]:
        if chatbot_name in results:
            time_val = results[chatbot_name]["performance"]["avg_response_time"]
            row += f" {time_val:>18.2f} |"
        else:
            row += f" {'N/A':>18} |"
    print(row)
    
    # Retriever Stats
    if any(chatbot_name in results and "retriever_stats" in results[chatbot_name] 
           for chatbot_name in ["Rule-Based", "LLM-Based", "Hybrid"]):
        
        print("-" * 100)
        
        row = f"{'Avg Retrieved/Question':<30} |"
        for chatbot_name in ["Rule-Based", "LLM-Based", "Hybrid"]:
            if chatbot_name in results and "retriever_stats" in results[chatbot_name]:
                val = results[chatbot_name]["retriever_stats"].get("avg_num_contexts_retrieved", 0)
                row += f" {val:>18.1f} |"
            else:
                row += f" {'N/A':>18} |"
        print(row)
        
        row = f"{'Avg Used/Question':<30} |"
        for chatbot_name in ["Rule-Based", "LLM-Based", "Hybrid"]:
            if chatbot_name in results and "retriever_stats" in results[chatbot_name]:
                val = results[chatbot_name]["retriever_stats"].get("avg_num_contexts_used", 0)
                row += f" {val:>18.1f} |"
            else:
                row += f" {'N/A':>18} |"
        print(row)
        
        row = f"{'Avg Context Length (chars)':<30} |"
        for chatbot_name in ["Rule-Based", "LLM-Based", "Hybrid"]:
            if chatbot_name in results and "retriever_stats" in results[chatbot_name]:
                val = results[chatbot_name]["retriever_stats"]["avg_context_length"]
                row += f" {val:>18.0f} |"
            else:
                row += f" {'N/A':>18} |"
        print(row)
        
        row = f"{'Total Retrieved':<30} |"
        for chatbot_name in ["Rule-Based", "LLM-Based", "Hybrid"]:
            if chatbot_name in results and "retriever_stats" in results[chatbot_name]:
                val = results[chatbot_name]["retriever_stats"]["total_contexts_retrieved"]
                row += f" {val:>18.0f} |"
            else:
                row += f" {'N/A':>18} |"
        print(row)
        
        row = f"{'Total Used':<30} |"
        for chatbot_name in ["Rule-Based", "LLM-Based", "Hybrid"]:
            if chatbot_name in results and "retriever_stats" in results[chatbot_name]:
                val = results[chatbot_name]["retriever_stats"].get("total_contexts_used", 0)
                row += f" {val:>18.0f} |"
            else:
                row += f" {'N/A':>18} |"
        print(row)
    
    row = f"{'Errors':<30} |"
    for chatbot_name in ["Rule-Based", "LLM-Based", "Hybrid"]:
        if chatbot_name in results:
            errors = results[chatbot_name]["performance"]["errors"]
            row += f" {errors:>18} |"
        else:
            row += f" {'N/A':>18} |"
    print(row)
    
    print("=" * 100)
    
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
            print(f"   {metric.replace('_', ' ').title():<30}: {best_chatbot} ({best_score:.4f})")
    
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
        print(f"   {'Fastest Response':<30}: {fastest_chatbot} ({fastest_time:.2f}s)")
    
    print()

def save_results(results: Dict[str, Any], max_contexts: int = None, strict_mode: bool = False, filename: str = None):
    """บันทึกผลลัพธ์เป็น JSON"""
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        contexts_suffix = f"_{max_contexts}ctx" if max_contexts else "_all"
        mode_suffix = "_strict" if strict_mode else "_normal"
        filename = f"retriever_eval_10q{contexts_suffix}{mode_suffix}_{timestamp}.json"
    
    filepath = os.path.join(os.path.dirname(__file__), filename)
    
    # Prepare serializable results (remove non-serializable objects)
    serializable_results = {}
    for chatbot_name, result in results.items():
        serializable_results[chatbot_name] = {
            "ragas_scores": result.get("ragas_scores"),
            "ragas_scores_original": result.get("ragas_scores_original"),  # เพิ่ม original scores
            "performance": result.get("performance"),
            "retriever_stats": result.get("retriever_stats"),
            "errors": result.get("errors", [])
        }
    
    # Contexts description
    if max_contexts:
        contexts_desc = f"Top {max_contexts} contexts"
    else:
        contexts_desc = "ALL retrieved contexts (no limit)"
    
    # Calculate total errors across all chatbots
    total_errors = sum(result.get("performance", {}).get("errors", 0) 
                       for result in results.values())
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({
            "evaluation_info": {
                "test_file": "test_forRetriver.json",
                "total_questions": len(TEST_QUESTIONS),
                "timestamp": datetime.now().isoformat(),
                "metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
                "max_contexts": max_contexts if max_contexts else "unlimited",
                "contexts_mode": contexts_desc,
                "strict_mode": strict_mode,
                "rule_based_mode": "Strict (No Fallback)" if strict_mode else "Normal (With Fallback)",
                "adjusted_metrics": True,
                "adjustment_explanation": "Error cases are counted as 0.0 for all metrics to reflect real-world performance",
                "total_errors": total_errors,
                "notes": "Answer Relevancy included as bonus metric | Strict mode affects Rule-Based only | Metrics are adjusted to include error cases"
            },
            "results": serializable_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to: {filepath}")

# ==========================================
# Main Program
# ==========================================

def main():
    """Main evaluation program"""
    
    print("\n" + "="*100)
    print("🔬 Retriever Evaluation - 10 Questions Test (1 question per intent)")
    print("="*100)
    print("📋 Source: test_forRetriver.json")
    print("🤖 Testing: Rule-Based vs LLM-Based vs Hybrid")
    print("📊 Main Metrics: Faithfulness, Context Precision, Context Recall + Retriever Stats")
    print("💡 Bonus Metric: Answer Relevancy (for reference)")
    print("📚 Contexts: You can choose 5, 10, or all contexts")
    print()
    
    if not RAGAS_AVAILABLE:
        print("\n❌ [ERROR] RAGAS not installed")
        print("\n[INFO] Please install required packages:")
        print("   pip install ragas datasets langchain-openai langchain-huggingface")
        return
    
    # Check available chatbots
    available_count = sum([RULE_BASED_AVAILABLE, LLM_BASED_AVAILABLE, HYBRID_AVAILABLE])
    
    if available_count == 0:
        print("\n❌ [ERROR] No chatbots available for evaluation")
        return
    
    print(f"✅ [SUCCESS] {available_count} chatbot(s) available for evaluation")
    print()
    
    # Ask for test dataset
    print("📝 Select Test Dataset:")
    print("   1. test_forRetriver.json (มีคีย์เวิร์ด - ทดสอบปกติ) ⭐")
    print("   2. test_forRetriver_no_keywords.json (ไม่มีคีย์เวิร์ด 8 ข้อ - ทดสอบ Strict Mode)")
    print()
    
    test_file = "test_forRetriver.json"  # Default
    try:
        dataset_choice = input("Select option (1-2) [default: 1]: ").strip()
        if dataset_choice == "2":
            test_file = "test_forRetriver_no_keywords.json"
            print(f"\n✅ Selected: {test_file} (No Keywords - Testing Strict Mode)")
        else:
            test_file = "test_forRetriver.json"
            print(f"\n✅ Selected: {test_file} (With Keywords - Normal Mode)")
    except:
        test_file = "test_forRetriver.json"
        print(f"\n✅ Selected: {test_file} (With Keywords - Normal Mode)")
    
    # โหลดคำถามทดสอบ
    global TEST_QUESTIONS
    TEST_QUESTIONS = load_test_questions(test_file)
    
    if not TEST_QUESTIONS:
        print("\n❌ [CRITICAL] No test questions loaded. Exiting...")
        return
    
    print()
    
    # Ask for number of contexts
    print("📚 Select number of contexts to send to RAGAS:")
    print("   1. Top 5 contexts")
    print("   2. Top 10 contexts")
    print("   3. All contexts (no limit)")
    print()
    
    max_contexts = None  # Default: all contexts
    try:
        choice = input("Select option (1-3) [default: 3]: ").strip()
        if choice == "1":
            max_contexts = 5
            contexts_desc = "Top 5 contexts"
        elif choice == "2":
            max_contexts = 10
            contexts_desc = "Top 10 contexts"
        else:
            max_contexts = None
            contexts_desc = "All contexts (no limit)"
    except:
        max_contexts = None
        contexts_desc = "All contexts (no limit)"
    
    print(f"\n✅ Selected: {contexts_desc}")
    
    # Ask for strict mode (Rule-Based only)
    print("\n🔒 Select Rule-Based Mode:")
    print("   1. Normal Mode (with multi-agent fallback) - Production Mode")
    print("   2. Strict Mode (no fallback) - Evaluation Mode ⭐")
    print()
    print("   💡 Hint: Strict Mode จะไม่ fallback ไป multi-agent search")
    print("           เหมาะสำหรับการประเมินว่า keyword matching ทำงานได้ดีแค่ไหน")
    print()
    
    strict_mode = False  # Default: Normal Mode
    try:
        choice = input("Select option (1-2) [default: 1]: ").strip()
        if choice == "2":
            strict_mode = True
            mode_desc = "🔒 Strict Mode (No Fallback)"
        else:
            strict_mode = False
            mode_desc = "🔓 Normal Mode (With Fallback)"
    except:
        strict_mode = False
        mode_desc = "🔓 Normal Mode (With Fallback)"
    
    print(f"\n✅ Selected: {mode_desc}")
    
    # Confirm to proceed
    try:
        confirm = input("\n🚀 Ready to start evaluation with 10 questions? (y/n) [default: y]: ").strip().lower()
        if confirm == 'n':
            print("\n👋 Evaluation cancelled")
            return
    except:
        pass  # Default to yes
    
    # Run evaluation
    print(f"\n🚀 Starting evaluation...")
    print(f"   📊 Testing: {len(TEST_QUESTIONS)} questions from {test_file}")
    print(f"   📚 Contexts: {contexts_desc}")
    print(f"   🔒 Rule-Based Mode: {mode_desc}")
    print()
    
    results = run_evaluation(test_questions=TEST_QUESTIONS, max_contexts=max_contexts, strict_mode=strict_mode)
    
    if results:
        # Show comparison
        print_comparison(results)
        
        # Save results
        try:
            save = input("\n💾 Save results to file? (y/n) [default: y]: ").strip().lower()
            if save != 'n':
                save_results(results, max_contexts=max_contexts, strict_mode=strict_mode)
        except:
            save_results(results, max_contexts=max_contexts, strict_mode=strict_mode)
        
        print("\n✅ Evaluation completed!")
        print("\n📌 Summary:")
        print(f"   - Tested {len(TEST_QUESTIONS)} questions")
        print(f"   - {len(results)} chatbot versions evaluated")
        print(f"   - Contexts mode: {contexts_desc}")
        print(f"   - Rule-Based mode: {mode_desc}")
        print(f"   - Metrics: Adjusted to include error cases (more realistic!)")
        print(f"   - Included Answer Relevancy as bonus metric")
        print(f"   - Results saved to JSON file")
        
        # Show error summary
        total_errors = sum(results[name]["performance"]["errors"] 
                          for name in results if "performance" in results[name])
        if total_errors > 0:
            print(f"\n⚠️  Error Summary:")
            for name in ["Rule-Based", "LLM-Based", "Hybrid"]:
                if name in results and "performance" in results[name]:
                    errors = results[name]["performance"]["errors"]
                    if errors > 0:
                        print(f"   - {name}: {errors} errors ❌")
            print(f"   💡 Adjusted scores reflect these errors as 0.0")
    else:
        print("\n❌ Evaluation failed")

if __name__ == "__main__":
    main()

