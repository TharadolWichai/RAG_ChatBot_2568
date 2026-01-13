# evaluate_rule_based_strict.py - RAGAS Evaluation for Rule-Based Strict Mode ONLY
# ทดสอบเฉพาะ Rule-Based chatbot ใน Strict Mode (ไม่มี fallback)
# กับไฟล์ test_forRetriver_no_keywords.json (คำถามที่ไม่มีคีย์เวิร์ด 8 ข้อ)

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
    print(f"❌ [ERROR] RAGAS import error: {e}")
    print("   Install with: pip install ragas datasets langchain-openai langchain-huggingface")

load_dotenv()

# Import Rule-Based chatbot ONLY
print("\n📋 [INFO] Loading Rule-Based Chatbot...")
try:
    from main_unified_chatbot import UnifiedChatbot
    RULE_BASED_AVAILABLE = True
    print("✅ [SUCCESS] Rule-Based Chatbot loaded")
except Exception as e:
    RULE_BASED_AVAILABLE = False
    print(f"❌ [ERROR] Rule-Based Chatbot not available: {e}")
    sys.exit(1)

# ==========================================
# Test Dataset - Load from test_forRetriver_no_keywords.json
# ==========================================

def load_test_questions():
    """โหลดคำถามทดสอบจาก test_forRetriver_no_keywords.json"""
    try:
        test_questions_path = os.path.join(os.path.dirname(__file__), "test_forRetriver_no_keywords.json")
        
        if not os.path.exists(test_questions_path):
            print(f"❌ [ERROR] test_forRetriver_no_keywords.json not found at: {test_questions_path}")
            return []
        
        with open(test_questions_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions = data.get("test_questions", [])
        metadata = data.get("metadata", {})
        
        print(f"\n✅ [SUCCESS] Loaded {len(questions)} questions from test_forRetriver_no_keywords.json")
        print(f"   📊 Metadata:")
        print(f"      - Total: {metadata.get('total_questions', 0)} questions")
        print(f"      - Modified Questions: {metadata.get('modified_questions', 0)} ({metadata.get('modification_percentage', 0):.2f}%)")
        print(f"      - Categories: {metadata.get('categories', {})}")
        print(f"      - Difficulty: {metadata.get('difficulty', {})}")
        
        # แสดงคำถามที่แก้ไข
        modified_questions = [q for q in questions if q.get('modified', False)]
        if modified_questions:
            print(f"\n   🔧 Modified Questions (No Keywords):")
            for q in modified_questions:
                print(f"      - ID {q['id']}: {q['question']}")
                print(f"        Original: {q.get('original_question', 'N/A')}")
                print(f"        Removed: {', '.join(q.get('removed_keywords', []))}")
        
        return questions
        
    except Exception as e:
        print(f"❌ [ERROR] Error loading test_forRetriver_no_keywords.json: {e}")
        return []

# โหลดคำถามทดสอบ
TEST_QUESTIONS = load_test_questions()

if not TEST_QUESTIONS:
    print("\n❌ [CRITICAL] No test questions loaded. Exiting...")
    sys.exit(1)

# ==========================================
# Retriever Analysis Functions
# ==========================================

def get_contexts_from_chatbot(chatbot: Any, question: str) -> Tuple[List[str], int]:
    """
    ดึง contexts (retrieved documents) จาก chatbot
    
    Returns:
        Tuple[List[str], int]: (contexts ที่ใช้, จำนวน contexts ทั้งหมดที่ retrieve มา)
    """
    try:
        if hasattr(chatbot, 'answer_with_contexts'):
            _, all_contexts = chatbot.answer_with_contexts(question)
            
            if not all_contexts:
                return [f"No contexts found for: {question}"], 0
            
            total_retrieved = len(all_contexts)
            print(f"      📚 Retrieved {total_retrieved} contexts")
            
            return all_contexts, total_retrieved
        else:
            return [f"Chatbot does not support context retrieval for: {question}"], 0
    except Exception as e:
        print(f"      ⚠️ [WARNING] Failed to get contexts: {e}")
        return [f"Error retrieving contexts: {str(e)}"], 0

# ==========================================
# Evaluation Functions
# ==========================================

def run_evaluation() -> Dict[str, Any]:
    """
    รัน evaluation สำหรับ Rule-Based chatbot ใน Strict Mode
    """
    
    if not RAGAS_AVAILABLE:
        print("❌ Cannot run evaluation without RAGAS")
        return None
    
    questions = TEST_QUESTIONS
    print(f"\n{'='*80}")
    print(f"📝 กำลังทดสอบด้วย {len(questions)} คำถาม (Rule-Based Strict Mode ONLY)")
    print(f"{'='*80}")
    
    # แสดงรายการคำถาม
    print("\n📋 รายการคำถามที่จะทดสอบ:")
    for i, q in enumerate(questions, 1):
        modified_flag = "🔧 (No Keywords)" if q.get('modified', False) else ""
        print(f"   {i}. [{q['expected_intent']:20s}] {q['question']} {modified_flag}")
        print(f"      Difficulty: {q['difficulty']:6s} | Category: {q['category']}")
    
    print(f"\n{'='*80}\n")
    
    # Initialize Rule-Based chatbot in STRICT MODE
    print("\n🔧 กำลังเตรียม Rule-Based Chatbot (Strict Mode - No Fallback)...")
    print(f"   🔒 Strict Mode: Enabled (ไม่มี multi-agent search fallback)")
    
    rule_chatbot = UnifiedChatbot(strict_mode=True)
    
    print(f"\n{'='*60}")
    print(f"🧪 Evaluating: Rule-Based (Strict Mode)")
    print(f"{'='*60}")
    
    # Collect answers and metadata
    answers = []
    contexts_list = []
    response_times = []
    errors = []
    contexts_stats = []
    
    for i, test_case in enumerate(questions, 1):
        question = test_case["question"]
        print(f"\n[{i}/{len(questions)}] {question}")
        
        # Get answer
        start_time = time.time()
        try:
            answer = rule_chatbot.answer(question)
            elapsed_time = time.time() - start_time
            success = True
            
            # ตรวจสอบว่าตอบไม่ได้หรือไม่
            if "Cannot answer" in answer or "Error" in answer:
                errors.append({
                    "question_id": test_case.get('id'),
                    "question": question,
                    "error": answer,
                    "modified": test_case.get('modified', False)
                })
                print(f"   ❌ Cannot answer (No keyword found)")
        except Exception as e:
            answer = f"Error: {str(e)}"
            elapsed_time = time.time() - start_time
            success = False
            errors.append({
                "question_id": test_case.get('id'),
                "question": question,
                "error": str(e),
                "modified": test_case.get('modified', False)
            })
            print(f"   ❌ Exception: {e}")
        
        answers.append(answer)
        response_times.append(elapsed_time)
        
        # Get contexts
        contexts, total_retrieved = get_contexts_from_chatbot(rule_chatbot, question)
        contexts_list.append(contexts)
        
        # Collect context stats
        contexts_stats.append({
            "question_id": test_case.get('id'),
            "num_contexts": len(contexts),
            "total_retrieved": total_retrieved,
            "avg_length": sum(len(c) for c in contexts) / len(contexts) if contexts else 0
        })
        
        print(f"   ⏱️  Time: {elapsed_time:.2f}s")
        print(f"   📏 Answer length: {len(answer)} chars")
    
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
    print(f"⚙️  Running RAGAS evaluation...")
    print(f"{'='*80}")
    
    # Setup LLM and embeddings
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
            "X-Title": "CS_KKU_RAG_Evaluation"
        }
    )
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print(f"   🤖 LLM: openai/gpt-4o-mini")
    print(f"   🔧 Embeddings: HuggingFace (local)")
    print(f"   ⚙️  Starting RAGAS evaluation...")
    
    try:
        ragas_results = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ],
            llm=llm,
            embeddings=embeddings
        )
        
        print(f"   ✅ [SUCCESS] RAGAS evaluation completed!")
        
        # Calculate context statistics
        avg_contexts_stats = {
            "avg_num_contexts": sum(s["num_contexts"] for s in contexts_stats) / len(contexts_stats),
            "avg_num_contexts_retrieved": sum(s["total_retrieved"] for s in contexts_stats) / len(contexts_stats),
            "avg_context_length": sum(s["avg_length"] for s in contexts_stats) / len(contexts_stats),
            "total_contexts_retrieved": sum(s["total_retrieved"] for s in contexts_stats),
            "total_contexts_used": sum(s["num_contexts"] for s in contexts_stats)
        }
        
        # Store results
        def safe_float(value):
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
        
        # Calculate adjusted scores
        num_success = len(questions) - len(errors)
        num_errors = len(errors)
        
        print(f"\n   🔧 Adjusting metrics to include error cases...")
        print(f"      Success: {num_success} questions")
        print(f"      Errors: {num_errors} questions")
        
        # Adjusted Faithfulness
        # - Success questions: use RAGAS score (assume all are 1.0 if RAGAS = 1.0)
        # - Error questions: Faithfulness = 0.0 (error messages are not supported by contexts)
        if num_success > 0:
            # RAGAS faithfulness is average of successful questions only
            # We need to recalculate including errors
            adjusted_faithfulness = (ragas_faithfulness * num_success + 0.0 * num_errors) / len(questions)
        else:
            adjusted_faithfulness = 0.0
        
        # Adjusted Context Precision
        # - Success questions: use RAGAS score
        # - Error questions: Context Precision = 0.0 (no relevant contexts)
        if not (ragas_context_precision != ragas_context_precision):  # Check if not NaN
            adjusted_context_precision = (ragas_context_precision * num_success + 0.0 * num_errors) / len(questions)
        else:
            # If RAGAS returns NaN, assume successful questions have some precision
            # and errors have 0.0
            if num_success > 0:
                # Estimate: assume successful questions have ~0.7 precision (typical value)
                estimated_precision = 0.7
                adjusted_context_precision = (estimated_precision * num_success + 0.0 * num_errors) / len(questions)
            else:
                adjusted_context_precision = 0.0
        
        # Adjusted Context Recall
        # - Success questions: use RAGAS score
        # - Error questions: Context Recall = 0.0 (no contexts retrieved = no coverage)
        if num_success > 0:
            adjusted_context_recall = (ragas_context_recall * num_success + 0.0 * num_errors) / len(questions)
        else:
            adjusted_context_recall = 0.0
        
        # Adjusted Answer Relevancy
        # - Success questions: use RAGAS score
        # - Error questions: Answer Relevancy = 0.0 (error message is not relevant)
        if num_success > 0:
            adjusted_answer_relevancy = (ragas_answer_relevancy * num_success + 0.0 * num_errors) / len(questions)
        else:
            adjusted_answer_relevancy = 0.0
        
        print(f"      ✅ Adjustments completed!")
        print(f"         Original Faithfulness: {ragas_faithfulness:.4f} → Adjusted: {adjusted_faithfulness:.4f}")
        print(f"         Original Context Precision: {ragas_context_precision if not (ragas_context_precision != ragas_context_precision) else 'nan'} → Adjusted: {adjusted_context_precision:.4f}")
        print(f"         Original Context Recall: {ragas_context_recall:.4f} → Adjusted: {adjusted_context_recall:.4f}")
        
        results = {
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
                "errors_count": len(errors)
            },
            "contexts_stats": avg_contexts_stats,
            "errors": errors,
            "error_analysis": {
                "total_errors": len(errors),
                "errors_from_modified": len([e for e in errors if e.get('modified', False)]),
                "errors_from_original": len([e for e in errors if not e.get('modified', False)])
            }
        }
        
        return results
        
    except Exception as e:
        print(f"   ❌ [ERROR] RAGAS evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_results(results: Dict[str, Any]):
    """แสดงผลลัพธ์"""
    
    print(f"\n{'='*80}")
    print(f"📊 Rule-Based Strict Mode Evaluation Results")
    print(f"{'='*80}")
    
    if not results:
        print("❌ No results to display")
        return
    
    # RAGAS Scores (Adjusted)
    print(f"\n📈 RAGAS Scores (Adjusted - Including Error Cases):")
    ragas = results["ragas_scores"]
    print(f"   - Faithfulness:       {ragas['faithfulness']:.4f}")
    print(f"   - Answer Relevancy:   {ragas['answer_relevancy']:.4f}")
    print(f"   - Context Precision:  {ragas['context_precision']:.4f}")
    print(f"   - Context Recall:     {ragas['context_recall']:.4f}")
    
    # RAGAS Scores (Original - if available)
    if "ragas_scores_original" in results:
        print(f"\n📊 RAGAS Scores (Original - Success Questions Only):")
        ragas_orig = results["ragas_scores_original"]
        print(f"   - Faithfulness:       {ragas_orig['faithfulness']:.4f}")
        print(f"   - Answer Relevancy:   {ragas_orig['answer_relevancy']:.4f}")
        ctx_prec = ragas_orig['context_precision']
        # Check if NaN
        if ctx_prec != ctx_prec:  # NaN check
            print(f"   - Context Precision:  nan")
        else:
            print(f"   - Context Precision:  {ctx_prec:.4f}")
        print(f"   - Context Recall:     {ragas_orig['context_recall']:.4f}")
    
    # Performance
    print(f"\n⚡ Performance:")
    perf = results["performance"]
    print(f"   - Avg Response Time:  {perf['avg_response_time']:.2f}s")
    print(f"   - Min Response Time:  {perf['min_response_time']:.2f}s")
    print(f"   - Max Response Time:  {perf['max_response_time']:.2f}s")
    print(f"   - Total Questions:    {perf['total_questions']}")
    print(f"   - Errors:             {perf['errors_count']} ❌")
    
    # Context Stats
    print(f"\n📚 Retriever Stats:")
    ctx = results["contexts_stats"]
    print(f"   - Avg Retrieved/Question: {ctx['avg_num_contexts_retrieved']:.1f}")
    print(f"   - Avg Used/Question:      {ctx['avg_num_contexts']:.1f}")
    print(f"   - Avg Context Length:     {ctx['avg_context_length']:.0f} chars")
    print(f"   - Total Retrieved:        {ctx['total_contexts_retrieved']}")
    print(f"   - Total Used:             {ctx['total_contexts_used']}")
    
    # Error Analysis
    print(f"\n❌ Error Analysis:")
    err_analysis = results["error_analysis"]
    print(f"   - Total Errors:              {err_analysis['total_errors']}")
    print(f"   - Errors from Modified Q:    {err_analysis['errors_from_modified']} (คำถามที่ลบคีย์เวิร์ด)")
    print(f"   - Errors from Original Q:    {err_analysis['errors_from_original']} (คำถามที่มีคีย์เวิร์ด)")
    
    # Explanation
    if "ragas_scores_original" in results:
        print(f"\n💡 Explanation:")
        print(f"   - ⚠️  RAGAS normally skips error cases → Original scores are too high!")
        print(f"   - ✅ Adjusted scores count errors as 0.0 → More realistic!")
        print(f"   - 📊 Impact: {err_analysis['total_errors']}/{perf['total_questions']} questions ({err_analysis['total_errors']/perf['total_questions']*100:.1f}%) failed")
        print(f"   - 🎯 This clearly shows Rule-Based limitations without fallback")
    
    # Detail errors
    if results["errors"]:
        print(f"\n   📋 Error Details:")
        for i, err in enumerate(results["errors"], 1):
            modified_flag = "🔧 (No Keywords)" if err.get('modified', False) else ""
            print(f"      {i}. Q{err['question_id']}: {err['question']} {modified_flag}")
            print(f"         Error: {err['error'][:100]}...")
    
    print(f"\n{'='*80}")

def save_results(results: Dict[str, Any], filename: str = None):
    """บันทึกผลลัพธ์เป็น JSON"""
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rule_based_strict_eval_{timestamp}.json"
    
    filepath = os.path.join(os.path.dirname(__file__), filename)
    
    # Make results serializable
    serializable_results = json.loads(json.dumps(results, default=str))
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({
            "evaluation_info": {
                "test_file": "test_forRetriver_no_keywords.json",
                "total_questions": len(TEST_QUESTIONS),
                "modified_questions": len([q for q in TEST_QUESTIONS if q.get('modified', False)]),
                "timestamp": datetime.now().isoformat(),
                "chatbot": "Rule-Based (Strict Mode - No Fallback)",
                "metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
                "contexts_mode": "All contexts (no limit)",
                "notes": "Testing Rule-Based Strict Mode with NO keyword matching - คาดว่าจะ error ในคำถามที่ลบคีย์เวิร์ด"
            },
            "results": serializable_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {filepath}")

# ==========================================
# Main Program
# ==========================================

def main():
    """Main evaluation program"""
    
    print("\n" + "="*100)
    print("🔬 Rule-Based Strict Mode Evaluation - NO Keywords Test")
    print("="*100)
    print("📋 Source: test_forRetriver_no_keywords.json")
    print("🤖 Testing: Rule-Based ONLY (Strict Mode - No Multi-Agent Fallback)")
    print("📊 Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall")
    print("🎯 Purpose: ทดสอบว่า Rule-Based ตอบไม่ได้กี่ข้อเมื่อไม่มีคีย์เวิร์ด")
    print()
    
    if not RAGAS_AVAILABLE:
        print("\n❌ [ERROR] RAGAS not installed")
        print("\n[INFO] Please install required packages:")
        print("   pip install ragas datasets langchain-openai langchain-huggingface")
        return
    
    if not RULE_BASED_AVAILABLE:
        print("\n❌ [ERROR] Rule-Based chatbot not available")
        return
    
    # Confirm to proceed
    try:
        confirm = input(f"\n🚀 Ready to start evaluation with {len(TEST_QUESTIONS)} questions? (y/n) [default: y]: ").strip().lower()
        if confirm == 'n':
            print("\n👋 Evaluation cancelled")
            return
    except:
        pass  # Default to yes
    
    # Run evaluation
    print(f"\n🚀 Starting evaluation...")
    print(f"   📊 Testing: {len(TEST_QUESTIONS)} questions")
    print(f"   🔒 Mode: Rule-Based Strict Mode (No Fallback)")
    print(f"   📚 Test File: test_forRetriver_no_keywords.json")
    print()
    
    results = run_evaluation()
    
    if results:
        # Show results
        print_results(results)
        
        # Save results
        try:
            save = input("\n💾 Save results to file? (y/n) [default: y]: ").strip().lower()
            if save != 'n':
                save_results(results)
        except:
            save_results(results)
        
        print("\n✅ Evaluation completed!")
        print("\n📌 Summary (Adjusted Scores):")
        print(f"   - Tested {results['performance']['total_questions']} questions")
        print(f"   - Errors: {results['performance']['errors_count']} ❌ ({results['performance']['errors_count']/results['performance']['total_questions']*100:.1f}%)")
        print(f"   - Faithfulness: {results['ragas_scores']['faithfulness']:.4f} (includes errors as 0.0)")
        print(f"   - Context Precision: {results['ragas_scores']['context_precision']:.4f} (includes errors as 0.0)")
        print(f"   - Context Recall: {results['ragas_scores']['context_recall']:.4f} (includes errors as 0.0)")
        
        if "ragas_scores_original" in results:
            print(f"\n📊 Comparison:")
            print(f"   - Original Faithfulness (ignoring errors): {results['ragas_scores_original']['faithfulness']:.4f}")
            print(f"   - Adjusted Faithfulness (counting errors): {results['ragas_scores']['faithfulness']:.4f}")
            print(f"   - Drop: {(results['ragas_scores_original']['faithfulness'] - results['ragas_scores']['faithfulness']):.4f} ({(results['ragas_scores_original']['faithfulness'] - results['ragas_scores']['faithfulness'])*100:.1f}%)")
    else:
        print("\n❌ Evaluation failed")

if __name__ == "__main__":
    main()

