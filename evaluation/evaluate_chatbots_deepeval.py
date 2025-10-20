# evaluate_chatbots_deepeval.py - DeepEval Evaluation Script
# เปรียบเทียบประสิทธิภาพของ 3 Chatbot Versions ด้วย DeepEval

import sys
import os
import json
import time
from typing import Dict, List, Any
from datetime import datetime

# Fix Windows encoding for Thai text and emojis
if sys.platform == 'win32':
    import io
    if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    if not isinstance(sys.stderr, io.TextIOWrapper) or sys.stderr.encoding != 'utf-8':
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'main_app'))

from dotenv import load_dotenv

# DeepEval imports
try:
    from deepeval import evaluate
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        FaithfulnessMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric
    )
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
    print("[SUCCESS] DeepEval loaded successfully")
except ImportError as e:
    DEEPEVAL_AVAILABLE = False
    print(f"[WARNING] DeepEval import error: {e}")
    print("   Install with: pip install deepeval")
except Exception as e:
    DEEPEVAL_AVAILABLE = False
    print(f"[WARNING] DeepEval error: {e}")

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
# Test Dataset - Load from test_questions.json
# ==========================================

def load_test_questions():
    """โหลดคำถามทดสอบจากไฟล์ test_questions.json"""
    try:
        test_questions_path = os.path.join(os.path.dirname(__file__), "test_questions.json")
        
        if not os.path.exists(test_questions_path):
            print(f"[WARNING] test_questions.json not found")
            return []
        
        with open(test_questions_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions = data.get("test_questions", [])
        print(f"[SUCCESS] Loaded {len(questions)} questions from test_questions.json")
        return questions
        
    except Exception as e:
        print(f"[WARNING] Error loading test_questions.json: {e}")
        return []

# โหลดคำถามทดสอบ
TEST_QUESTIONS = load_test_questions()

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

def get_contexts_from_chatbot(chatbot: Any, question: str, max_contexts: int = 5) -> List[str]:
    """
    ดึง contexts (retrieved documents) จาก chatbot
    สำหรับ DeepEval evaluation
    """
    try:
        if hasattr(chatbot, 'answer_with_contexts'):
            _, contexts = chatbot.answer_with_contexts(question)
            
            if not contexts:
                return [f"No contexts found for: {question}"]
            
            # Limit contexts
            limited_contexts = contexts[:max_contexts]
            
            if len(contexts) > max_contexts:
                print(f"   🔧 Reduced contexts: {len(contexts)} → {max_contexts}")
            
            return limited_contexts
        else:
            return [f"Chatbot does not support context retrieval"]
    except Exception as e:
        print(f"   [WARNING] Failed to get contexts: {e}")
        return [f"Error retrieving contexts: {str(e)}"]

def run_evaluation(test_size: int = None, max_contexts: int = 5) -> Dict[str, Any]:
    """
    รัน evaluation สำหรับ chatbot ทั้ง 3 versions ด้วย DeepEval
    """
    
    if not DEEPEVAL_AVAILABLE:
        print("❌ Cannot run evaluation without DeepEval")
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
    
    # Get OpenRouter API key for DeepEval
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ Error: OPENROUTER_API_KEY not found in .env")
        return None
    
    # Set up environment for DeepEval to use OpenRouter
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
    
    # Evaluate each chatbot
    for chatbot_name, chatbot in chatbot_configs:
        print(f"\n{'='*60}")
        print(f"🧪 Evaluating: {chatbot_name}")
        print(f"{'='*60}")
        
        wrapper = ChatbotWrapper(chatbot_name, chatbot)
        
        # Collect test cases
        test_cases = []
        response_times = []
        errors = []
        
        for i, test_case_data in enumerate(questions, 1):
            question = test_case_data["question"]
            ground_truth = test_case_data["ground_truth"]
            
            print(f"\n[{i}/{len(questions)}] {question}")
            
            # Get answer
            result = wrapper.answer(question)
            answer = result["answer"]
            response_times.append(result["time"])
            
            if not result["success"]:
                errors.append({
                    "question": question,
                    "error": result["error"]
                })
            
            # Get contexts
            contexts = get_contexts_from_chatbot(chatbot, question, max_contexts=max_contexts)
            
            print(f"   ⏱️  Time: {result['time']:.2f}s")
            print(f"   📏 Answer length: {len(answer)} chars")
            print(f"   📚 Contexts: {len(contexts)}")
            
            # Create DeepEval test case
            test_case = LLMTestCase(
                input=question,
                actual_output=answer,
                expected_output=ground_truth,
                retrieval_context=contexts
            )
            test_cases.append(test_case)
        
        # Define metrics
        print(f"\n[INFO] Running DeepEval evaluation...")
        print(f"   [INFO] Using OpenRouter with gpt-4o-mini")
        
        try:
            # Create metrics
            answer_relevancy = AnswerRelevancyMetric(
                threshold=0.7,
                model="openai/gpt-4o-mini",
                include_reason=True
            )
            
            faithfulness = FaithfulnessMetric(
                threshold=0.7,
                model="openai/gpt-4o-mini",
                include_reason=True
            )
            
            contextual_precision = ContextualPrecisionMetric(
                threshold=0.7,
                model="openai/gpt-4o-mini",
                include_reason=True
            )
            
            contextual_recall = ContextualRecallMetric(
                threshold=0.7,
                model="openai/gpt-4o-mini",
                include_reason=True
            )
            
            # Run evaluation
            eval_results = evaluate(
                test_cases=test_cases,
                metrics=[
                    answer_relevancy,
                    faithfulness,
                    contextual_precision,
                    contextual_recall
                ]
            )
            
            print(f"   [SUCCESS] DeepEval evaluation completed!")
            
            # Calculate average scores
            avg_scores = {
                "answer_relevancy": 0.0,
                "faithfulness": 0.0,
                "contextual_precision": 0.0,
                "contextual_recall": 0.0
            }
            
            for test_case in test_cases:
                for metric_name in avg_scores.keys():
                    metric_score = getattr(test_case, metric_name, 0.0)
                    if metric_score is not None:
                        avg_scores[metric_name] += metric_score
            
            # Average
            num_cases = len(test_cases)
            if num_cases > 0:
                for key in avg_scores:
                    avg_scores[key] /= num_cases
            
            # Store results
            results[chatbot_name] = {
                "deepeval_scores": avg_scores,
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
            print(f"   Answer Relevancy:      {avg_scores['answer_relevancy']:.4f}")
            print(f"   Faithfulness:          {avg_scores['faithfulness']:.4f}")
            print(f"   Contextual Precision:  {avg_scores['contextual_precision']:.4f}")
            print(f"   Contextual Recall:     {avg_scores['contextual_recall']:.4f}")
            print(f"   Avg Response Time:     {results[chatbot_name]['performance']['avg_response_time']:.2f}s")
            
        except Exception as e:
            print(f"   [ERROR] DeepEval evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            
            results[chatbot_name] = {
                "deepeval_scores": None,
                "performance": {
                    "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
                    "total_questions": len(questions),
                    "errors": len(errors)
                },
                "error": str(e)
            }
    
    return results

def print_comparison(results: Dict[str, Any]):
    """แสดงผลการเปรียบเทียบแบบตาราง"""
    
    print("\n" + "="*80)
    print("📊 DEEPEVAL COMPARISON SUMMARY")
    print("="*80)
    
    if not results:
        print("❌ No results to compare")
        return
    
    # Table header
    print(f"\n{'Metric':<25} | {'Rule-Based':<15} | {'LLM-Based':<15} | {'Hybrid':<15}")
    print("-" * 80)
    
    # DeepEval Metrics
    metrics = [
        "answer_relevancy",
        "faithfulness",
        "contextual_precision",
        "contextual_recall"
    ]
    
    metric_display = {
        "answer_relevancy": "Answer Relevancy",
        "faithfulness": "Faithfulness",
        "contextual_precision": "Contextual Precision",
        "contextual_recall": "Contextual Recall"
    }
    
    for metric in metrics:
        row = f"{metric_display[metric]:<25} |"
        
        for chatbot_name in ["Rule-Based", "LLM-Based", "Hybrid"]:
            if chatbot_name in results and results[chatbot_name].get("deepeval_scores"):
                score = results[chatbot_name]["deepeval_scores"].get(metric, 0.0)
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
            if chatbot_name in results and results[chatbot_name].get("deepeval_scores"):
                score = results[chatbot_name]["deepeval_scores"].get(metric, 0.0)
                if score > best_score:
                    best_score = score
                    best_chatbot = chatbot_name
        
        if best_chatbot:
            print(f"   {metric_display[metric]:<25}: {best_chatbot} ({best_score:.4f})")
    
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
        filename = f"deepeval_results_{timestamp}.json"
    
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
    print("🔬 DeepEval Evaluation - Unified Chatbot Comparison")
    print("="*80)
    print("Comparing 3 Versions: Rule-Based vs LLM-Based vs Hybrid")
    print()
    
    if not DEEPEVAL_AVAILABLE:
        print("\n[ERROR] DeepEval not installed")
        print("\n[INFO] Please install required packages:")
        print("   pip install deepeval")
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
    print("   3. Small test (10 questions)")
    print("   4. Medium test (30 questions)")
    print("   5. Full test (all questions)")
    print()
    
    try:
        choice = input("Select option (1-5) [default: 2]: ").strip()
        
        if choice == "1":
            test_size = 3
        elif choice == "3":
            test_size = 10
        elif choice == "4":
            test_size = 30
        elif choice == "5":
            test_size = None
        else:
            test_size = 6
    except:
        test_size = 6
    
    # Ask user for max contexts
    print("\n⚙️  Context Limit Options:")
    print("   1. Ultra Fast mode - 2 contexts 🚀")
    print("   2. Fast mode - 3 contexts")
    print("   3. Balanced mode - 5 contexts ✨")
    print("   4. Full mode - 10 contexts")
    print()
    
    try:
        context_choice = input("Select option (1-4) [default: 1]: ").strip()
        
        if context_choice == "2":
            max_contexts = 3
            print("   🚀 Using 3 contexts - Fast mode")
        elif context_choice == "3":
            max_contexts = 5
            print("   ⚖️  Using 5 contexts - Balanced mode")
        elif context_choice == "4":
            max_contexts = 10
            print("   🐌 Using 10 contexts - Full mode")
        else:
            max_contexts = 2
            print("   🚀🚀 Using 2 contexts - Ultra Fast mode")
    except:
        max_contexts = 2
        print("   🚀🚀 Using 2 contexts - Ultra Fast mode (default)")
    
    # Run evaluation
    print(f"\n🚀 Starting DeepEval evaluation...")
    print(f"   📊 Testing: {test_size if test_size else 'all'} questions")
    print(f"   📚 Max contexts per question: {max_contexts}")
    print()
    results = run_evaluation(test_size=test_size, max_contexts=max_contexts)
    
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

