# evaluate_chatbots_no_emoji.py - RAGAS Evaluation Script (No Emoji Version)
# สำหรับ Windows Terminal ที่ไม่รองรับ Unicode

import sys
import os
import time
import json
from typing import Dict, List
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'main_app'))

# RAGAS imports
try:
    import ragas
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

def load_test_questions():
    """Load test questions from JSON file"""
    try:
        with open('test_questions.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load test questions: {e}")
        return []

# ==========================================
# Evaluation Functions
# ==========================================

def run_evaluation(chatbot_name: str, chatbot_instance, questions: List[Dict]) -> Dict:
    """Run RAGAS evaluation on a chatbot"""
    
    print(f"\n[INFO] Evaluating {chatbot_name}...")
    
    responses = []
    response_times = []
    contexts = []
    errors = []
    
    # Get responses from chatbot
    for i, question_data in enumerate(questions, 1):
        question = question_data["question"]
        print(f"   Question {i}/{len(questions)}: {question[:50]}...")
        
        try:
            start_time = time.time()
            response = chatbot_instance.answer(question)
            response_time = time.time() - start_time
            
            responses.append(response)
            response_times.append(response_time)
            contexts.append([question_data.get("ground_truth", "")])
            
            print(f"      Response time: {response_time:.2f}s")
            
        except Exception as e:
            print(f"      [ERROR] {e}")
            responses.append("")
            response_times.append(0)
            contexts.append([""])
            errors.append(f"Question {i}: {e}")
    
    # Create dataset for RAGAS
    eval_data = {
        "question": [q["question"] for q in questions],
        "answer": responses,
        "contexts": contexts,
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
            os.environ["OPENAI_API_KEY"] = openrouter_key
            os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
        else:
            print("   [ERROR] No API key found for RAGAS evaluation")
            print("   Please set OPENAI_API_KEY or OPENROUTER_API_KEY in .env")
            raise ValueError("API key required for RAGAS evaluation")
    
    try:
        ragas_results = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ],
            llm=None,  # Let RAGAS use default LLM
            embeddings=None  # Let RAGAS use default embeddings
        )
        
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
        
        results = {
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
        print(f"   Avg Response Time:  {results['performance']['avg_response_time']:.2f}s")
        
        return results
        
    except Exception as e:
        print(f"   [ERROR] RAGAS evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        
        results = {
            "ragas_scores": None,
            "performance": {
                "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
                "min_response_time": min(response_times) if response_times else 0,
                "max_response_time": max(response_times) if response_times else 0,
                "total_questions": len(questions),
                "errors": len(errors)
            },
            "errors": errors
        }
        
        return results

def print_comparison_summary(results: Dict[str, Dict]):
    """Print comparison summary"""
    
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    # Table header
    print(f"{'Metric':<20} | {'Rule-Based':<15} | {'LLM-Based':<15} | {'Hybrid':<15}")
    print("-" * 80)
    
    # RAGAS metrics
    metrics = ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"]
    for metric in metrics:
        rule_score = results.get("Rule-Based", {}).get("ragas_scores", {}).get(metric.lower().replace(" ", "_"), "N/A")
        llm_score = results.get("LLM-Based", {}).get("ragas_scores", {}).get(metric.lower().replace(" ", "_"), "N/A")
        hybrid_score = results.get("Hybrid", {}).get("ragas_scores", {}).get(metric.lower().replace(" ", "_"), "N/A")
        
        if rule_score != "N/A":
            rule_score = f"{rule_score:.4f}"
        if llm_score != "N/A":
            llm_score = f"{llm_score:.4f}"
        if hybrid_score != "N/A":
            hybrid_score = f"{hybrid_score:.4f}"
            
        print(f"{metric:<20} | {rule_score:>13} | {llm_score:>13} | {hybrid_score:>13} |")
    
    print("-" * 80)
    
    # Performance metrics
    rule_time = results.get("Rule-Based", {}).get("performance", {}).get("avg_response_time", "N/A")
    llm_time = results.get("LLM-Based", {}).get("performance", {}).get("avg_response_time", "N/A")
    hybrid_time = results.get("Hybrid", {}).get("performance", {}).get("avg_response_time", "N/A")
    
    if rule_time != "N/A":
        rule_time = f"{rule_time:.2f}s"
    if llm_time != "N/A":
        llm_time = f"{llm_time:.2f}s"
    if hybrid_time != "N/A":
        hybrid_time = f"{hybrid_time:.2f}s"
    
    print(f"{'Avg Response Time (s)':<20} | {rule_time:>13} | {llm_time:>13} | {hybrid_time:>13} |")
    
    rule_errors = results.get("Rule-Based", {}).get("performance", {}).get("errors", "N/A")
    llm_errors = results.get("LLM-Based", {}).get("performance", {}).get("errors", "N/A")
    hybrid_errors = results.get("Hybrid", {}).get("performance", {}).get("errors", "N/A")
    
    print(f"{'Errors':<20} | {rule_errors:>13} | {llm_errors:>13} | {hybrid_errors:>13} |")
    print("="*80)
    
    # Find best performers
    print("\n[BEST] Best Performers:")
    
    # Fastest response time
    times = []
    if rule_time != "N/A":
        times.append(("Rule-Based", float(rule_time.replace("s", ""))))
    if llm_time != "N/A":
        times.append(("LLM-Based", float(llm_time.replace("s", ""))))
    if hybrid_time != "N/A":
        times.append(("Hybrid", float(hybrid_time.replace("s", ""))))
    
    if times:
        fastest = min(times, key=lambda x: x[1])
        print(f"   Fastest Response         : {fastest[0]} ({fastest[1]:.2f}s)")

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
        choice = input("Select test size (1-3): ").strip()
        if choice == "1":
            num_questions = 3
        elif choice == "2":
            num_questions = 6
        elif choice == "3":
            num_questions = None  # All questions
        else:
            print("[WARNING] Invalid choice, using quick test (3 questions)")
            num_questions = 3
    except:
        print("[WARNING] Using quick test (3 questions)")
        num_questions = 3
    
    # Load test questions
    questions = load_test_questions()
    if not questions:
        print("[ERROR] No test questions available")
        return
    
    if num_questions:
        questions = questions[:num_questions]
    
    print(f"\n[INFO] Using {len(questions)} questions for evaluation")
    print()
    
    # Run evaluations
    results = {}
    
    if RULE_BASED_AVAILABLE:
        try:
            chatbot = UnifiedChatbot()
            results["Rule-Based"] = run_evaluation("Rule-Based", chatbot, questions)
        except Exception as e:
            print(f"[ERROR] Rule-Based evaluation failed: {e}")
    
    if LLM_BASED_AVAILABLE:
        try:
            chatbot = UnifiedChatbotLLM()
            results["LLM-Based"] = run_evaluation("LLM-Based", chatbot, questions)
        except Exception as e:
            print(f"[ERROR] LLM-Based evaluation failed: {e}")
    
    if HYBRID_AVAILABLE:
        try:
            chatbot = UnifiedChatbotHybrid()
            results["Hybrid"] = run_evaluation("Hybrid", chatbot, questions)
        except Exception as e:
            print(f"[ERROR] Hybrid evaluation failed: {e}")
    
    # Print comparison
    if results:
        print_comparison_summary(results)
        
        # Ask to save results
        try:
            save_choice = input("\nSave results to file? (y/n) [default: y]: ").strip().lower()
            if save_choice != 'n':
                filename = f"evaluation_results_{int(time.time())}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"[SUCCESS] Results saved to {filename}")
        except:
            pass
        
        print("\n[SUCCESS] Evaluation completed!")
    else:
        print("\n[ERROR] No evaluation results available")

if __name__ == "__main__":
    main()
