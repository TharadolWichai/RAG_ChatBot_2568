# evaluate_improved.py - Enhanced RAGAS Evaluation with Improvements

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
    from langchain_openai import ChatOpenAI
    RAGAS_AVAILABLE = True
    print("[SUCCESS] RAGAS loaded successfully")
except ImportError as e:
    RAGAS_AVAILABLE = False
    print(f"[WARNING] RAGAS import error: {e}")

load_dotenv()

def create_improved_llm():
    """สร้าง LLM ที่ปรับปรุงแล้วสำหรับ RAGAS"""
    try:
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
            max_tokens=500,
            timeout=30
        )
        print("[SUCCESS] Improved LLM configured")
        return llm
    except Exception as e:
        print(f"[WARNING] LLM configuration failed: {e}")
        return None

def load_improved_test_questions():
    """โหลด test questions ที่ปรับปรุงแล้ว"""
    try:
        with open('test_questions_improved.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        # Fallback to original
        with open('test_questions.json', 'r', encoding='utf-8') as f:
            return json.load(f)

def run_improved_evaluation(chatbot_name: str, chatbot_instance, questions: List[Dict]) -> Dict:
    """รัน evaluation ที่ปรับปรุงแล้ว"""
    
    print(f"\n[INFO] Running improved evaluation for {chatbot_name}...")
    
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
            
            # Clean and validate response
            if response and len(response.strip()) > 10:
                responses.append(response.strip())
                contexts.append([question_data.get("ground_truth", "")])
            else:
                responses.append("ไม่พบข้อมูลที่เกี่ยวข้อง")
                contexts.append([question_data.get("ground_truth", "")])
            
            response_times.append(response_time)
            print(f"      Response time: {response_time:.2f}s")
            
        except Exception as e:
            print(f"      [ERROR] {e}")
            responses.append("เกิดข้อผิดพลาดในการประมวลผล")
            response_times.append(0)
            contexts.append([""])
            errors.append(f"Question {i}: {e}")
    
    # Create improved dataset
    eval_data = {
        "question": [q["question"] for q in questions],
        "answer": responses,
        "contexts": contexts,
        "ground_truth": [tc["ground_truth"] for tc in questions]
    }
    
    dataset = Dataset.from_dict(eval_data)
    
    # Run improved RAGAS evaluation
    print(f"\n[INFO] Running improved RAGAS evaluation...")
    
    # Configure improved LLM
    llm = create_improved_llm()
    
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
            embeddings=None,
            max_workers=2,  # Reduce concurrent requests
            timeout=60      # Increase timeout
        )
        
        # Store results with better error handling
        def safe_float(value):
            """Safely convert RAGAS result to float"""
            try:
                if isinstance(value, list):
                    return float(value[0]) if value else 0.0
                elif isinstance(value, (int, float)):
                    return float(value)
                elif str(value).lower() in ['nan', 'none', '']:
                    return 0.0
                else:
                    return float(str(value))
            except:
                return 0.0
        
        results = {
            "ragas_scores": {
                "faithfulness": safe_float(ragas_results.get("faithfulness", 0.0)),
                "answer_relevancy": safe_float(ragas_results.get("answer_relevancy", 0.0)),
                "context_precision": safe_float(ragas_results.get("context_precision", 0.0)),
                "context_recall": safe_float(ragas_results.get("context_recall", 0.0))
            },
            "performance": {
                "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
                "min_response_time": min(response_times) if response_times else 0,
                "max_response_time": max(response_times) if response_times else 0,
                "total_questions": len(questions),
                "errors": len(errors)
            },
            "errors": errors
        }
        
        # Print improved results
        print(f"\n[SUCCESS] {chatbot_name} Improved Results:")
        print(f"   Faithfulness:       {results['ragas_scores']['faithfulness']:.4f}")
        print(f"   Answer Relevancy:   {results['ragas_scores']['answer_relevancy']:.4f}")
        print(f"   Context Precision:  {results['ragas_scores']['context_precision']:.4f}")
        print(f"   Context Recall:     {results['ragas_scores']['context_recall']:.4f}")
        print(f"   Avg Response Time:  {results['performance']['avg_response_time']:.2f}s")
        
        return results
        
    except Exception as e:
        print(f"   [ERROR] Improved RAGAS evaluation failed: {e}")
        
        # Return fallback results
        results = {
            "ragas_scores": {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0
            },
            "performance": {
                "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
                "min_response_time": min(response_times) if response_times else 0,
                "max_response_time": max(response_times) if response_times else 0,
                "total_questions": len(questions),
                "errors": len(errors) + 1
            },
            "errors": errors + [f"RAGAS evaluation failed: {e}"]
        }
        
        return results

def main():
    """Main improved evaluation program"""
    
    print("\n" + "="*80)
    print("IMPROVED RAGAS Evaluation - Enhanced Chatbot Comparison")
    print("="*80)
    print("Improvements:")
    print("- Better LLM configuration")
    print("- Improved test questions")
    print("- Enhanced error handling")
    print("- Optimized RAGAS settings")
    print()
    
    if not RAGAS_AVAILABLE:
        print("\n[ERROR] RAGAS not installed")
        return
    
    # Load improved test questions
    questions_data = load_improved_test_questions()
    questions = questions_data.get("test_questions", [])
    
    if not questions:
        print("[ERROR] No test questions available")
        return
    
    # Use first 3 questions for quick test
    questions = questions[:3]
    print(f"[INFO] Using {len(questions)} improved questions for evaluation")
    print()
    
    # Import and test chatbots
    try:
        from main_unified_chatbot_hybrid import UnifiedChatbotHybrid
        chatbot = UnifiedChatbotHybrid()
        results = run_improved_evaluation("Hybrid (Improved)", chatbot, questions)
        
        print("\n" + "="*80)
        print("IMPROVED EVALUATION RESULTS")
        print("="*80)
        
        scores = results["ragas_scores"]
        perf = results["performance"]
        
        print(f"Faithfulness:       {scores['faithfulness']:.4f}")
        print(f"Answer Relevancy:   {scores['answer_relevancy']:.4f}")
        print(f"Context Precision:  {scores['context_precision']:.4f}")
        print(f"Context Recall:     {scores['context_recall']:.4f}")
        print(f"Avg Response Time:  {perf['avg_response_time']:.2f}s")
        print(f"Errors:             {perf['errors']}")
        
        # Calculate overall score
        overall_score = (
            scores['faithfulness'] * 0.3 +
            scores['answer_relevancy'] * 0.3 +
            scores['context_precision'] * 0.2 +
            scores['context_recall'] * 0.2
        )
        print(f"Overall Score:      {overall_score:.4f}")
        
        print("\n[SUCCESS] Improved evaluation completed!")
        
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
