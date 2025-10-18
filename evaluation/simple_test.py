# simple_test.py - Simple Manual Testing Script
# ทดสอบ chatbot แบบง่ายๆ โดยไม่ใช้ RAGAS

import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'main_app'))

from dotenv import load_dotenv
load_dotenv()

print("📦 กำลังโหลด Chatbot ทั้ง 3 versions...")

# Import chatbots
try:
    from main_unified_chatbot import UnifiedChatbot
    RULE_BASED_AVAILABLE = True
    print("✅ Rule-Based loaded")
except Exception as e:
    RULE_BASED_AVAILABLE = False
    print(f"⚠️ Rule-Based not available: {e}")

try:
    from main_unified_chatbot_llm import UnifiedChatbotLLM
    LLM_BASED_AVAILABLE = True
    print("✅ LLM-Based loaded")
except Exception as e:
    LLM_BASED_AVAILABLE = False
    print(f"⚠️ LLM-Based not available: {e}")

try:
    from main_unified_chatbot_hybrid import UnifiedChatbotHybrid
    HYBRID_AVAILABLE = True
    print("✅ Hybrid loaded")
except Exception as e:
    HYBRID_AVAILABLE = False
    print(f"⚠️ Hybrid not available: {e}")

print()

# Simple test questions
SIMPLE_TESTS = [
    "อาจารย์สมชาย",
    "ติดต่อวิทยาลัย",
    "ทุนการศึกษา",
    "Web Hosting",
    "กลุ่มวิจัย AIDA"
]

def test_chatbot(chatbot, chatbot_name: str, questions: list):
    """ทดสอบ chatbot ด้วยคำถามชุดหนึ่ง"""
    
    print("\n" + "="*80)
    print(f"🧪 Testing: {chatbot_name}")
    print("="*80)
    
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] Question: {question}")
        print("-" * 80)
        
        start_time = time.time()
        
        try:
            answer = chatbot.answer(question)
            elapsed = time.time() - start_time
            
            # Show first 200 chars of answer
            answer_preview = answer[:200] + "..." if len(answer) > 200 else answer
            
            print(f"✅ Answer ({elapsed:.2f}s):")
            print(answer_preview)
            
            results.append({
                "question": question,
                "answer": answer,
                "time": elapsed,
                "success": True
            })
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Error ({elapsed:.2f}s): {e}")
            
            results.append({
                "question": question,
                "answer": None,
                "time": elapsed,
                "success": False,
                "error": str(e)
            })
    
    # Summary
    print("\n" + "="*80)
    print(f"📊 Summary for {chatbot_name}:")
    print("="*80)
    
    total_time = sum(r["time"] for r in results)
    success_count = sum(1 for r in results if r["success"])
    avg_time = total_time / len(results) if results else 0
    
    print(f"✅ Successful: {success_count}/{len(results)}")
    print(f"⏱️  Total Time: {total_time:.2f}s")
    print(f"⏱️  Avg Time: {avg_time:.2f}s")
    
    return results

def compare_versions(questions: list = None):
    """เปรียบเทียบ chatbot ทั้ง 3 versions"""
    
    if questions is None:
        questions = SIMPLE_TESTS
    
    print("\n" + "="*80)
    print("🔬 SIMPLE COMPARISON TEST")
    print("="*80)
    print(f"📝 Testing with {len(questions)} questions")
    print()
    
    all_results = {}
    
    # Test Rule-Based
    if RULE_BASED_AVAILABLE:
        print("\n🔧 Initializing Rule-Based...")
        rule_chatbot = UnifiedChatbot()
        all_results["Rule-Based"] = test_chatbot(rule_chatbot, "Rule-Based", questions)
    
    # Test LLM-Based
    if LLM_BASED_AVAILABLE:
        print("\n🤖 Initializing LLM-Based...")
        llm_chatbot = UnifiedChatbotLLM()
        all_results["LLM-Based"] = test_chatbot(llm_chatbot, "LLM-Based", questions)
    
    # Test Hybrid
    if HYBRID_AVAILABLE:
        print("\n🔀 Initializing Hybrid...")
        hybrid_chatbot = UnifiedChatbotHybrid()
        all_results["Hybrid"] = test_chatbot(hybrid_chatbot, "Hybrid", questions)
    
    # Final comparison
    print("\n" + "="*80)
    print("📊 FINAL COMPARISON")
    print("="*80)
    
    print(f"\n{'Version':<20} | {'Success Rate':<15} | {'Avg Time':<12} | {'Total Time':<12}")
    print("-" * 80)
    
    for version, results in all_results.items():
        success_count = sum(1 for r in results if r["success"])
        success_rate = f"{success_count}/{len(results)}"
        avg_time = sum(r["time"] for r in results) / len(results)
        total_time = sum(r["time"] for r in results)
        
        print(f"{version:<20} | {success_rate:<15} | {avg_time:>10.2f}s | {total_time:>10.2f}s")
    
    print("="*80)

def main():
    """Main program"""
    
    print("\n" + "="*80)
    print("🧪 Simple Chatbot Testing (No RAGAS)")
    print("="*80)
    print("📝 Quick manual testing of 3 chatbot versions")
    print()
    
    # Check available chatbots
    available = sum([RULE_BASED_AVAILABLE, LLM_BASED_AVAILABLE, HYBRID_AVAILABLE])
    
    if available == 0:
        print("❌ No chatbots available!")
        return
    
    print(f"✅ {available} chatbot(s) available\n")
    
    # Ask for test mode
    print("📊 Test Options:")
    print("   1. Quick test (5 questions)")
    print("   2. Custom questions (enter your own)")
    print()
    
    try:
        choice = input("Select option (1-2) [default: 1]: ").strip()
        
        if choice == "2":
            # Custom questions
            print("\nEnter questions (one per line, empty line to finish):")
            questions = []
            while True:
                q = input(f"Question {len(questions)+1}: ").strip()
                if not q:
                    break
                questions.append(q)
            
            if not questions:
                print("⚠️ No questions entered, using default")
                questions = SIMPLE_TESTS
        else:
            questions = SIMPLE_TESTS
    except:
        questions = SIMPLE_TESTS
    
    # Run comparison
    compare_versions(questions)
    
    print("\n✅ Testing completed!")

if __name__ == "__main__":
    main()

