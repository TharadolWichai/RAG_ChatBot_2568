#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_context_retrieval.py - ทดสอบการดึง contexts จาก chatbot

"""
สคริปต์ทดสอบ method answer_with_contexts() ของ chatbot
เพื่อให้แน่ใจว่า Context Precision สามารถวัดได้อย่างถูกต้อง
"""

import sys
import os

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'main_app'))

print("="*80)
print("🧪 Context Retrieval Test")
print("="*80)
print("Testing if chatbots can return contexts for RAGAS evaluation\n")

# Import chatbot classes
try:
    from main_unified_chatbot import UnifiedChatbot
    print("✅ Rule-Based Chatbot imported")
    RULE_BASED_OK = True
except Exception as e:
    print(f"❌ Rule-Based Chatbot import failed: {e}")
    RULE_BASED_OK = False

try:
    from main_unified_chatbot_llm import UnifiedChatbotLLM
    print("✅ LLM-Based Chatbot imported")
    LLM_BASED_OK = True
except Exception as e:
    print(f"❌ LLM-Based Chatbot import failed: {e}")
    LLM_BASED_OK = False

try:
    from main_unified_chatbot_hybrid import UnifiedChatbotHybrid
    print("✅ Hybrid Chatbot imported")
    HYBRID_OK = True
except Exception as e:
    print(f"❌ Hybrid Chatbot import failed: {e}")
    HYBRID_OK = False

print()

# Test questions
TEST_QUESTIONS = [
    "จองห้องประชุม",
    "ติดต่อวิทยาลัย",
    "อาจารย์พุธษดี"
]

def test_chatbot(name: str, chatbot: any, test_question: str):
    """ทดสอบการดึง contexts จาก chatbot"""
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"Question: {test_question}")
    print(f"{'='*80}\n")
    
    # Check if method exists
    if not hasattr(chatbot, 'answer_with_contexts'):
        print(f"❌ Method 'answer_with_contexts' not found in {name}")
        return False
    
    try:
        # Get answer and contexts
        answer, contexts = chatbot.answer_with_contexts(test_question)
        
        print(f"\n✅ Successfully retrieved answer and contexts!")
        print(f"\n📝 Answer Preview (first 200 chars):")
        print(f"   {answer[:200]}...")
        
        print(f"\n📚 Contexts Retrieved: {len(contexts)}")
        print(f"\n🔍 Context Previews:")
        for i, ctx in enumerate(contexts[:3], 1):
            preview = ctx[:150].replace('\n', ' ')
            print(f"   [{i}] {preview}...")
        
        # Check if contexts are real (not placeholders)
        if contexts and not any("No specific contexts" in c or "Error" in c for c in contexts):
            print(f"\n✨ Contexts look valid (not placeholders)")
            return True
        else:
            print(f"\n⚠️  Contexts might be placeholders or empty")
            return False
            
    except Exception as e:
        print(f"\n❌ Error testing {name}: {e}")
        import traceback
        traceback.print_exc()
        return False

# Run tests
print("\n" + "="*80)
print("🚀 Starting Context Retrieval Tests")
print("="*80)

results = {}
test_question = TEST_QUESTIONS[0]  # Use first question for testing

if RULE_BASED_OK:
    print("\n📋 Initializing Rule-Based Chatbot...")
    try:
        chatbot = UnifiedChatbot()
        results["Rule-Based"] = test_chatbot("Rule-Based Chatbot", chatbot, test_question)
    except Exception as e:
        print(f"❌ Failed to initialize Rule-Based Chatbot: {e}")
        results["Rule-Based"] = False

if LLM_BASED_OK:
    print("\n📋 Initializing LLM-Based Chatbot...")
    try:
        chatbot = UnifiedChatbotLLM()
        results["LLM-Based"] = test_chatbot("LLM-Based Chatbot", chatbot, test_question)
    except Exception as e:
        print(f"❌ Failed to initialize LLM-Based Chatbot: {e}")
        results["LLM-Based"] = False

if HYBRID_OK:
    print("\n📋 Initializing Hybrid Chatbot...")
    try:
        chatbot = UnifiedChatbotHybrid()
        results["Hybrid"] = test_chatbot("Hybrid Chatbot", chatbot, test_question)
    except Exception as e:
        print(f"❌ Failed to initialize Hybrid Chatbot: {e}")
        results["Hybrid"] = False

# Summary
print("\n" + "="*80)
print("📊 TEST SUMMARY")
print("="*80)

if results:
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {name:<20}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nContext retrieval is working correctly.")
        print("Context Precision should now be calculated properly in RAGAS evaluation.")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("\nPlease check the errors above.")
else:
    print("❌ No chatbots could be tested")

print("="*80)

