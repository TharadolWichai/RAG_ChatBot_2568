#!/usr/bin/env python3
"""
Test script to verify GPT-4o API access via OpenRouter
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_gpt4o_access():
    """Test GPT-4o API access"""
    print("\n" + "="*60)
    print("Testing GPT-4o API Access via OpenRouter")
    print("="*60)
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("\n[ERROR] No API key found!")
        print("Please set OPENAI_API_KEY or OPENROUTER_API_KEY in .env")
        return False
    
    print(f"\n[INFO] API Key: {api_key[:15]}...{api_key[-10:]}")
    print(f"[INFO] Key format: {'OpenRouter' if api_key.startswith('sk-or-') else 'OpenAI'}")
    
    # Test with LangChain
    try:
        from langchain_openai import ChatOpenAI
        
        print("\n[INFO] Creating ChatOpenAI instance...")
        llm = ChatOpenAI(
            model="openai/gpt-4o-2024-11-20",
            temperature=0.1,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/ChatBot_RAG_CS_KKU",
                "X-Title": "CS_KKU_Test"
            }
        )
        
        print("[INFO] Testing simple query...")
        response = llm.invoke("Say 'API test successful' in Thai")
        
        print("\n[SUCCESS] API Test Passed!")
        print(f"[RESPONSE] {response.content}")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] API Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gpt4o_access()
    if success:
        print("\n" + "="*60)
        print("[INFO] GPT-4o API is working correctly!")
        print("[INFO] You can now run evaluate_chatbots.py")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("[ERROR] GPT-4o API test failed")
        print("[INFO] Please check your API key and configuration")
        print("="*60)

