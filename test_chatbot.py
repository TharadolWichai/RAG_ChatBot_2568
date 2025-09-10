#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append('main_app')

# Import the chatbot function
from main_links import manual_qa_chain

def test_chatbot():
    print("🔗 ทดสอบระบบแชทบอทลิงก์")
    print("="*50)
    
    # Test questions
    test_questions = [
        "ขอลิงก์จองห้องประชุม",
        "ลิงก์อัปโหลดไฟล์", 
        "ระบบโครงงาน",
        "ห้องสมุด"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🔍 ทดสอบที่ {i}: {question}")
        print("-" * 40)
        
        try:
            result = manual_qa_chain(question)
            print(f"🤖 ผลลัพธ์: {result[:200]}...")
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาด: {e}")
        
        print("-" * 40)

if __name__ == "__main__":
    test_chatbot()
