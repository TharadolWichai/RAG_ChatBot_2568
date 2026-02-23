"""
json_response_logger.py
Helper functions สำหรับบันทึก question, answer, contexts ลง JSON
สำหรับการประเมินผลด้วย RAGAS

Usage:
    from json_response_logger import save_response_to_json
    
    save_response_to_json(
        question_id=1,
        question="...",
        answer="...",
        contexts=["...", "..."],
        classification_info={...},
        response_time=1.23,
        output_file="results_hybrid.json"
    )
"""

import json
import os
from typing import List, Dict, Any
from datetime import datetime


def load_existing_results(filepath: str) -> Dict[str, Any]:
    """
    โหลดไฟล์ JSON ที่มีอยู่
    
    Args:
        filepath: path ไปยังไฟล์ JSON
        
    Returns:
        Dict ที่มี metadata และ results
    """
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"⚠️ Warning: Failed to load existing file: {e}")
        return None


def create_new_results_file(filepath: str, chatbot_type: str = "UnifiedChatbotAutomated") -> Dict[str, Any]:
    """
    สร้างโครงสร้างไฟล์ JSON ใหม่
    
    Args:
        filepath: path ไปยังไฟล์ JSON
        chatbot_type: ชื่อประเภท chatbot
        
    Returns:
        Dict โครงสร้างใหม่
    """
    now = datetime.now().isoformat()
    
    return {
        "metadata": {
            "chatbot_type": chatbot_type,
            "classification_method": "hybrid",
            "created_at": now,
            "last_updated": now,
            "total_questions": 0
        },
        "results": []
    }


def update_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    อัพเดท metadata (total_questions, last_updated)
    
    Args:
        data: Dict ที่มี metadata และ results
        
    Returns:
        Dict ที่อัพเดทแล้ว
    """
    data["metadata"]["last_updated"] = datetime.now().isoformat()
    data["metadata"]["total_questions"] = len(data["results"])
    return data


def save_response_to_json(
    question_id: int,
    question: str,
    answer: str,
    contexts: List[str],
    classification_info: Dict[str, Any],
    response_time: float,
    output_file: str = "results_hybrid.json",
    chatbot_type: str = "UnifiedChatbotAutomated"
) -> None:
    """
    บันทึก response ลง JSON file (แบบ manual - เรียกทีละครั้ง)
    จะ append ข้อมูลเข้าไปในไฟล์ที่มีอยู่ หรือสร้างไฟล์ใหม่ถ้ายังไม่มี
    
    Args:
        question_id: ID ของคำถาม (ควรเป็น unique)
        question: คำถาม
        answer: คำตอบจากโมเดล
        contexts: รายการ contexts ที่ retrieve มา
        classification_info: ข้อมูล classification (intent, confidence, method, reason)
        response_time: เวลาในการตอบ (วินาที)
        output_file: path ไปยังไฟล์ output (default: results_hybrid.json)
        chatbot_type: ชื่อประเภท chatbot (default: UnifiedChatbotAutomated)
    """
    # Load existing data or create new
    data = load_existing_results(output_file)
    
    if data is None:
        print(f"📄 สร้างไฟล์ใหม่: {output_file}")
        data = create_new_results_file(output_file, chatbot_type)
    else:
        print(f"📂 เพิ่มข้อมูลลงไฟล์: {output_file}")
    
    # Check if ID already exists
    existing_ids = [result["id"] for result in data["results"]]
    if question_id in existing_ids:
        print(f"⚠️ Warning: ID {question_id} already exists. Overwriting...")
        # Remove existing entry
        data["results"] = [r for r in data["results"] if r["id"] != question_id]
    
    # Create new result entry
    new_result = {
        "id": question_id,
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "num_contexts": len(contexts),
        "classification": classification_info,
        "response_time": round(response_time, 2),
        "timestamp": datetime.now().isoformat()
    }
    
    # Append to results
    data["results"].append(new_result)
    
    # Sort by ID
    data["results"].sort(key=lambda x: x["id"])
    
    # Update metadata
    data = update_metadata(data)
    
    # Save to file
    try:
        # Create directory if not exists
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ บันทึกสำเร็จ: ID {question_id}")
        print(f"   - Question: {question[:50]}{'...' if len(question) > 50 else ''}")
        print(f"   - Contexts: {len(contexts)}")
        print(f"   - Classification: {classification_info['intent']} ({classification_info['confidence']:.2f})")
        print(f"   - Response time: {response_time:.2f}s")
        print(f"   - Total in file: {data['metadata']['total_questions']} questions")
        
    except Exception as e:
        print(f"❌ Error saving to file: {e}")
        raise


def get_results_summary(filepath: str) -> None:
    """
    แสดงสรุปข้อมูลในไฟล์ JSON
    
    Args:
        filepath: path ไปยังไฟล์ JSON
    """
    data = load_existing_results(filepath)
    
    if data is None:
        print(f"❌ ไม่พบไฟล์: {filepath}")
        return
    
    print(f"\n{'='*60}")
    print(f"📊 สรุปข้อมูลในไฟล์: {filepath}")
    print(f"{'='*60}")
    
    metadata = data.get("metadata", {})
    results = data.get("results", [])
    
    print(f"\n📋 Metadata:")
    print(f"   Chatbot Type: {metadata.get('chatbot_type', 'N/A')}")
    print(f"   Classification: {metadata.get('classification_method', 'N/A')}")
    print(f"   Created: {metadata.get('created_at', 'N/A')}")
    print(f"   Last Updated: {metadata.get('last_updated', 'N/A')}")
    print(f"   Total Questions: {metadata.get('total_questions', 0)}")
    
    if results:
        print(f"\n📝 Results:")
        
        # Classification breakdown
        intents = {}
        methods = {}
        total_contexts = 0
        total_time = 0
        
        for result in results:
            intent = result.get("classification", {}).get("intent", "unknown")
            method = result.get("classification", {}).get("method", "unknown")
            
            intents[intent] = intents.get(intent, 0) + 1
            methods[method] = methods.get(method, 0) + 1
            total_contexts += result.get("num_contexts", 0)
            total_time += result.get("response_time", 0)
        
        print(f"\n   🎯 Intent Distribution:")
        for intent, count in sorted(intents.items(), key=lambda x: x[1], reverse=True):
            print(f"      {intent}: {count}")
        
        print(f"\n   🔧 Classification Method:")
        for method, count in sorted(methods.items(), key=lambda x: x[1], reverse=True):
            print(f"      {method}: {count}")
        
        print(f"\n   📚 Contexts:")
        print(f"      Total: {total_contexts}")
        print(f"      Average: {total_contexts/len(results):.1f} per question")
        
        print(f"\n   ⏱️  Response Time:")
        print(f"      Total: {total_time:.2f}s")
        print(f"      Average: {total_time/len(results):.2f}s per question")
        
        print(f"\n   📊 Questions (ID: Question):")
        for result in results:
            q_id = result.get("id")
            question = result.get("question", "")
            intent = result.get("classification", {}).get("intent", "unknown")
            print(f"      {q_id}: {question[:50]}{'...' if len(question) > 50 else ''} [{intent}]")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # Test/Example usage
    print("📝 Testing JSON Response Logger...")
    
    # Example 1: Save a response
    save_response_to_json(
        question_id=1,
        question="อาจารย์สมชาย",
        answer="พบข้อมูลอาจารย์สมชาย...",
        contexts=[
            "อาจารย์ สมชาย ใจดี\nตำแหน่ง: อาจารย์ประจำ...",
            "ผศ.ดร.สมชาย ศรีสุข\nความเชี่ยวชาญ: Machine Learning..."
        ],
        classification_info={
            "intent": "allpeople",
            "confidence": 9.0,
            "method": "rule_based",
            "reason": "High confidence from keyword matching"
        },
        response_time=2.35,
        output_file="test_results.json"
    )
    
    # Example 2: Save another response
    save_response_to_json(
        question_id=2,
        question="ติดต่อวิทยาลัย",
        answer="ข้อมูลติดต่อวิทยาลัยการคอมพิวเตอร์...",
        contexts=[
            "เบอร์โทร: 043-123456\nอีเมล: admin@cskku.com...",
        ],
        classification_info={
            "intent": "contact",
            "confidence": 12.0,
            "method": "rule_based",
            "reason": "High confidence from keyword matching"
        },
        response_time=1.45,
        output_file="test_results.json"
    )
    
    # Show summary
    get_results_summary("test_results.json")
    
    print("✅ Test completed!")

