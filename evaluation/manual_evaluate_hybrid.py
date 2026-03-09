"""
manual_evaluate_hybrid.py
สคริปต์สำหรับถามคำถามและบันทึกผลลัพธ์แบบ Manual
ใช้กับ UnifiedChatbotAutomated (Hybrid Classification)

Usage:
    python manual_evaluate_hybrid.py
    
หรือระบุ dataset:
    python manual_evaluate_hybrid.py --dataset my_dataset.json --output results_hybrid.json
"""

import sys
import os
import json
import time
import argparse
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'main_app'))

from dotenv import load_dotenv
load_dotenv()

# Import chatbot และ logger
try:
    from main_unified_chatbot_automated import UnifiedChatbotAutomated
    CHATBOT_AVAILABLE = True
except Exception as e:
    print(f"❌ Error loading chatbot: {e}")
    CHATBOT_AVAILABLE = False

from json_response_logger import save_response_to_json, get_results_summary


def load_dataset(filepath: str) -> List[Dict[str, Any]]:
    """
    โหลด dataset จากไฟล์ JSON
    
    Args:
        filepath: path ไปยังไฟล์ dataset
        
    Returns:
        List ของคำถาม
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # รองรับหลาย format
        if "dataset" in data:
            dataset = data["dataset"]
        elif "test_questions" in data:
            dataset = data["test_questions"]
        else:
            dataset = data if isinstance(data, list) else []
        
        print(f"✅ โหลด dataset สำเร็จ: {len(dataset)} คำถาม")
        return dataset
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return []


def manual_evaluate(
    chatbot: UnifiedChatbotAutomated,
    dataset: List[Dict[str, Any]],
    output_file: str = "results_hybrid.json",
    start_from: int = 1
) -> None:
    """
    ประเมินผลแบบ manual (ถามทีละคำถาม)
    
    Args:
        chatbot: UnifiedChatbotAutomated instance
        dataset: รายการคำถาม
        output_file: ไฟล์ output
        start_from: เริ่มจาก ID ไหน (สำหรับ resume)
    """
    print(f"\n{'='*60}")
    print(f"🎯 เริ่มการประเมินผลแบบ Manual")
    print(f"{'='*60}")
    print(f"   จำนวนคำถาม: {len(dataset)}")
    print(f"   ไฟล์ output: {output_file}")
    print(f"   เริ่มจาก ID: {start_from}")
    print(f"{'='*60}\n")
    
    total_time = 0
    errors = []
    
    for i, item in enumerate(dataset, 1):
        question_id = item.get("id", i)
        question = item.get("question", "")
        
        # Skip if before start_from
        if question_id < start_from:
            print(f"⏭️  ข้าม ID {question_id} (เริ่มจาก {start_from})")
            continue
        
        print(f"\n{'#'*60}")
        print(f"📝 คำถามที่ {i}/{len(dataset)} (ID: {question_id})")
        print(f"{'#'*60}")
        print(f"❓ {question}")
        print()
        
        try:
            # วัดเวลา
            start_time = time.time()
            
            # ถามคำถาม (ได้ answer, contexts, classification)
            answer, contexts, classification = chatbot.answer_with_contexts(question)
            
            # คำนวณเวลา
            response_time = time.time() - start_time
            total_time += response_time
            
            # บันทึกลง JSON
            save_response_to_json(
                question_id=question_id,
                question=question,
                answer=answer,
                contexts=contexts,
                classification_info=classification,
                response_time=response_time,
                output_file=output_file,
                chatbot_type="UnifiedChatbotAutomated"
            )
            
            print(f"\n{'='*60}")
            print(f"✅ บันทึก ID {question_id} สำเร็จ!")
            print(f"{'='*60}\n")
            
        except KeyboardInterrupt:
            print(f"\n\n⚠️ ผู้ใช้หยุดการทำงาน (Ctrl+C)")
            print(f"   บันทึกไปแล้ว: {i-1} คำถาม")
            print(f"   สามารถ resume จาก ID {question_id} ได้")
            break
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            
            errors.append({
                "id": question_id,
                "question": question,
                "error": str(e)
            })
            
            # ถามว่าจะ continue หรือไม่
            print(f"\n⚠️ เกิดข้อผิดพลาดกับ ID {question_id}")
            response = input("ต้องการข้ามไปคำถามถัดไปไหม? (y/n): ").strip().lower()
            if response != 'y':
                print("❌ หยุดการทำงาน")
                break
    
    # สรุปผลลัพธ์
    print(f"\n{'='*60}")
    print(f"🎉 การประเมินผลเสร็จสิ้น!")
    print(f"{'='*60}")
    print(f"   เวลารวม: {total_time:.2f}s")
    print(f"   เวลาเฉลี่ย: {total_time/len(dataset):.2f}s per question")
    
    if errors:
        print(f"\n⚠️ เกิดข้อผิดพลาด {len(errors)} คำถาม:")
        for error in errors:
            print(f"   - ID {error['id']}: {error['error']}")
    
    print(f"\n{'='*60}\n")
    
    # แสดงสรุปข้อมูลในไฟล์
    get_results_summary(output_file)


def interactive_mode(chatbot: UnifiedChatbotAutomated, output_file: str = "results_interactive.json"):
    """
    โหมดถาม-ตอบแบบ interactive
    ถามทีละคำถามและบันทึกทันที
    
    Args:
        chatbot: UnifiedChatbotAutomated instance
        output_file: ไฟล์ output
    """
    print(f"\n{'='*60}")
    print(f"💬 โหมด Interactive")
    print(f"{'='*60}")
    print(f"   พิมพ์คำถามเพื่อทดสอบ")
    print(f"   พิมพ์ 'exit' เพื่อออก")
    print(f"   พิมพ์ 'summary' เพื่อดูสรุป")
    print(f"{'='*60}\n")
    
    question_counter = 1
    
    while True:
        try:
            question = input(f"\n❓ คำถามที่ {question_counter} (หรือ 'exit'): ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['exit', 'quit', 'ออก']:
                print("👋 ออกจากโหมด interactive")
                break
            
            if question.lower() in ['summary', 'สรุป']:
                get_results_summary(output_file)
                continue
            
            print()
            
            # วัดเวลา
            start_time = time.time()
            
            # ถามคำถาม
            answer, contexts, classification = chatbot.answer_with_contexts(question)
            
            # คำนวณเวลา
            response_time = time.time() - start_time
            
            # แสดงผลลัพธ์
            print(f"\n{'='*60}")
            print(f"🤖 คำตอบ:")
            print(f"{'='*60}")
            print(answer)
            print(f"\n{'='*60}")
            print(f"📊 สถิติ:")
            print(f"   - Contexts: {len(contexts)}")
            print(f"   - Intent: {classification['intent']}")
            print(f"   - Confidence: {classification['confidence']:.2f}")
            print(f"   - Method: {classification['method']}")
            print(f"   - Time: {response_time:.2f}s")
            print(f"{'='*60}\n")
            
            # ถามว่าจะบันทึกหรือไม่
            save = input("💾 บันทึกลง JSON ไหม? (y/n): ").strip().lower()
            
            if save == 'y':
                save_response_to_json(
                    question_id=question_counter,
                    question=question,
                    answer=answer,
                    contexts=contexts,
                    classification_info=classification,
                    response_time=response_time,
                    output_file=output_file
                )
                question_counter += 1
            else:
                print("⏭️  ข้ามการบันทึก")
                
        except KeyboardInterrupt:
            print("\n\n👋 ออกจากโหมด interactive (Ctrl+C)")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Manual Evaluation Tool for UnifiedChatbotAutomated"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to dataset JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results_hybrid.json",
        help="Output file path (default: results_hybrid.json)"
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        help="Start from question ID (for resuming)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode (ask questions manually)"
    )
    parser.add_argument(
        "--summary",
        type=str,
        help="Show summary of existing results file"
    )
    
    args = parser.parse_args()
    
    # Show summary only
    if args.summary:
        get_results_summary(args.summary)
        return
    
    # Check chatbot availability
    if not CHATBOT_AVAILABLE:
        print("❌ Chatbot not available!")
        return
    
    # Initialize chatbot
    print("\n🤖 กำลังเริ่มต้น UnifiedChatbotAutomated...")
    try:
        chatbot = UnifiedChatbotAutomated()
    except Exception as e:
        print(f"❌ Error initializing chatbot: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Interactive mode
    if args.interactive:
        interactive_mode(chatbot, args.output)
        return
    
    # Dataset mode
    if not args.dataset:
        print("❌ Error: ต้องระบุ --dataset หรือใช้ --interactive")
        return
    
    # Load dataset
    dataset = load_dataset(args.dataset)
    
    if not dataset:
        print("❌ Dataset is empty or invalid!")
        return
    
    # Run manual evaluation
    manual_evaluate(chatbot, dataset, args.output, args.start_from)


if __name__ == "__main__":
    main()

