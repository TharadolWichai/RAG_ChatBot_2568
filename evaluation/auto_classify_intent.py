#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_classify_intent.py - จำแนก intent อัตโนมัติสำหรับ test questions

วิเคราะห์คำถามและกำหนด expected_intent, category, difficulty โดยอัตโนมัติ
"""

import sys
import json
import re
from pathlib import Path

# Fix Windows encoding issue
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


# Intent classification rules
INTENT_KEYWORDS = {
    "allpeople": [
        r'อาจารย์(?!.*รับ)', r'ผศ\.', r'ดร\.', r'รศ\.', r'ผู้ช่วยศาสตราจารย์', 
        r'รองศาสตราจารย์', r'ศาสตราจารย์', r'บุคลากร', r'คณะผู้บริหาร',
        r'ติดต่ออาจารย์', r'อีเมล.*อาจารย์', r'email.*อาจารย์',
        r'ความเชี่ยวชาญ', r'งานวิจัย.*อาจารย์', r'สาขา.*อาจารย์'
    ],
    "bsc_entrance": [
        r'รับเข้า', r'สมัคร(?!.*งาน|.*ทุน)', r'รอบ.*portfolio', r'โควตา',
        r'admission', r'tcas', r'คะแนน.*สอบ', r'tgat', r'a-level',
        r'เกณฑ์.*รับ', r'รับสมัคร', r'รอบที่.*1.*2.*3', r'netsat',
        r'ค่าธรรมเนียม.*ศึกษา', r'ค่าเล่าเรียน', r'หลักสูตร.*ปริญญาตรี',
        r'โครงการทุน.*ช้างเผือก', r'โครงการพิเศษ'
    ],
    "scholarship": [
        r'ทุน(?!.*ช้างเผือก)', r'scholarship', r'ทุน.*การศึกษา', r'ทุน.*วิจัย',
        r'ทุน.*asean', r'ทุน.*gms', r'ทุน.*นานาชาติ', r'ทุน.*บัณฑิต',
        r'เงื่อนไข.*ทุน', r'สมัคร.*ทุน', r'รับทุน', r'คุณสมบัติ.*ทุน'
    ],
    "research": [
        r'กลุ่มวิจัย', r'research.*group', r'aida', r'dsai', r'isne',
        r'งานวิจัย(?!.*อาจารย์)', r'ผลงาน.*วิจัย', r'โครงการ.*วิจัย',
        r'หัวหน้า.*กลุ่ม', r'สมาชิก.*กลุ่ม'
    ],
    "student_club": [
        r'สโมสร', r'ประธาน.*สโมสร', r'กิจกรรม.*นักศึกษา', r'ชุมนุม',
        r'องค์กร.*นักศึกษา', r'คณะกรรมการ.*นักศึกษา'
    ],
    "students": [
        r'ลิงก์.*นักศึกษา', r'ระบบ.*นักศึกษา', r'โครงงาน', r'project',
        r'ลงทะเบียน', r'ตารางเรียน', r'ตาราง.*สอบ', r'ระบบ.*คอมพิวเตอร์'
    ],
    "digital_services": [
        r'web.*hosting', r'virtual.*machine', r'vm', r'apple.*store',
        r'บริการ.*ดิจิตอล', r'บริการ.*ออนไลน์', r'cloud', r'server',
        r'เครื่องเสมือน', r'โฮสติ้ง'
    ],
    "links": [
        r'ลิงก์(?!.*นักศึกษา)', r'link', r'url', r'เว็บไซต์',
        r'จอง.*ห้องประชุม', r'จอง.*ห้อง', r'ระบบ.*จอง'
    ],
    "contact": [
        r'ติดต่อ.*วิทยาลัย', r'เบอร์.*วิทยาลัย', r'โทร.*วิทยาลัย',
        r'ที่อยู่.*วิทยาลัย', r'อีเมล.*วิทยาลัย', r'contact',
        r'fax', r'แฟกซ์', r'สำนักงาน'
    ],
    "news": [
        r'ข่าว', r'ประกาศ', r'กิจกรรม', r'event', r'news',
        r'แจ้ง', r'ข้อมูล.*ล่าสุด'
    ]
}

# Category classification
CATEGORY_PATTERNS = {
    "simple": [
        r'^.{0,50}$',  # คำถามสั้นๆ
        r'คือ.*อะไร$', r'อะไร$', r'เท่าไร$', r'ใช่.*ไหม$',
        r'มี.*ไหม$', r'มี.*อะไร.*บ้าง$', r'^ขอ.*ข้อมูล',
        r'^ลิงก์', r'^เบอร์', r'^อีเมล', r'^ติดต่อ'
    ],
    "complex": [
        r'อธิบาย', r'วิธี.*การ', r'ขั้นตอน', r'เงื่อนไข',
        r'คุณสมบัติ', r'เปรียบเทียบ', r'ความแตกต่าง',
        r'และ.*และ', r'.*พร้อม.*', r'.{80,}'  # คำถามยาว
    ],
    "ambiguous": [
        r'^มี.*บ้าง$', r'^บริการ.*อะไร', r'^ข้อมูล.*เกี่ยวกับ',
        r'^ขอ.*ข้อมูล(?!.*อาจารย์)'
    ]
}

# Difficulty classification
DIFFICULTY_PATTERNS = {
    "easy": [
        r'^ลิงก์', r'^เบอร์', r'^อีเมล', r'^ติดต่อ',
        r'คือ.*อะไร$', r'อะไร$', r'มี.*ไหม$'
    ],
    "hard": [
        r'อธิบาย.*และ.*วิธี', r'เปรียบเทียบ', r'ความแตกต่าง',
        r'วิเคราะห์', r'.{100,}'  # คำถามยาวมาก
    ]
}


def classify_intent(question: str) -> str:
    """
    จำแนก intent จากคำถาม
    
    Args:
        question: คำถาม
        
    Returns:
        intent string
    """
    question_lower = question.lower()
    
    # หาคะแนนสูงสุด
    max_score = 0
    best_intent = "unknown"
    
    for intent, keywords in INTENT_KEYWORDS.items():
        score = 0
        for pattern in keywords:
            if re.search(pattern, question, re.IGNORECASE):
                score += 1
        
        if score > max_score:
            max_score = score
            best_intent = intent
    
    return best_intent


def classify_category(question: str) -> str:
    """
    จำแนก category จากคำถาม
    
    Args:
        question: คำถาม
        
    Returns:
        category string
    """
    # ตรวจสอบแต่ละ category
    for category, patterns in CATEGORY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return category
    
    return "general"


def classify_difficulty(question: str) -> str:
    """
    จำแนก difficulty จากคำถาม
    
    Args:
        question: คำถาม
        
    Returns:
        difficulty string
    """
    # ตรวจสอบ easy
    for pattern in DIFFICULTY_PATTERNS.get("easy", []):
        if re.search(pattern, question, re.IGNORECASE):
            return "easy"
    
    # ตรวจสอบ hard
    for pattern in DIFFICULTY_PATTERNS.get("hard", []):
        if re.search(pattern, question, re.IGNORECASE):
            return "hard"
    
    return "medium"


def auto_classify_json(input_file: str, output_file: str = None):
    """
    จำแนก intent อัตโนมัติสำหรับ JSON file
    
    Args:
        input_file: path to input JSON file
        output_file: path to output JSON file (optional)
    """
    print(f"\n{'='*80}")
    print("🤖 Auto Intent Classification for RAGAS Test Questions")
    print(f"{'='*80}")
    
    # อ่าน JSON
    print(f"\n📖 Reading JSON file: {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Successfully loaded JSON")
        print(f"   📊 Questions: {len(data.get('test_questions', []))}")
    except FileNotFoundError:
        print(f"❌ Error: ไม่พบไฟล์ {input_file}")
        return None
    except Exception as e:
        print(f"❌ Error reading JSON: {e}")
        return None
    
    # จำแนก intent
    print(f"\n🔄 Classifying intents...")
    test_questions = data.get('test_questions', [])
    
    intent_stats = {}
    category_stats = {}
    difficulty_stats = {}
    
    for i, q in enumerate(test_questions, 1):
        question = q['question']
        
        # จำแนก
        intent = classify_intent(question)
        category = classify_category(question)
        difficulty = classify_difficulty(question)
        
        # อัพเดท
        q['expected_intent'] = intent
        q['category'] = category
        q['difficulty'] = difficulty
        
        # สถิติ
        intent_stats[intent] = intent_stats.get(intent, 0) + 1
        category_stats[category] = category_stats.get(category, 0) + 1
        difficulty_stats[difficulty] = difficulty_stats.get(difficulty, 0) + 1
        
        # แสดงผล
        if i <= 5:  # แสดงเฉพาะ 5 คำถามแรก
            print(f"\n   [{i}] {question[:60]}{'...' if len(question) > 60 else ''}")
            print(f"       → Intent: {intent} | Category: {category} | Difficulty: {difficulty}")
    
    if len(test_questions) > 5:
        print(f"\n   ... และอีก {len(test_questions) - 5} คำถาม")
    
    # อัพเดท metadata
    data['metadata']['categories'] = category_stats
    data['metadata']['difficulty'] = difficulty_stats
    data['metadata']['intents_coverage'] = intent_stats
    
    # กำหนด output file
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_classified.json"
    
    # บันทึกไฟล์
    print(f"\n💾 Saving to: {output_file}")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Successfully saved classified JSON")
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return None
    
    # แสดง statistics
    print(f"\n{'='*80}")
    print(f"📊 Classification Summary:")
    print(f"{'='*80}")
    
    print(f"\n🎯 Intent Distribution:")
    for intent, count in sorted(intent_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(test_questions)) * 100
        print(f"   {intent:<20} : {count:>3} ({percentage:>5.1f}%)")
    
    print(f"\n📂 Category Distribution:")
    for category, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(test_questions)) * 100
        print(f"   {category:<20} : {count:>3} ({percentage:>5.1f}%)")
    
    print(f"\n⚡ Difficulty Distribution:")
    for difficulty, count in sorted(difficulty_stats.items(), key=lambda x: ['easy', 'medium', 'hard'].index(x[0])):
        percentage = (count / len(test_questions)) * 100
        print(f"   {difficulty:<20} : {count:>3} ({percentage:>5.1f}%)")
    
    print(f"\n{'='*80}")
    print(f"✅ Classification completed!")
    print(f"   📄 Output file: {output_file}")
    print(f"   📊 Total questions: {len(test_questions)}")
    print(f"{'='*80}\n")
    
    return output_file


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='จำแนก intent อัตโนมัติสำหรับ test questions JSON'
    )
    parser.add_argument(
        'json_file',
        help='Path to JSON file'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output JSON file path (optional)',
        default=None
    )
    
    args = parser.parse_args()
    
    # Classify
    output_file = auto_classify_json(args.json_file, args.output)
    
    if output_file:
        print("\n💡 Next Steps:")
        print(f"\n1. ตรวจสอบและแก้ไข intent ถ้าจำเป็น:")
        print(f"   notepad {output_file}")
        print(f"\n2. ใช้กับ evaluation script:")
        print(f"   python evaluate_chatbots.py")
        print(f"\n3. หรือแทนที่ไฟล์เดิม:")
        print(f"   mv {output_file} test_questions.json")


if __name__ == "__main__":
    main()

