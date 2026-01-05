#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
excel_to_test_questions_complete.py - Complete solution for Excel to Test Questions

แปลงไฟล์ Excel เป็น test_questions.json สำหรับ RAGAS evaluation
(รวมทุกขั้นตอนในไฟล์เดียว)
"""

import sys
import os
import json
import re
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

try:
    import pandas as pd
except ImportError:
    print("❌ Error: pandas ไม่ได้ติดตั้ง")
    print("กรุณาติดตั้งด้วยคำสั่ง: pip install pandas openpyxl")
    sys.exit(1)


# ==========================================
# Intent Classification Rules
# ==========================================

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

CATEGORY_PATTERNS = {
    "simple": [
        r'^.{0,50}$',
        r'คือ.*อะไร$', r'อะไร$', r'เท่าไร$', r'ใช่.*ไหม$',
        r'มี.*ไหม$', r'มี.*อะไร.*บ้าง$', r'^ขอ.*ข้อมูล',
        r'^ลิงก์', r'^เบอร์', r'^อีเมล', r'^ติดต่อ'
    ],
    "complex": [
        r'อธิบาย', r'วิธี.*การ', r'ขั้นตอน', r'เงื่อนไข',
        r'คุณสมบัติ', r'เปรียบเทียบ', r'ความแตกต่าง',
        r'และ.*และ', r'.*พร้อม.*', r'.{80,}'
    ],
    "ambiguous": [
        r'^มี.*บ้าง$', r'^บริการ.*อะไร', r'^ข้อมูล.*เกี่ยวกับ',
        r'^ขอ.*ข้อมูล(?!.*อาจารย์)'
    ]
}

DIFFICULTY_PATTERNS = {
    "easy": [
        r'^ลิงก์', r'^เบอร์', r'^อีเมล', r'^ติดต่อ',
        r'คือ.*อะไร$', r'อะไร$', r'มี.*ไหม$'
    ],
    "hard": [
        r'อธิบาย.*และ.*วิธี', r'เปรียบเทียบ', r'ความแตกต่าง',
        r'วิเคราะห์', r'.{100,}'
    ]
}


def classify_intent(question: str) -> str:
    """จำแนก intent จากคำถาม"""
    max_score = 0
    best_intent = "unknown"
    
    for intent, keywords in INTENT_KEYWORDS.items():
        score = sum(1 for pattern in keywords if re.search(pattern, question, re.IGNORECASE))
        if score > max_score:
            max_score = score
            best_intent = intent
    
    return best_intent


def classify_category(question: str) -> str:
    """จำแนก category จากคำถาม"""
    for category, patterns in CATEGORY_PATTERNS.items():
        if any(re.search(pattern, question, re.IGNORECASE) for pattern in patterns):
            return category
    return "general"


def classify_difficulty(question: str) -> str:
    """จำแนก difficulty จากคำถาม"""
    for pattern in DIFFICULTY_PATTERNS.get("easy", []):
        if re.search(pattern, question, re.IGNORECASE):
            return "easy"
    
    for pattern in DIFFICULTY_PATTERNS.get("hard", []):
        if re.search(pattern, question, re.IGNORECASE):
            return "hard"
    
    return "medium"


def convert_excel_to_test_questions(excel_file: str, output_file: str = "test_questions_new.json"):
    """
    แปลง Excel เป็น test_questions.json แบบครบวงจร
    
    Args:
        excel_file: path to Excel file
        output_file: path to output JSON file
    
    Returns:
        path to output file or None if failed
    """
    
    print("\n" + "="*80)
    print("🚀 Excel to Test Questions - Complete Conversion")
    print("="*80)
    
    # Step 1: Read Excel
    print(f"\n📖 Step 1: Reading Excel file...")
    print(f"   File: {excel_file}")
    
    try:
        df = pd.read_excel(excel_file)
        print(f"   ✅ Success! Rows: {len(df)}, Columns: {len(df.columns)}")
    except FileNotFoundError:
        print(f"   ❌ Error: ไม่พบไฟล์ {excel_file}")
        return None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None
    
    # Find columns
    print(f"\n🔍 Step 2: Detecting columns...")
    column_mapping = {}
    
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if any(kw in col_lower for kw in ['question', 'คำถาม']):
            column_mapping['question'] = col
        elif any(kw in col_lower for kw in ['ground', 'truth', 'answer', 'คำตอบ', 'เฉลย']):
            if 'question' not in column_mapping or col != column_mapping['question']:
                column_mapping['ground_truth'] = col
    
    if 'question' not in column_mapping:
        print(f"   ❌ Error: ไม่พบ column สำหรับคำถาม")
        return None
    
    print(f"   ✅ Found columns:")
    for key, value in column_mapping.items():
        print(f"      {key}: {value}")
    
    # Convert and classify
    print(f"\n🔄 Step 3: Converting and classifying...")
    test_questions = []
    skipped = 0
    
    intent_stats = {}
    category_stats = {}
    difficulty_stats = {}
    
    for idx, row in df.iterrows():
        question = row[column_mapping['question']]
        
        if pd.isna(question) or str(question).strip() == '':
            skipped += 1
            continue
        
        question = str(question).strip()
        
        # Classify
        intent = classify_intent(question)
        category = classify_category(question)
        difficulty = classify_difficulty(question)
        
        # Ground truth
        ground_truth = ""
        if 'ground_truth' in column_mapping:
            gt = row[column_mapping['ground_truth']]
            if not pd.isna(gt):
                ground_truth = str(gt).strip()
        
        if not ground_truth:
            ground_truth = f"ควรตอบคำถาม: {question}"
        
        # Create test case
        test_case = {
            "id": len(test_questions) + 1,
            "question": question,
            "expected_intent": intent,
            "ground_truth": ground_truth,
            "category": category,
            "difficulty": difficulty
        }
        
        test_questions.append(test_case)
        
        # Stats
        intent_stats[intent] = intent_stats.get(intent, 0) + 1
        category_stats[category] = category_stats.get(category, 0) + 1
        difficulty_stats[difficulty] = difficulty_stats.get(difficulty, 0) + 1
        
        # Preview first 3
        if len(test_questions) <= 3:
            print(f"\n   [{len(test_questions)}] {question[:60]}{'...' if len(question) > 60 else ''}")
            print(f"       → {intent} | {category} | {difficulty}")
    
    if len(test_questions) > 3:
        print(f"\n   ... และอีก {len(test_questions) - 3} คำถาม")
    
    print(f"\n   ✅ Converted: {len(test_questions)} questions (skipped {skipped} empty rows)")
    
    # Create output JSON
    output_json = {
        "test_questions": test_questions,
        "metadata": {
            "total_questions": len(test_questions),
            "categories": category_stats,
            "difficulty": difficulty_stats,
            "intents_coverage": intent_stats,
            "source": Path(excel_file).name,
            "converted_from": "Excel"
        }
    }
    
    # Save
    print(f"\n💾 Step 4: Saving to {output_file}...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_json, f, ensure_ascii=False, indent=2)
        print(f"   ✅ Saved successfully!")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None
    
    # Summary
    print(f"\n" + "="*80)
    print(f"📊 Summary")
    print(f"="*80)
    
    print(f"\n🎯 Intent Distribution:")
    for intent, count in sorted(intent_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(test_questions)) * 100
        print(f"   {intent:<20} : {count:>3} ({percentage:>5.1f}%)")
    
    print(f"\n📂 Category: {', '.join(f'{k}={v}' for k, v in category_stats.items())}")
    print(f"⚡ Difficulty: {', '.join(f'{k}={v}' for k, v in difficulty_stats.items())}")
    
    print(f"\n{'='*80}")
    print(f"✅ Conversion completed successfully!")
    print(f"   📄 Output: {output_file}")
    print(f"   📊 Questions: {len(test_questions)}")
    print(f"{'='*80}\n")
    
    return output_file


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='แปลง Excel เป็น test_questions.json สำหรับ RAGAS (ครบวงจร)'
    )
    parser.add_argument(
        'excel_file',
        help='Path to Excel file'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output JSON file (default: test_questions_new.json)',
        default='test_questions_new.json'
    )
    parser.add_argument(
        '--replace',
        help='Replace existing test_questions.json',
        action='store_true'
    )
    
    args = parser.parse_args()
    
    # Convert
    output_file = convert_excel_to_test_questions(args.excel_file, args.output)
    
    if output_file:
        print("\n💡 Next Steps:")
        
        if args.replace:
            import shutil
            backup = "test_questions_backup.json"
            target = "test_questions.json"
            
            if os.path.exists(target):
                shutil.copy(target, backup)
                print(f"\n   📦 Backed up: {target} → {backup}")
            
            shutil.copy(output_file, target)
            print(f"   ✅ Replaced: {target}")
            print(f"\n   🚀 Ready to evaluate:")
            print(f"      python evaluate_chatbots.py")
        else:
            print(f"\n   1. Review the file:")
            print(f"      notepad {output_file}")
            print(f"\n   2. Run evaluation:")
            print(f"      python evaluate_chatbots.py")
            print(f"\n   3. Or replace test_questions.json:")
            print(f"      python {sys.argv[0]} \"{args.excel_file}\" --replace")
        
        print()


if __name__ == "__main__":
    main()

