#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
csv_to_test_questions.py - แปลง CSV เป็น test_questions.json

แปลงไฟล์ CSV ที่มีคำถามและ ground truth เป็น JSON
กรองเฉพาะแถวที่มีข้อมูลครบถ้วน
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
    print("กรุณาติดตั้งด้วยคำสั่ง: pip install pandas")
    sys.exit(1)


# Intent classification rules (same as before)
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
        r'กลุ่มวิจัย', r'research.*group', r'aida', r'dsai', r'isne', r'i-serg',
        r'งานวิจัย(?!.*อาจารย์)', r'ผลงาน.*วิจัย', r'โครงการ.*วิจัย',
        r'หัวหน้า.*กลุ่ม', r'สมาชิก.*กลุ่ม'
    ],
    "student_club": [
        r'สโมสร', r'ประธาน.*สโมสร', r'กิจกรรม.*นักศึกษา', r'ชุมนุม',
        r'องค์กร.*นักศึกษา', r'คณะกรรมการ.*นักศึกษา'
    ],
    "students": [
        r'ลิงก์.*นักศึกษา', r'ระบบ.*นักศึกษา', r'โครงงาน', r'project',
        r'ลงทะเบียน', r'ตารางเรียน', r'ตาราง.*สอบ', r'eproject'
    ],
    "digital_services": [
        r'web.*hosting', r'virtual.*machine', r'vm', r'apple.*store',
        r'บริการ.*ดิจิตอล', r'บริการ.*ออนไลน์', r'cloud', r'server',
        r'เครื่องเสมือน', r'โฮสติ้ง', r'chatgpt', r'google.*play', r'grammarly'
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
    "graduate": [
        r'ปริญญาโท', r'ปริญญาเอก', r'บัณฑิตศึกษา', r'm\.sc\.', r'ph\.d\.',
        r'master', r'doctoral', r'dissertation', r'geo-informatics',
        r'data science.*artificial intelligence'
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


def convert_csv_to_test_questions(csv_file: str, output_file: str = "test_questions_from_csv.json"):
    """
    แปลง CSV เป็น test_questions.json
    กรองเฉพาะแถวที่มีข้อมูลครบถ้วน
    
    Args:
        csv_file: path to CSV file
        output_file: path to output JSON file
    
    Returns:
        path to output file or None if failed
    """
    
    print("\n" + "="*80)
    print("📊 CSV to Test Questions Converter")
    print("="*80)
    
    # Try different encodings
    encodings = ['utf-8-sig', 'utf-8', 'cp874', 'tis-620', 'latin1']
    df = None
    
    print(f"\n📖 Step 1: Reading CSV file...")
    print(f"   File: {csv_file}")
    
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_file, encoding=encoding)
            print(f"   ✅ Success with encoding: {encoding}")
            print(f"   📊 Rows: {len(df)}, Columns: {len(df.columns)}")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"   ⚠️  Error with {encoding}: {e}")
            continue
    
    if df is None:
        print(f"   ❌ Error: ไม่สามารถอ่านไฟล์ได้")
        return None
    
    # Show columns
    print(f"\n📋 Columns detected:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i}. {col}")
    
    # Find question and ground_truth columns
    print(f"\n🔍 Step 2: Finding question and ground_truth columns...")
    
    question_col = None
    ground_truth_col = None
    
    # Find columns by index or name
    for col in df.columns:
        col_str = str(col).lower()
        if 'question' in col_str or 'คำถาม' in col_str:
            question_col = col
            print(f"   ✅ Question column: {col}")
        elif 'ground' in col_str or 'truth' in col_str or 'คำตอบ' in col_str:
            ground_truth_col = col
            print(f"   ✅ Ground truth column: {col}")
    
    # If not found by name, use index (column 1 = question, column 3 = ground_truth)
    if question_col is None and len(df.columns) >= 2:
        question_col = df.columns[1]  # Second column
        print(f"   ✅ Using column index 1 for questions: {question_col}")
    
    if ground_truth_col is None and len(df.columns) >= 4:
        ground_truth_col = df.columns[3]  # Fourth column
        print(f"   ✅ Using column index 3 for ground_truth: {ground_truth_col}")
    
    if question_col is None:
        print(f"   ❌ Error: ไม่พบ column สำหรับคำถาม")
        return None
    
    if ground_truth_col is None:
        print(f"   ⚠️  Warning: ไม่พบ column สำหรับ ground_truth")
    
    # Convert and classify
    print(f"\n🔄 Step 3: Converting and filtering...")
    test_questions = []
    skipped_empty_question = 0
    skipped_empty_ground_truth = 0
    
    intent_stats = {}
    category_stats = {}
    difficulty_stats = {}
    
    for idx, row in df.iterrows():
        # Get question
        question = row[question_col]
        
        # Skip if question is empty
        if pd.isna(question) or str(question).strip() == '':
            skipped_empty_question += 1
            continue
        
        question = str(question).strip()
        
        # Get ground truth
        ground_truth = ""
        if ground_truth_col:
            gt = row[ground_truth_col]
            if not pd.isna(gt):
                ground_truth = str(gt).strip()
        
        # Skip if ground_truth is empty (ตามที่ user ต้องการ)
        if not ground_truth or ground_truth == '':
            skipped_empty_ground_truth += 1
            continue
        
        # Classify
        intent = classify_intent(question)
        category = classify_category(question)
        difficulty = classify_difficulty(question)
        
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
        
        # Preview first 5
        if len(test_questions) <= 5:
            print(f"\n   [{len(test_questions)}] {question[:60]}{'...' if len(question) > 60 else ''}")
            print(f"       → {intent} | {category} | {difficulty}")
            print(f"       GT: {ground_truth[:80]}{'...' if len(ground_truth) > 80 else ''}")
    
    if len(test_questions) > 5:
        print(f"\n   ... และอีก {len(test_questions) - 5} คำถาม")
    
    print(f"\n   ✅ Converted: {len(test_questions)} questions")
    print(f"   ⚠️  Skipped:")
    print(f"      - Empty questions: {skipped_empty_question}")
    print(f"      - Empty ground_truth: {skipped_empty_ground_truth}")
    print(f"      - Total skipped: {skipped_empty_question + skipped_empty_ground_truth}")
    
    if len(test_questions) == 0:
        print("\n   ❌ Error: ไม่มีคำถามที่ใช้งานได้")
        return None
    
    # Create output JSON
    output_json = {
        "test_questions": test_questions,
        "metadata": {
            "total_questions": len(test_questions),
            "categories": category_stats,
            "difficulty": difficulty_stats,
            "intents_coverage": intent_stats,
            "source": Path(csv_file).name,
            "converted_from": "CSV"
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
        bar = '█' * int(percentage / 2)
        print(f"   {intent:<20} : {count:>3} ({percentage:>5.1f}%) {bar}")
    
    print(f"\n📂 Category Distribution:")
    for category, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(test_questions)) * 100
        print(f"   {category:<20} : {count:>3} ({percentage:>5.1f}%)")
    
    print(f"\n⚡ Difficulty Distribution:")
    for difficulty, count in sorted(difficulty_stats.items(), key=lambda x: ['easy', 'medium', 'hard'].index(x[0])):
        percentage = (count / len(test_questions)) * 100
        print(f"   {difficulty:<20} : {count:>3} ({percentage:>5.1f}%)")
    
    print(f"\n{'='*80}")
    print(f"✅ Conversion completed successfully!")
    print(f"   📄 Output: {output_file}")
    print(f"   📊 Valid questions: {len(test_questions)}")
    print(f"   🗑️  Skipped (empty data): {skipped_empty_question + skipped_empty_ground_truth}")
    print(f"{'='*80}\n")
    
    return output_file


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='แปลง CSV เป็น test_questions.json (กรองเฉพาะข้อมูลที่ครบถ้วน)'
    )
    parser.add_argument(
        'csv_file',
        help='Path to CSV file'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output JSON file (default: test_questions_from_csv.json)',
        default='test_questions_from_csv.json'
    )
    parser.add_argument(
        '--replace',
        help='Replace existing test_questions.json',
        action='store_true'
    )
    
    args = parser.parse_args()
    
    # Convert
    output_file = convert_csv_to_test_questions(args.csv_file, args.output)
    
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
            print(f"      cd evaluation")
            print(f"      python evaluate_chatbots.py")
        else:
            print(f"\n   1. Review the file:")
            print(f"      notepad {output_file}")
            print(f"\n   2. Move to evaluation folder and rename:")
            print(f"      move {output_file} evaluation\\test_questions.json")
            print(f"\n   3. Run evaluation:")
            print(f"      cd evaluation")
            print(f"      python evaluate_chatbots.py")
            print(f"\n   4. Or run with --replace flag:")
            print(f"      python {sys.argv[0]} \"{args.csv_file}\" --replace")
        
        print()


if __name__ == "__main__":
    main()

