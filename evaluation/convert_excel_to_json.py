#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_excel_to_json.py - แปลงไฟล์ Excel เป็น JSON สำหรับ RAGAS evaluation

ใช้สำหรับแปลงไฟล์ Excel ที่มีคำถามและ ground truth
ให้เป็น format JSON ที่ใช้กับ evaluate_chatbots.py
"""

import sys
import json
from pathlib import Path

# Fix Windows encoding issue
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    print("❌ Error: pandas ไม่ได้ติดตั้ง")
    print("\nกรุณาติดตั้งด้วยคำสั่ง:")
    print("   pip install pandas openpyxl")
    sys.exit(1)


def convert_excel_to_json(excel_file: str, output_file: str = None):
    """
    แปลงไฟล์ Excel เป็น JSON
    
    Args:
        excel_file: path to Excel file
        output_file: output JSON file path (optional)
    """
    
    print(f"\n{'='*80}")
    print("📊 Excel to JSON Converter for RAGAS Evaluation")
    print(f"{'='*80}")
    
    # อ่านไฟล์ Excel
    print(f"\n📖 Reading Excel file: {excel_file}")
    try:
        df = pd.read_excel(excel_file)
        print(f"✅ Successfully read Excel file")
        print(f"   📊 Rows: {len(df)}, Columns: {len(df.columns)}")
    except FileNotFoundError:
        print(f"❌ Error: ไม่พบไฟล์ {excel_file}")
        return None
    except Exception as e:
        print(f"❌ Error reading Excel: {e}")
        return None
    
    # แสดง columns
    print(f"\n📋 Columns in Excel:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i}. {col}")
    
    # ตรวจสอบว่ามี columns ที่จำเป็นหรือไม่
    required_columns = {
        'question': ['question', 'คำถาม', 'Question', 'คำถาม/question'],
        'ground_truth': ['ground_truth', 'answer', 'คำตอบ', 'Ground Truth', 'คำตอบที่ถูกต้อง']
    }
    
    # หา column mapping
    column_mapping = {}
    
    # หา question column
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if any(keyword in col_lower for keyword in ['question', 'คำถาม']):
            column_mapping['question'] = col
            break
    
    # หา ground_truth column
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if any(keyword in col_lower for keyword in ['ground', 'truth', 'answer', 'คำตอบ', 'เฉลย']):
            if col != column_mapping.get('question'):  # ไม่ใช่ column เดียวกับ question
                column_mapping['ground_truth'] = col
                break
    
    # หา expected_intent column (optional)
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if any(keyword in col_lower for keyword in ['intent', 'category', 'ประเภท', 'หมวดหมู่']):
            column_mapping['expected_intent'] = col
            break
    
    # หา difficulty column (optional)
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if any(keyword in col_lower for keyword in ['difficulty', 'level', 'ระดับ', 'ความยาก']):
            column_mapping['difficulty'] = col
            break
    
    print(f"\n🔍 Column Mapping:")
    for key, value in column_mapping.items():
        print(f"   {key}: {value}")
    
    # ตรวจสอบว่ามี columns ที่จำเป็น
    if 'question' not in column_mapping:
        print("\n❌ Error: ไม่พบ column สำหรับ 'question'")
        print("   กรุณาตรวจสอบว่าไฟล์ Excel มี column ที่มีชื่อว่า 'question' หรือ 'คำถาม'")
        return None
    
    if 'ground_truth' not in column_mapping:
        print("\n⚠️  Warning: ไม่พบ column สำหรับ 'ground_truth'")
        print("   จะใช้ค่า default แทน")
    
    # แปลงเป็น JSON format
    print(f"\n🔄 Converting to JSON format...")
    
    test_questions = []
    skipped = 0
    
    for idx, row in df.iterrows():
        # อ่านคำถาม
        question = row[column_mapping['question']]
        
        # ข้าม row ที่ว่างเปล่า
        if pd.isna(question) or str(question).strip() == '':
            skipped += 1
            continue
        
        # สร้าง test case
        test_case = {
            "id": idx + 1,
            "question": str(question).strip(),
            "expected_intent": "unknown",  # default
            "ground_truth": "",
            "category": "general",  # default
            "difficulty": "medium"  # default
        }
        
        # เพิ่ม ground_truth ถ้ามี
        if 'ground_truth' in column_mapping:
            ground_truth = row[column_mapping['ground_truth']]
            if not pd.isna(ground_truth):
                test_case['ground_truth'] = str(ground_truth).strip()
        
        # เพิ่ม expected_intent ถ้ามี
        if 'expected_intent' in column_mapping:
            intent = row[column_mapping['expected_intent']]
            if not pd.isna(intent):
                test_case['expected_intent'] = str(intent).strip().lower()
        
        # เพิ่ม difficulty ถ้ามี
        if 'difficulty' in column_mapping:
            difficulty = row[column_mapping['difficulty']]
            if not pd.isna(difficulty):
                test_case['difficulty'] = str(difficulty).strip().lower()
        
        # ถ้าไม่มี ground_truth ให้สร้าง default
        if not test_case['ground_truth']:
            test_case['ground_truth'] = f"ควรตอบคำถาม: {test_case['question']}"
        
        test_questions.append(test_case)
    
    print(f"\n✅ Conversion completed!")
    print(f"   📊 Total questions: {len(test_questions)}")
    print(f"   ⚠️  Skipped empty rows: {skipped}")
    
    # สร้าง metadata
    categories = {}
    difficulties = {}
    intents = {}
    
    for q in test_questions:
        categories[q['category']] = categories.get(q['category'], 0) + 1
        difficulties[q['difficulty']] = difficulties.get(q['difficulty'], 0) + 1
        intents[q['expected_intent']] = intents.get(q['expected_intent'], 0) + 1
    
    # สร้าง output JSON
    output_json = {
        "test_questions": test_questions,
        "metadata": {
            "total_questions": len(test_questions),
            "categories": categories,
            "difficulty": difficulties,
            "intents_coverage": intents,
            "source": Path(excel_file).name,
            "converted_from": "Excel"
        }
    }
    
    # กำหนด output file
    if output_file is None:
        excel_path = Path(excel_file)
        output_file = excel_path.parent / f"{excel_path.stem}_converted.json"
    
    # บันทึกไฟล์
    print(f"\n💾 Saving to: {output_file}")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_json, f, ensure_ascii=False, indent=2)
        print(f"✅ Successfully saved JSON file")
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return None
    
    # แสดง preview
    print(f"\n📋 Preview (first 3 questions):")
    print(f"{'-'*80}")
    for i, q in enumerate(test_questions[:3], 1):
        print(f"\n{i}. Question: {q['question']}")
        print(f"   Ground Truth: {q['ground_truth'][:100]}{'...' if len(q['ground_truth']) > 100 else ''}")
        print(f"   Intent: {q['expected_intent']}")
        print(f"   Difficulty: {q['difficulty']}")
    
    # แสดง metadata
    print(f"\n{'-'*80}")
    print(f"\n📊 Metadata Summary:")
    print(f"   Total Questions: {output_json['metadata']['total_questions']}")
    print(f"\n   Categories:")
    for cat, count in categories.items():
        print(f"      - {cat}: {count}")
    print(f"\n   Difficulty:")
    for diff, count in difficulties.items():
        print(f"      - {diff}: {count}")
    print(f"\n   Intents:")
    for intent, count in intents.items():
        print(f"      - {intent}: {count}")
    
    print(f"\n{'='*80}")
    print(f"✅ Conversion completed successfully!")
    print(f"   📄 Output file: {output_file}")
    print(f"   📊 Questions: {len(test_questions)}")
    print(f"{'='*80}\n")
    
    return output_file


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='แปลงไฟล์ Excel เป็น JSON สำหรับ RAGAS evaluation'
    )
    parser.add_argument(
        'excel_file',
        help='Path to Excel file'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output JSON file path (optional)',
        default=None
    )
    
    args = parser.parse_args()
    
    # Convert
    output_file = convert_excel_to_json(args.excel_file, args.output)
    
    if output_file:
        print("\n💡 How to use this file:")
        print(f"\n1. ตรวจสอบไฟล์ JSON:")
        print(f"   cat {output_file}")
        print(f"\n2. ใช้กับ evaluation script:")
        print(f"   python evaluate_chatbots.py --questions {output_file}")
        print(f"\n3. หรือแก้ไขชื่อเป็น test_questions.json:")
        print(f"   mv {output_file} test_questions.json")


if __name__ == "__main__":
    main()

