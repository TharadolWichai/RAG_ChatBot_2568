#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
remap_intents_to_collections.py - จัดประเภท intent ให้ตรงกับ Collections

แมป intent เดิมให้ตรงกับชื่อ collections ที่มีใน ChromaDB
"""

import sys
import json

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Mapping จาก intent เดิมไปยัง collection names
INTENT_TO_COLLECTION_MAP = {
    # ตรงกันอยู่แล้ว
    "allpeople": "allpeople",
    "bsc_entrance": "bsc_entrance",
    "digital_services": "digital_services",
    "scholarship": "scholarship",
    "student_club": "student_club",
    
    # ต้องแมปใหม่
    "research": "researchgroup",  # research → researchgroup
    "students": "services",        # students → services (ลิงก์ระบบนักศึกษา)
    "links": "services",           # links → services
    "contact": "services",         # contact → services
    "news": "services",            # news → services
}

# Keywords สำหรับจัดประเภทคำถาม unknown
COLLECTION_KEYWORDS = {
    "graduate": [
        r'ปริญญาโท', r'ปริญญาเอก', r'บัณฑิตศึกษา', r'm\.sc\.', r'ph\.d\.',
        r'master', r'doctoral', r'dissertation', r'geo-informatics',
        r'data science.*artificial intelligence', r'ระดับบัณฑิต',
        r'หลักสูตรนานาชาติ.*เต็มจำนวน'
    ],
    "services": [
        r'ลิงก์', r'link', r'แบบฟอร์ม', r'ติดต่อ', r'contact',
        r'ห้องปฏิบัติการ', r'network', r'ระบบ', r'ขั้นตอน',
        r'แจ้งซ่อม', r'ถอนวิชา', r'เลื่อนสอบ', r'สำเร็จการศึกษา',
        r'สหกิจศึกษา', r'ลาพัก', r'เปลี่ยนสาขา', r'โอนรายวิชา',
        r'คุมสอบ', r'ข่าว', r'แบบฟอร์ม'
    ],
    "bsc_entrance": [
        r'รับเข้า', r'สมัคร', r'portfolio', r'โควตา', r'คะแนน',
        r'รอบที่', r'admission', r'ค่าเทอม', r'หลักสูตร.*ปริญญาตรี',
        r'netsat', r'tgat', r'a-level'
    ]
}


def classify_unknown_questions(question: str, current_intent: str) -> str:
    """
    จำแนกคำถามที่เป็น unknown ให้ตรงกับ collection
    
    Args:
        question: คำถาม
        current_intent: intent ปัจจุบัน
    
    Returns:
        collection name ใหม่
    """
    import re
    
    # ถ้าไม่ใช่ unknown ให้แมปตามปกติ
    if current_intent != "unknown":
        return INTENT_TO_COLLECTION_MAP.get(current_intent, current_intent)
    
    # จัดประเภทคำถาม unknown
    question_lower = question.lower()
    
    # ตรวจสอบแต่ละ collection
    for collection, keywords in COLLECTION_KEYWORDS.items():
        for pattern in keywords:
            if re.search(pattern, question, re.IGNORECASE):
                return collection
    
    # ถ้ายังไม่ตรง ให้เป็น services (บริการทั่วไป)
    return "services"


def remap_intents(input_file: str, output_file: str = None):
    """
    แมป intent ใหม่ให้ตรงกับ collection names
    
    Args:
        input_file: path to input JSON
        output_file: path to output JSON
    """
    
    print(f"\n{'='*80}")
    print("🔄 Remapping Intents to Collection Names")
    print(f"{'='*80}")
    
    # อ่าน JSON
    print(f"\n📖 Reading: {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"   ✅ Loaded {len(data['test_questions'])} questions")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None
    
    # แมป intents
    print(f"\n🔄 Remapping intents...")
    
    collection_stats = {}
    changes = []
    
    for i, q in enumerate(data['test_questions'], 1):
        old_intent = q['expected_intent']
        new_intent = classify_unknown_questions(q['question'], old_intent)
        
        # อัพเดท intent
        q['expected_intent'] = new_intent
        
        # สถิติ
        collection_stats[new_intent] = collection_stats.get(new_intent, 0) + 1
        
        # บันทึกการเปลี่ยนแปลง
        if old_intent != new_intent:
            changes.append({
                "id": i,
                "question": q['question'][:60] + "..." if len(q['question']) > 60 else q['question'],
                "old": old_intent,
                "new": new_intent
            })
    
    # แสดงการเปลี่ยนแปลง
    if changes:
        print(f"\n📝 Changes made ({len(changes)} questions):")
        for change in changes[:10]:  # แสดงแค่ 10 อันแรก
            print(f"\n   [{change['id']}] {change['question']}")
            print(f"       {change['old']} → {change['new']}")
        
        if len(changes) > 10:
            print(f"\n   ... และอีก {len(changes) - 10} คำถาม")
    else:
        print(f"\n   ✅ No changes needed (all intents already match collections)")
    
    # อัพเดท metadata
    data['metadata']['intents_coverage'] = collection_stats
    
    # กำหนด output file
    if output_file is None:
        from pathlib import Path
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_remapped.json"
    
    # บันทึก
    print(f"\n💾 Saving to: {output_file}")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"   ✅ Saved successfully!")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None
    
    # แสดง summary
    print(f"\n{'='*80}")
    print(f"📊 Collection Distribution")
    print(f"{'='*80}")
    
    # เรียงตาม collection name
    for collection in sorted(collection_stats.keys()):
        count = collection_stats[collection]
        percentage = (count / len(data['test_questions'])) * 100
        bar = '█' * int(percentage / 2)
        print(f"   {collection:<25} : {count:>3} ({percentage:>5.1f}%) {bar}")
    
    print(f"\n{'='*80}")
    print(f"✅ Remapping completed!")
    print(f"   📄 Output: {output_file}")
    print(f"   📊 Questions: {len(data['test_questions'])}")
    print(f"   🔄 Changes: {len(changes)}")
    print(f"{'='*80}\n")
    
    return output_file


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='แมป intent ให้ตรงกับ ChromaDB collection names'
    )
    parser.add_argument(
        'input_file',
        help='Path to input JSON file',
        nargs='?',
        default='evaluation/test_questions.json'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output JSON file (optional)',
        default=None
    )
    parser.add_argument(
        '--replace',
        help='Replace the input file',
        action='store_true'
    )
    
    args = parser.parse_args()
    
    # Remap
    if args.replace:
        output_file = args.input_file
    else:
        output_file = args.output
    
    result = remap_intents(args.input_file, output_file)
    
    if result:
        print("\n💡 Collections in ChromaDB:")
        print("   1. allpeople          - อาจารย์และบุคลากร")
        print("   2. bsc_entrance       - การรับเข้าปริญญาตรี")
        print("   3. digital_services   - บริการดิจิทัล")
        print("   4. graduate           - บัณฑิตศึกษา")
        print("   5. researchgroup      - กลุ่มวิจัย")
        print("   6. scholarship        - ทุนการศึกษา")
        print("   7. services           - บริการทั่วไป (ลิงก์, ติดต่อ, ข่าว)")
        print("   8. student_club       - สโมสรนักศึกษา")
        
        if not args.replace:
            print(f"\n📋 Next steps:")
            print(f"   1. Review the remapped file:")
            print(f"      notepad {result}")
            print(f"\n   2. Replace test_questions.json:")
            print(f"      copy {result} evaluation\\test_questions.json")
            print(f"\n   3. Or run with --replace flag:")
            print(f"      python {sys.argv[0]} {args.input_file} --replace")


if __name__ == "__main__":
    main()

