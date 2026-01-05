#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
excel_to_test_questions.py - One-stop solution for converting Excel to RAGAS test questions

แปลงไฟล์ Excel เป็น JSON และ classify intent อัตโนมัติในขั้นตอนเดียว
"""

import sys
import os
import json
import re
from pathlib import Path

# Fix Windows encoding issue (do this ONCE at the top level)
if sys.platform == 'win32':
    import io
    if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    if not isinstance(sys.stderr, io.TextIOWrapper) or sys.stderr.encoding != 'utf-8':
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

try:
    import pandas as pd
except ImportError:
    print("❌ Error: pandas ไม่ได้ติดตั้ง")
    print("กรุณาติดตั้งด้วยคำสั่ง: pip install pandas openpyxl")
    sys.exit(1)


def excel_to_test_questions(excel_file: str, output_name: str = "test_questions_new.json"):
    """
    แปลง Excel เป็น test questions JSON แบบครบวงจร
    
    Args:
        excel_file: path to Excel file
        output_name: ชื่อไฟล์ output (default: test_questions_new.json)
    """
    
    print("\n" + "="*80)
    print("🚀 Excel to Test Questions - Complete Workflow")
    print("="*80)
    print()
    print("📋 Workflow:")
    print("   1. Convert Excel to JSON")
    print("   2. Auto-classify intents")
    print("   3. Generate final test_questions.json")
    print()
    print("="*80)
    
    # Step 1: Convert Excel to JSON
    print("\n" + "="*80)
    print("📊 Step 1: Converting Excel to JSON...")
    print("="*80)
    
    temp_json = convert_excel_to_json(excel_file, output_file="temp_converted.json")
    
    if not temp_json:
        print("\n❌ Failed to convert Excel to JSON")
        return None
    
    # Step 2: Auto-classify intents
    print("\n" + "="*80)
    print("🤖 Step 2: Auto-classifying intents...")
    print("="*80)
    
    final_json = auto_classify_json(temp_json, output_file=output_name)
    
    if not final_json:
        print("\n❌ Failed to classify intents")
        return None
    
    # Cleanup temp file
    try:
        os.remove(temp_json)
        print(f"\n🗑️  Cleaned up temporary file: {temp_json}")
    except:
        pass
    
    # Summary
    print("\n" + "="*80)
    print("✅ Complete! Test Questions Ready")
    print("="*80)
    print(f"\n📄 Final output: {final_json}")
    print()
    
    return final_json


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='แปลง Excel เป็น test_questions.json สำหรับ RAGAS evaluation (ครบวงจร)'
    )
    parser.add_argument(
        'excel_file',
        help='Path to Excel file'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output JSON filename (default: test_questions_new.json)',
        default='test_questions_new.json'
    )
    parser.add_argument(
        '--replace',
        help='Replace existing test_questions.json',
        action='store_true'
    )
    
    args = parser.parse_args()
    
    # Convert
    output_file = excel_to_test_questions(args.excel_file, args.output)
    
    if output_file:
        print("\n💡 Next Steps:")
        print()
        
        if args.replace:
            # Replace existing test_questions.json
            import shutil
            backup_file = "test_questions_backup.json"
            target_file = "test_questions.json"
            
            if os.path.exists(target_file):
                shutil.copy(target_file, backup_file)
                print(f"   📦 Backed up existing file to: {backup_file}")
            
            shutil.copy(output_file, target_file)
            print(f"   ✅ Replaced {target_file} with new questions")
            print()
            print(f"   🚀 Ready to run evaluation:")
            print(f"      python evaluate_chatbots.py")
        else:
            print(f"   1. Review the generated file:")
            print(f"      notepad {output_file}")
            print()
            print(f"   2. Use with evaluation (specify file):")
            print(f"      python evaluate_chatbots.py")
            print()
            print(f"   3. Or replace existing test_questions.json:")
            print(f"      python excel_to_test_questions.py \"{args.excel_file}\" --replace")
        
        print()
        print("="*80)


if __name__ == "__main__":
    main()

