"""
eval_chatbot.py — Standalone Evaluation Chatbot (Terminal REPL)

ใช้สำหรับวัดผลโดยเฉพาะ แยก API key และ model ออกจากระบบ deploy
- Model  : gemini-2.5-flash
- API    : KKU Intelsphere (eval key)
- ตอบบน terminal แบบ interactive
- บันทึก Q&A + contexts เป็น JSON ทีละข้อ (เลือกได้)
"""

import os
import sys
import time
import json
from datetime import datetime

# ─── 1. ตั้ง path ให้หา module ของโปรเจกต์เจอ ───────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MAIN_APP_DIR = os.path.join(PROJECT_ROOT, "main_app")
for path in [PROJECT_ROOT, MAIN_APP_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

# ─── 2. โหลด .env (เพื่อเอา AstraDB config) ─────────────────────────────────
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ─── 3. Override ด้วยค่า evaluation โดยเฉพาะ (ต้องทำก่อน import chatbot) ────
EVAL_API_KEY   = "sk_jagl3qrwlrHFBJYG85fs9nIeVtZM90sq3nuZlnTCNWhMmBKvEU1kXHtf20OHR4Jo"
EVAL_BASE_URL  = "https://gen.ai.kku.ac.th/api/v1"
EVAL_MODEL     = "gemini-2.5-flash"

os.environ["OPENAI_API_KEY"]   = EVAL_API_KEY
os.environ["OPENAI_BASE_URL"]  = EVAL_BASE_URL
os.environ["CHATBOT_MODEL"]    = EVAL_MODEL
os.environ["OPENAI_MODEL"]     = EVAL_MODEL

# ─── 4. Import chatbot หลัก ───────────────────────────────────────────────────
print("กำลังโหลด chatbot...")
from main_unified_chatbot_automated import UnifiedChatbotAutomated

# ─── 5. Helpers ──────────────────────────────────────────────────────────────
def _print_contexts(contexts: list[str]) -> None:
    if not contexts:
        print("  (ไม่มี context ที่ดึงมา)")
        return
    for i, ctx in enumerate(contexts, 1):
        snippet = ctx[:300].replace("\n", " ")
        print(f"  [{i}] {snippet}{'...' if len(ctx) > 300 else ''}")


def _save_entry(filepath: str, entry: dict) -> None:
    """เพิ่ม entry เข้าไปใน JSON array ที่ไฟล์ filepath"""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []
    data.append(entry)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  บันทึกแล้ว → {filepath}  (รวม {len(data)} ข้อ)")

# ─── 6. REPL หลัก ────────────────────────────────────────────────────────────
def main() -> None:
    output_dir = os.path.join(PROJECT_ROOT, "evaluation", "eval_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # ถามว่าจะใช้ไฟล์ใหม่หรือไฟล์เดิม
    print("\n" + "=" * 60)
    print("  Evaluation Chatbot — KKU CS")
    print("=" * 60)
    print("\nเลือกโหมด:")
    print("  [1] สร้างไฟล์ใหม่")
    print("  [2] เพิ่มข้อมูลลงไฟล์เดิม")
    
    try:
        mode = input("\nเลือก (1/2): ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nออกจากโปรแกรม")
        return
    
    entry_index: int = 0
    
    if mode == "2":
        # แสดง list ไฟล์ที่มี
        existing_files = sorted([f for f in os.listdir(output_dir) if f.startswith("eval_session_") and f.endswith(".json")])
        
        if not existing_files:
            print("\n❌ ไม่มีไฟล์เดิม จะสร้างไฟล์ใหม่แทน")
            mode = "1"
        else:
            print("\nไฟล์ที่มีอยู่:")
            for i, fname in enumerate(existing_files, 1):
                # นับจำนวน entries ในไฟล์
                fpath = os.path.join(output_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        count = len(data) if isinstance(data, list) else 0
                except:
                    count = 0
                print(f"  [{i}] {fname} ({count} ข้อ)")
            
            try:
                file_idx = input(f"\nเลือกไฟล์ (1-{len(existing_files)}): ").strip()
                file_idx = int(file_idx)
                if 1 <= file_idx <= len(existing_files):
                    output_file = os.path.join(output_dir, existing_files[file_idx - 1])
                    # โหลดไฟล์เพื่อนับ entry_index
                    with open(output_file, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                        entry_index = len(existing_data) if isinstance(existing_data, list) else 0
                    print(f"\n✅ จะเพิ่มลงไฟล์: {existing_files[file_idx - 1]}")
                    print(f"   เริ่มต่อจาก index: {entry_index + 1}")
                else:
                    print("\n❌ เลือกไม่ถูกต้อง จะสร้างไฟล์ใหม่แทน")
                    mode = "1"
            except (ValueError, KeyboardInterrupt, EOFError):
                print("\n❌ เลือกไม่ถูกต้อง จะสร้างไฟล์ใหม่แทน")
                mode = "1"
    
    if mode == "1" or mode != "2":
        # สร้างไฟล์ใหม่
        session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"eval_session_{session_ts}.json")
        print(f"\n✅ จะสร้างไฟล์ใหม่: eval_session_{session_ts}.json")

    print("\n" + "=" * 60)
    print(f"  Model  : {EVAL_MODEL}")
    print(f"  API    : KKU Intelsphere (eval key)")
    print(f"  Output : {os.path.basename(output_file)}")
    print("=" * 60)
    print("คำสั่งพิเศษ:")
    print("  /exit      — ออกจากโปรแกรม")
    print("  /contexts  — แสดง context ของคำถามล่าสุด")
    print("  /help      — แสดงคำสั่ง")
    print("=" * 60 + "\n")

    chatbot = UnifiedChatbotAutomated()
    last_contexts: list[str] = []
    last_question: str = ""
    last_answer: str = ""
    last_elapsed: float = 0.0

    while True:
        try:
            question = input("คุณ: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nออกจากโปรแกรม")
            break

        if not question:
            continue

        # ─── คำสั่งพิเศษ ───────────────────────────────────────────────────
        if question.lower() in ("/exit", "/quit", "exit", "quit"):
            print("ออกจากโปรแกรม")
            break

        if question.lower() == "/contexts":
            print("\n── Contexts ของคำถามล่าสุด ──")
            _print_contexts(last_contexts)
            print()
            continue

        if question.lower() == "/help":
            print("\n  /exit     — ออก")
            print("  /contexts — contexts ล่าสุด")
            print("  /help     — แสดงคำสั่ง\n")
            continue

        # ─── ถามตอบปกติ ────────────────────────────────────────────────────
        print()
        t0 = time.perf_counter()
        try:
            answer, contexts, _ = chatbot.answer_with_contexts(question)
            last_contexts = contexts or []
        except Exception as exc:
            print(f"[ERROR] {exc}\n")
            continue

        elapsed = time.perf_counter() - t0
        last_question = question
        last_answer = answer
        last_elapsed = elapsed

        print(f"บอท: {answer}")
        print(f"\n  ⏱ {elapsed:.2f}s  |  contexts: {len(last_contexts)} รายการ")

        # ─── ถามว่าจะบันทึกมั้ย ────────────────────────────────────────────
        try:
            save_input = input("\n  บันทึกข้อนี้? [y/n]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nออกจากโปรแกรม")
            break

        if save_input == "y":
            entry_index += 1
            entry = {
                "index": entry_index,
                "timestamp": datetime.now().isoformat(),
                "question": last_question,
                "answer": last_answer,
                "contexts": last_contexts,
                "response_time_sec": round(last_elapsed, 3),
                "model": EVAL_MODEL,
            }
            _save_entry(output_file, entry)
        else:
            print("  ข้ามการบันทึก")
        print()


if __name__ == "__main__":
    main()
