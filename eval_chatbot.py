"""
eval_chatbot.py — Standalone Evaluation Chatbot (Terminal REPL)

ใช้สำหรับวัดผลโดยเฉพาะ แยก API key และ model ออกจากระบบ deploy
- Model  : gemini-2.5-pro
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
EVAL_MODEL     = "gpt-5-mini"

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
    # กำหนดชื่อไฟล์ output ตาม timestamp ของ session นี้
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(PROJECT_ROOT, "evaluation", "eval_results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"eval_session_{session_ts}.json")

    print("\n" + "=" * 60)
    print("  Evaluation Chatbot — KKU CS")
    print(f"  Model  : {EVAL_MODEL}")
    print(f"  API    : KKU Intelsphere (eval key)")
    print(f"  Output : evaluation/eval_results/eval_session_{session_ts}.json")
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
    entry_index: int = 0

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
