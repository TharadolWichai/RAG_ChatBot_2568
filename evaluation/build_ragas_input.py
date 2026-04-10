"""
รวม eval_session + eval_dataset -> ไฟล์เดียวในรูปแบบ RAGAS (เหมือน ragas_input.json)
สำหรับอัปโหลดใน Google Colab หรือรัน evaluate_ragas.py (เปลี่ยน INPUT_FILE)

รัน: python evaluation/build_ragas_input.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "eval_results"

SESSION_FILE = RESULTS / "eval_session_20260324_110144.json"
DATASET_FILE = RESULTS / "eval_dataset_20260324.json"
OUTPUT_FILE = RESULTS / "ragas_input_20260324.json"


def main() -> None:
    session = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
    dataset = json.loads(DATASET_FILE.read_text(encoding="utf-8"))

    by_idx: dict = {r["index"]: r for r in dataset}
    rows: list = []
    for s in sorted(session, key=lambda x: x["index"]):
        i = s["index"]
        d = by_idx.get(i)
        if not d:
            raise SystemExit(f"eval_dataset ไม่มี index {i} — ให้ sync จำนวนแถวกับ session")
        q_sess = (s.get("question") or "").strip()
        q_ds = (d.get("question") or "").strip()
        if q_sess != q_ds:
            print(f"[warn] index {i}: question ต่างกันเล็กน้อย — ใช้ข้อความจาก session")
        rows.append(
            {
                "user_input": s["question"],
                "response": s.get("answer") or "",
                "retrieved_contexts": list(s.get("contexts") or []),
                "reference": d.get("expected_answer") or "",
            }
        )

    OUTPUT_FILE.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} rows -> {OUTPUT_FILE}")
    print("Colab: อัปโหลดไฟล์นี้แทน ragas_input.json แล้วตั้ง INPUT_FILE = 'ragas_input_20260324.json'")


if __name__ == "__main__":
    main()
