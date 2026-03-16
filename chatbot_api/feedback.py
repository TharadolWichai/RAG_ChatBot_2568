"""
Feedback API - สำหรับรับและเก็บ feedback จากผู้ใช้
"""
from fastapi import APIRouter, HTTPException
from chatbot_api.schemas import FeedbackRequest, FeedbackResponse
from datetime import datetime
from pathlib import Path
import json
import os

router = APIRouter()

# กำหนด folder สำหรับเก็บ feedback
FEEDBACK_DIR = Path("feedbacks/sessions")


def ensure_feedback_dir():
    """สร้าง folder feedbacks/sessions ถ้ายังไม่มี"""
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/api/v1/feedback", response_model=FeedbackResponse)
def save_feedback(req: FeedbackRequest):
    """
    บันทึก feedback จากผู้ใช้
    
    Parameters:
    - session_id: ID ของ session
    - question: คำถามที่ถาม
    - answer: คำตอบที่ได้
    - model: โมเดลที่ใช้
    - intent: Intent ที่จับได้
    - rating: คะแนน 1-5 ดาว
    - comment: ความคิดเห็น
    
    Returns:
    - status: ok/error
    - feedback_id: ID ของ feedback ที่บันทึก
    """
    try:
        # สร้าง folder ถ้ายังไม่มี
        ensure_feedback_dir()
        
        # ชื่อไฟล์ตาม session_id
        filepath = FEEDBACK_DIR / f"{req.session_id}.json"
        
        # โหลด feedback เดิม (ถ้ามี)
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            # สร้างใหม่
            data = {
                "session_id": req.session_id,
                "created_at": datetime.now().isoformat(),
                "feedbacks": []
            }
        
        # สร้าง feedback ใหม่
        feedback_id = f"fb_{len(data['feedbacks']) + 1:03d}"
        new_feedback = {
            "feedback_id": feedback_id,
            "timestamp": datetime.now().isoformat(),
            "question": req.question,
            "answer": req.answer,
            "model": req.model,
            "intent": req.intent,
            "rating": req.rating,
            "comment": req.comment
        }
        
        # เพิ่มเข้าไป
        data['feedbacks'].append(new_feedback)
        
        # อัพเดท summary
        ratings = [f['rating'] for f in data['feedbacks']]
        data['summary'] = {
            "total_feedbacks": len(data['feedbacks']),
            "avg_rating": sum(ratings) / len(ratings),
            "last_updated": datetime.now().isoformat()
        }
        
        # บันทึกลงไฟล์
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved feedback: {req.session_id} - {feedback_id} - Rating: {req.rating}⭐")
        
        return FeedbackResponse(
            status="ok",
            feedback_id=feedback_id,
            message=f"Feedback saved successfully (Total: {len(data['feedbacks'])})"
        )
        
    except Exception as e:
        print(f"❌ Error saving feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/feedback/stats")
def get_feedback_stats():
    """
    ดู statistics ของ feedback ทั้งหมด
    
    Returns:
    - total_feedbacks: จำนวน feedback ทั้งหมด
    - total_sessions: จำนวน session ทั้งหมด
    - avg_rating: คะแนนเฉลี่ย
    - satisfaction_rate: % ของ 4-5 ดาว
    - distribution: จำนวนแต่ละดาว
    """
    try:
        ensure_feedback_dir()
        
        all_ratings = []
        total_sessions = 0
        
        # อ่านทุกไฟล์
        for filepath in FEEDBACK_DIR.glob("*.json"):
            total_sessions += 1
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for fb in data.get('feedbacks', []):
                    all_ratings.append(fb['rating'])
        
        if not all_ratings:
            return {
                "total_feedbacks": 0,
                "total_sessions": 0,
                "message": "No feedback yet"
            }
        
        # คำนวณ metrics
        avg_rating = sum(all_ratings) / len(all_ratings)
        satisfied = len([r for r in all_ratings if r >= 4])
        satisfaction_rate = (satisfied / len(all_ratings)) * 100
        
        return {
            "total_feedbacks": len(all_ratings),
            "total_sessions": total_sessions,
            "avg_rating": round(avg_rating, 2),
            "satisfaction_rate": round(satisfaction_rate, 1),
            "distribution": {
                "5_star": all_ratings.count(5),
                "4_star": all_ratings.count(4),
                "3_star": all_ratings.count(3),
                "2_star": all_ratings.count(2),
                "1_star": all_ratings.count(1),
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/feedback/export")
def export_feedbacks():
    """
    Export ทุก feedback เป็น CSV สำหรับดาวน์โหลด
    """
    try:
        from io import StringIO
        import csv
        from fastapi.responses import StreamingResponse
        
        ensure_feedback_dir()
        
        all_feedbacks = []
        
        # อ่านทุกไฟล์
        for filepath in FEEDBACK_DIR.glob("*.json"):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for fb in data.get('feedbacks', []):
                    all_feedbacks.append({
                        'session_id': data['session_id'],
                        'feedback_id': fb['feedback_id'],
                        'timestamp': fb['timestamp'],
                        'question': fb['question'],
                        'answer': fb['answer'][:100] + '...' if len(fb['answer']) > 100 else fb['answer'],
                        'model': fb.get('model', 'N/A'),
                        'intent': fb.get('intent', 'N/A'),
                        'rating': fb['rating'],
                        'comment': fb['comment']
                    })
        
        if not all_feedbacks:
            raise HTTPException(status_code=404, detail="No feedback found")
        
        # สร้าง CSV (UTF-8 + BOM เพื่อให้ Excel เปิดภาษาไทยได้ถูกต้อง)
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=all_feedbacks[0].keys())
        writer.writeheader()
        writer.writerows(all_feedbacks)
        csv_content = output.getvalue()
        # เพิ่ม BOM ให้ Excel รู้ว่าเป็น UTF-8
        body = ("\ufeff" + csv_content).encode("utf-8")
        
        return StreamingResponse(
            iter([body]),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f"attachment; filename=feedbacks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
