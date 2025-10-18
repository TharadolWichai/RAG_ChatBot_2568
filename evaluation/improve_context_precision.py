# improve_context_precision.py - ปรับปรุง Context Precision

"""
วิธีปรับปรุง Context Precision:

ปัญหาปัจจุบัน: Context Precision = 0.0
- ระบบดึงข้อมูล 10 รายการทั้งหมด
- ไม่ได้กรองเฉพาะข้อมูลที่เกี่ยวข้อง
- ส่งข้อมูลที่ไม่เกี่ยวข้องไปให้ LLM

วิธีแก้ไข:
1. เพิ่ม relevance threshold
2. จำกัดจำนวน documents
3. ปรับปรุง scoring
"""

def improve_retriever_settings():
    """ปรับปรุงการตั้งค่า retriever"""
    
    print("=== วิธีปรับปรุง Context Precision ===")
    print()
    
    print("1. เพิ่ม Relevance Threshold:")
    print("   - ตั้ง minimum_score = 0.3")
    print("   - กรองเฉพาะ documents ที่มีความเกี่ยวข้องสูง")
    print("   - ตัดข้อมูลที่มีคะแนนต่ำออก")
    print()
    
    print("2. จำกัดจำนวน Documents:")
    print("   - ลดจาก 10 เป็น 5 documents")
    print("   - เลือกเฉพาะ top 5 ที่มีคะแนนสูงสุด")
    print("   - ลด noise และเพิ่ม precision")
    print()
    
    print("3. ปรับปรุง Scoring Algorithm:")
    print("   - เพิ่ม weight ให้ keyword matching")
    print("   - ลด weight ของ vector similarity อย่างเดียว")
    print("   - เพิ่ม exact match bonus")
    print()
    
    print("4. Context Filtering:")
    print("   - ตรวจสอบความยาวของ context")
    print("   - ตัดทอน context ที่ยาวเกินไป")
    print("   - เน้นส่วนที่เกี่ยวข้องกับคำถาม")

def create_improved_retriever_code():
    """สร้างโค้ดสำหรับปรับปรุง retriever"""
    
    code = '''
# ในไฟล์ retriever (เช่น main_links.py)

def _get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
    """ปรับปรุงแล้ว: จำกัดจำนวน documents และเพิ่ม threshold"""
    
    # เพิ่ม relevance threshold
    MIN_RELEVANCE_SCORE = 0.3
    
    # ดึงข้อมูลแบบเดิม
    docs = self._get_comprehensive_search(query, k=10)
    
    # กรองเฉพาะ documents ที่มีคะแนนสูง
    filtered_docs = []
    for doc in docs:
        score = getattr(doc, 'score', 0.0)
        if score >= MIN_RELEVANCE_SCORE:
            filtered_docs.append(doc)
    
    # จำกัดจำนวน documents
    return filtered_docs[:5]
    '''
    
    print("=== โค้ดตัวอย่างการปรับปรุง ===")
    print(code)

if __name__ == "__main__":
    improve_retriever_settings()
    print()
    create_improved_retriever_code()
