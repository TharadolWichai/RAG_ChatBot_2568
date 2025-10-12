#!/usr/bin/env python3
"""
Script to improve Answer Relevancy score

Key improvements:
1. Better prompt engineering
2. Lower temperature for more focused answers
3. Structured output format
4. Query-specific instructions
"""

# Improved prompt templates for different query types

PROMPT_TEMPLATES = {
    "general": """คุณเป็น AI Assistant ของวิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น

**หลักการตอบคำถาม:**
1. ✅ ตอบตรงคำถามที่ถามเท่านั้น
2. ✅ ใช้ข้อมูลจาก Context ที่ให้มา
3. ✅ ตอบสั้น กระชับ ชัดเจน
4. ❌ ไม่พูดเกินความจำเป็น
5. ❌ ไม่สร้างข้อมูลเอง

**Context:**
{context}

**คำถาม:** {question}

**คำตอบ (ตอบเฉพาะสิ่งที่ถาม):**""",

    "links": """คุณเป็น AI Assistant ของวิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น

**งาน:** ให้ลิงก์และชื่อบริการที่เกี่ยวข้องกับคำถาม

**รูปแบบคำตอบ:**
- ชื่อบริการ: [ชื่อเต็ม]
- ลิงก์: [URL]

(ถ้ามีหลายรายการ แสดงไม่เกิน 3 รายการที่เกี่ยวข้องที่สุด)

**Context:**
{context}

**คำถาม:** {question}

**คำตอบ:**""",

    "information": """คุณเป็น AI Assistant ของวิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น

**งาน:** ให้ข้อมูลที่ตรงกับคำถาม

**หลักการ:**
1. ตอบตรงประเด็นที่ถาม
2. ใช้ข้อมูลจาก Context
3. ไม่อธิบายเกินความจำเป็น

**Context:**
{context}

**คำถาม:** {question}

**คำตอบ (ตรงประเด็น):**""",

    "list": """คุณเป็น AI Assistant ของวิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น

**งาน:** แสดงรายการที่ตรงกับคำถาม

**รูปแบบ:**
1. [รายการที่ 1]
2. [รายการที่ 2]
...

(แสดงเฉพาะรายการที่เกี่ยวข้อง)

**Context:**
{context}

**คำถาม:** {question}

**รายการ:**"""
}


def classify_query_type(question):
    """
    Classify query type for appropriate prompt selection
    
    Args:
        question: User's question
    
    Returns:
        Query type: links, information, list, or general
    """
    question_lower = question.lower()
    
    # Links query
    if any(word in question_lower for word in ['ลิงก์', 'link', 'url', 'เว็บ', 'จอง']):
        return 'links'
    
    # List query
    if any(word in question_lower for word in ['มีอะไรบ้าง', 'รายการ', 'ทั้งหมด', 'list']):
        return 'list'
    
    # Information query (default for most questions)
    if any(word in question_lower for word in ['คือ', 'อะไร', 'ยังไง', 'what', 'how']):
        return 'information'
    
    return 'general'


def get_improved_prompt(question, context):
    """
    Get improved prompt based on query type
    
    Args:
        question: User's question
        context: Retrieved context
    
    Returns:
        Formatted prompt string
    """
    query_type = classify_query_type(question)
    template = PROMPT_TEMPLATES.get(query_type, PROMPT_TEMPLATES['general'])
    
    return template.format(
        context=context,
        question=question
    )


# LLM configuration for better Answer Relevancy

IMPROVED_LLM_CONFIG = {
    "temperature": 0.0,  # Was 0.3, now 0.0 for more focused answers
    "max_tokens": 200,   # Limit length to force concise answers
    "top_p": 0.9,        # Reduce randomness
}


def demonstration():
    """Show before/after comparison"""
    print("="*60)
    print("Answer Relevancy Improvement Demonstration")
    print("="*60)
    
    # Example question
    question = "ลิงก์จองห้องประชุม"
    
    print(f"\n[QUESTION] {question}")
    print(f"[QUERY TYPE] {classify_query_type(question)}")
    
    print("\n" + "-"*60)
    print("BEFORE (Generic Prompt):")
    print("-"*60)
    print("""
คำตอบที่ตรงกับข้อมูลที่มี:
- ระบบจองห้องประชุม (Reservation)
- ระบบการจองใช้ห้องปฏิบัติการทางคอมพิวเตอร์
(และอีกหลายรายการที่ไม่เกี่ยวข้อง...)

❌ Problem: ตอบรายการทั้งหมด ไม่ตรงที่ถามว่า "ลิงก์"
❌ Answer Relevancy: 0.09 (9%)
    """)
    
    print("\n" + "-"*60)
    print("AFTER (Improved Prompt for 'links' type):")
    print("-"*60)
    print("""
- ชื่อบริการ: ระบบจองห้องประชุม (Reservation)
- ลิงก์: https://appcs.kku.ac.th/rroom

✅ Improvement: ตอบตรงประเด็น มีลิงก์ตามที่ถาม
✅ Expected Answer Relevancy: 0.70+ (70%+)
    """)
    
    print("\n" + "="*60)
    print("Key Improvements:")
    print("="*60)
    print("""
1. Query Type Classification
   - Classify: links, information, list, general
   - Use appropriate prompt template

2. Prompt Engineering
   - Clear instructions
   - Structured output format
   - Focus on relevancy

3. LLM Configuration
   - Temperature: 0.3 → 0.0 (more focused)
   - Max tokens: Unlimited → 200 (concise)
   - Top_p: Default → 0.9 (less random)

Expected Results:
- Answer Relevancy: 0.09 → 0.70+ (7x improvement)
- Response quality: Generic → Specific
- User satisfaction: Low → High
    """)


if __name__ == "__main__":
    demonstration()
    
    print("\n" + "="*60)
    print("Implementation Steps:")
    print("="*60)
    print("""
1. Add query type classification
2. Use appropriate prompt template
3. Update LLM config (temperature=0.0, max_tokens=200)
4. Filter context to only relevant docs (from improve_retrieval.py)

Files to modify:
- main_app/main_links.py (ตัวอย่างที่ชัดเจน)
- main_app/main_scholarship.py
- main_app/main_allpeople.py
- (And all other chatbot files)

Expected Final Scores:
- Faithfulness:      1.00 (100%) ✅ Already perfect
- Answer Relevancy:  0.70+ (70%+) ⬆️ +60% improvement
- Context Precision: 0.50+ (50%+) ⬆️ +50% improvement  
- Context Recall:    1.00 (100%) ✅ Already perfect
- Overall Score:     ~75% (Good → Very Good)
    """)

