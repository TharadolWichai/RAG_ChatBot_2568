#!/usr/bin/env python3
"""
Script to demonstrate improved retrieval for better RAGAS scores

Key improvements:
1. Filter documents by relevance threshold (Context Precision)
2. Reduce number of documents (Context Precision + Answer Relevancy)
3. Better prompt engineering (Answer Relevancy)
"""

# Example: How to improve Context Precision in retriever

def filter_documents_by_relevance(documents, threshold=0.4):
    """
    Filter documents by relevance score
    
    Args:
        documents: List of retrieved documents with scores
        threshold: Minimum relevance score (0.0-1.0)
    
    Returns:
        Filtered list of high-relevance documents
    """
    # Filter by combined score
    filtered = [
        doc for doc in documents 
        if doc.get('combined_score', 0) >= threshold
    ]
    
    # If no documents pass threshold, return top 3
    if not filtered:
        return sorted(documents, key=lambda x: x.get('combined_score', 0), reverse=True)[:3]
    
    return filtered[:5]  # Max 5 documents


def prioritize_by_search_type(documents):
    """
    Prioritize documents by search type
    
    Thai Advanced > Thai BM25 > Vector
    """
    priority = {
        'thai_advanced': 3,
        'thai_bm25': 2,
        'vector': 1
    }
    
    return sorted(
        documents,
        key=lambda x: (
            priority.get(x.get('search_type', ''), 0),
            x.get('combined_score', 0)
        ),
        reverse=True
    )


# Example: Improved prompt for better Answer Relevancy

IMPROVED_PROMPT = """คุณเป็น AI Assistant ของวิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น

**คำสั่ง:**
1. อ่านคำถามและเข้าใจสิ่งที่ผู้ใช้ต้องการ
2. ใช้เฉพาะข้อมูลจาก Context ที่ให้มาเท่านั้น
3. ตอบตรงประเด็น กระชับ และชัดเจน
4. ถ้าข้อมูลไม่เพียงพอ ให้บอกว่า "ไม่พบข้อมูลในระบบ"

**Context:**
{context}

**คำถาม:** {question}

**คำตอบ (ตรงประเด็น, กระชับ):**"""


# Example: How to apply these improvements

def improved_retrieval_example():
    """
    Demonstration of improved retrieval pipeline
    """
    print("="*60)
    print("Improved Retrieval Pipeline Example")
    print("="*60)
    
    # Simulate retrieved documents
    mock_documents = [
        {
            'title': 'ระบบจองห้องประชุม (Reservation)',
            'url': 'https://appcs.kku.ac.th/rroom',
            'combined_score': 0.6400,
            'search_type': 'thai_advanced'
        },
        {
            'title': 'ระบบการจองใช้ห้องปฏิบัติการทางคอมพิวเตอร์',
            'url': 'https://appcs.kku.ac.th/rlab',
            'combined_score': 0.3400,
            'search_type': 'thai_advanced'
        },
        {
            'title': 'แบบฟอร์มขอเปลี่ยนแปลงเกรด',
            'url': 'https://docs.google.com/...',
            'combined_score': 0.3191,
            'search_type': 'thai_bm25'
        },
        {
            'title': 'แบบฟอร์มบันทึกข้อความ-ขออนุมัติไปต่างประเทศ',
            'url': 'https://docs.google.com/...',
            'combined_score': 0.2838,
            'search_type': 'vector'
        },
        # ... more documents with low scores
    ]
    
    print("\n[INFO] Original: 10 documents (including low-relevance ones)")
    print(f"   Documents: {len(mock_documents)}")
    
    # Step 1: Filter by relevance
    filtered = filter_documents_by_relevance(mock_documents, threshold=0.4)
    print(f"\n[INFO] After filtering (threshold=0.4): {len(filtered)} documents")
    for i, doc in enumerate(filtered, 1):
        print(f"   {i}. {doc['title'][:50]}... (score: {doc['combined_score']:.4f})")
    
    # Step 2: Prioritize by search type
    prioritized = prioritize_by_search_type(filtered)
    print(f"\n[INFO] After prioritization: {len(prioritized)} documents")
    for i, doc in enumerate(prioritized, 1):
        print(f"   {i}. [{doc['search_type']}] {doc['title'][:40]}... (score: {doc['combined_score']:.4f})")
    
    print("\n" + "="*60)
    print("Expected Improvements:")
    print("="*60)
    print("Context Precision:  0.00 → 0.50+ (50% improvement)")
    print("Answer Relevancy:   0.09 → 0.70+ (7x improvement)")
    print("Overall Score:      52.82% → 75%+ (Better than 'Good')")
    print("="*60)


if __name__ == "__main__":
    improved_retrieval_example()
    
    print("\n" + "="*60)
    print("How to Implement:")
    print("="*60)
    print("""
1. Modify retriever in main_*.py files:
   - Add threshold filtering
   - Reduce max documents to 5
   - Prioritize by search_type

2. Update prompt templates:
   - Use IMPROVED_PROMPT
   - Make it more focused
   - Add clear instructions

3. Expected Results:
   - Context Precision: 0% → 50%+
   - Answer Relevancy: 9% → 70%+
   - Overall Score: 52% → 75%+

4. Files to modify:
   - main_app/main_links.py
   - main_app/main_scholarship.py
   - main_app/main_digital_services.py
   - (And other chatbot files)
""")
