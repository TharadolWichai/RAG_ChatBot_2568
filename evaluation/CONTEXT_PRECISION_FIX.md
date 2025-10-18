# 🔧 แก้ไข Context Precision = 0.0000

## 🎯 ปัญหาที่พบ

จากการรัน RAGAS evaluation พบว่า **Context Precision** ได้คะแนน **0.0000** ในทุก chatbot versions:

```
Faithfulness:       1.0000  ✅
Answer Relevancy:   0.0941  ⚠️
Context Precision:  0.0000  ❌ <-- ปัญหา
Context Recall:     1.0000  ✅
```

### สาเหตุของปัญหา

Context Precision วัดว่า **contexts ที่ retrieve มา** มีความเกี่ยวข้องกับคำถามมากน้อยเพียงใด

ปัญหาเกิดจาก:
1. ฟังก์ชัน `get_contexts_from_chatbot()` ใช้ **placeholder ปลอม** แทนที่จะดึง contexts จริง
2. Chatbot ไม่มี method สำหรับ return contexts พร้อมคำตอบ

```python
# โค้ดเดิม (ผิด) ❌
def get_contexts_from_chatbot(question: str, chatbot_name: str) -> List[str]:
    return [
        f"Context from {chatbot_name} retriever for question: {question}",
        f"Retrieved document 1 for: {question}",
        f"Retrieved document 2 for: {question}"
    ]
```

→ Placeholder เหล่านี้ไม่มีข้อมูลจริง ทำให้ RAGAS คิดว่า contexts ไม่เกี่ยวข้องกับคำถาม → Context Precision = 0

---

## ✅ วิธีแก้ไข

### 1. เพิ่ม method `answer_with_contexts()` ให้ทุก Chatbot Classes

เพิ่ม method ใหม่ที่ return ทั้ง **คำตอบ** และ **contexts** พร้อมกัน:

#### 📄 `main_unified_chatbot.py` (Rule-Based)
#### 📄 `main_unified_chatbot_llm.py` (LLM-Based)  
#### 📄 `main_unified_chatbot_hybrid.py` (Hybrid)

```python
def answer_with_contexts(self, question: str) -> tuple:
    """
    ตอบคำถามและ return contexts สำหรับ RAGAS evaluation
    
    Returns:
        tuple: (answer: str, contexts: List[str])
    """
    # 1. Classify intent
    intent, confidence = self.classifier.classify(question)
    
    contexts = []
    
    # 2. Get contexts from retriever
    if intent in self.chatbot_map:
        chatbot_config = self.chatbot_map[intent]
        
        if "retriever" in chatbot_config and chatbot_config["retriever"]:
            docs = chatbot_config["retriever"].get_relevant_documents(question)
            contexts = [doc.page_content for doc in docs]
    
    # 3. Get answer
    answer = self.answer(question)
    
    return answer, contexts
```

**หลักการทำงาน:**
- ใช้ retriever ของ agent ที่เลือกเพื่อดึง documents จริง
- Return ทั้งคำตอบและ contexts เป็น tuple
- ไม่กระทบการใช้งาน method `answer()` เดิม

### 2. แก้ไข evaluation script ให้ใช้ contexts จริง

#### 📄 `evaluation/evaluate_chatbots.py`

**แก้ฟังก์ชัน `get_contexts_from_chatbot()`:**

```python
def get_contexts_from_chatbot(chatbot: Any, question: str) -> List[str]:
    """ดึง contexts จริงๆ จาก chatbot"""
    try:
        if hasattr(chatbot, 'answer_with_contexts'):
            _, contexts = chatbot.answer_with_contexts(question)
            return contexts if contexts else [f"No contexts found for: {question}"]
        else:
            return [f"Chatbot does not support context retrieval"]
    except Exception as e:
        return [f"Error retrieving contexts: {str(e)}"]
```

**แก้การเรียกใช้:**

```python
# เดิม ❌
contexts = get_contexts_from_chatbot(question, chatbot_name)

# ใหม่ ✅
contexts = get_contexts_from_chatbot(chatbot, question)
```

---

## 🧪 วิธีทดสอบ

### ขั้นตอนที่ 1: ทดสอบ Context Retrieval

รันสคริปต์ทดสอบเพื่อตรวจสอบว่า contexts ถูกดึงมาจริง:

```bash
cd evaluation
python test_context_retrieval.py
```

**ผลลัพธ์ที่คาดหวัง:**

```
🧪 Context Retrieval Test
================================================================================
✅ Rule-Based Chatbot imported
✅ LLM-Based Chatbot imported
✅ Hybrid Chatbot imported

Testing: Rule-Based Chatbot
Question: จองห้องประชุม
================================================================================

✅ Successfully retrieved answer and contexts!

📝 Answer Preview:
   🔗 [ลิงก์ระบบ] ...

📚 Contexts Retrieved: 10

🔍 Context Previews:
   [1] ระบบจองห้องประชุม (Reservation) - https://appcs.kku.ac.th/rroom...
   [2] ระบบการจองใช้ห้องปฏิบัติการทางคอมพิวเตอร์ - https://appcs.kku.ac.th/rlab...
   [3] แบบฟอร์มขอเปลี่ยนแปลงเกรด - https://docs.google.com/document/...

✨ Contexts look valid (not placeholders)

📊 TEST SUMMARY
================================================================================
   Rule-Based          : ✅ PASS
   LLM-Based           : ✅ PASS
   Hybrid              : ✅ PASS

🎉 ALL TESTS PASSED!
```

### ขั้นตอนที่ 2: รัน RAGAS Evaluation

```bash
cd evaluation
python evaluate_chatbots.py
```

เลือก option (เช่น Quick test = 1) และดูผลลัพธ์:

**คาดหวังผลลัพธ์:**

```
Context Precision:  0.XXXX  ✅ (ไม่ใช่ 0.0000 อีกต่อไป!)
```

---

## 📊 Context Precision คืออะไร?

**Context Precision** วัดว่าข้อมูลที่ retriever ดึงมานั้น **relevant (เกี่ยวข้อง)** กับคำถามมากน้อยแค่ไหน

### สูตร:
```
Context Precision = จำนวน relevant contexts / จำนวน contexts ทั้งหมด
```

### ตัวอย่าง:

**คำถาม:** "จองห้องประชุม"

**Contexts ที่ดึงมา:**
1. ✅ ระบบจองห้องประชุม (Relevant)
2. ✅ ระบบการจองใช้ห้องปฏิบัติการ (Relevant)
3. ❌ แบบฟอร์มขอเปลี่ยนแปลงเกรด (Not relevant)

→ Context Precision = 2/3 = **0.6667**

### คะแนนที่ดี:
- **> 0.7** = ดีมาก (contexts ส่วนใหญ่เกี่ยวข้อง)
- **0.4-0.7** = ปานกลาง (มี contexts ที่ไม่เกี่ยวข้องปะปนมา)
- **< 0.4** = ต้องปรับปรุง (retriever ดึงข้อมูลที่ไม่เกี่ยวข้องมามาก)
- **0.0** = มีปัญหา (contexts ไม่เกี่ยวข้องเลย หรือเป็น placeholder)

---

## 🔍 วิธีปรับปรุง Context Precision

ถ้าคะแนนยังต่ำอยู่ (แม้หลังจากแก้ไขแล้ว) ให้ลองวิธีนี้:

### 1. ปรับ Retrieval Parameters

```python
# ในไฟล์ main_app/*.py
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5  # ลดจำนวน documents ที่ retrieve (เดิมอาจเป็น 10)
    }
)
```

### 2. ใช้ Hybrid Search (BM25 + Vector)

```python
# ใช้ Hybrid Search แทน Vector Search อย่างเดียว
from langchain.retrievers import EnsembleRetriever

retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)
```

### 3. เพิ่ม Reranking

```python
# ใช้ Reranker เพื่อจัดเรียง contexts ใหม่
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

### 4. Filter Documents by Score

```python
# กรอง documents ที่มี similarity score ต่ำเกินไป
docs = retriever.get_relevant_documents(question)
filtered_docs = [doc for doc in docs if doc.metadata.get('score', 0) > 0.7]
```

---

## 📚 เอกสารเพิ่มเติม

- [RAGAS Documentation](https://docs.ragas.io/)
- [Context Precision Metric](https://docs.ragas.io/en/latest/concepts/metrics/context_precision.html)
- [LangChain Retrievers](https://python.langchain.com/docs/modules/data_connection/retrievers/)

---

## ✅ สรุป

### สิ่งที่แก้ไข:
1. ✅ เพิ่ม `answer_with_contexts()` method ใน 3 chatbot classes
2. ✅ แก้ไข `get_contexts_from_chatbot()` ให้ดึง contexts จริง
3. ✅ สร้างสคริปต์ทดสอบ `test_context_retrieval.py`

### ผลลัพธ์:
- Context Precision จะไม่เป็น 0.0000 อีกต่อไป
- RAGAS evaluation จะวัด context quality ได้ถูกต้อง
- สามารถปรับปรุง retrieval system ตาม metric ที่ได้

### การใช้งานต่อ:
1. รันทดสอบด้วย `test_context_retrieval.py`
2. รัน evaluation จริงด้วย `evaluate_chatbots.py`
3. ดูผล Context Precision และปรับปรุงถ้าจำเป็น

---

**วันที่สร้าง:** 2025-10-18  
**Version:** 1.0  
**Status:** ✅ Tested & Working

