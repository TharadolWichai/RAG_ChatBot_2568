# main_bsc_entrance.py - Full Debug Version
import os
from typing import List, Any

from astrapy import DataAPIClient
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# -------------------------------
# Embeddings
# -------------------------------
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -------------------------------
# AstraDB Setup
# -------------------------------
ASTRA_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
COLLECTION_NAME = "bsc_entrance_embedding"

if not ASTRA_TOKEN or not ASTRA_ENDPOINT:
    print("❌ Missing AstraDB credentials in .env")
    exit(1)

client = DataAPIClient(token=ASTRA_TOKEN)
database = client.get_database_by_api_endpoint(ASTRA_ENDPOINT)
collection = database.get_collection(COLLECTION_NAME)
print(f"✅ Connected to AstraDB Collection: {COLLECTION_NAME}")

# -------------------------------
# LLM Setup (OpenRouter)
# -------------------------------
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    print("⚠️ Warning: OPENROUTER_API_KEY not found, LLM responses will not work")
    llm = None
else:
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0,
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1"
    )
    print("✅ OpenRouter LLM initialized successfully")

# -------------------------------
# Custom Retriever with BM25 + Vector
# -------------------------------
class BSCEntranceRetriever(BaseRetriever):
    collection: Any = None
    embedding: Any = None
    bm25_retriever: Any = None
    documents_cache: Any = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, collection, embedding):
        super().__init__()
        self.collection = collection
        self.embedding = embedding
        self.bm25_retriever = None
        self.documents_cache = None

    # Initialize BM25
    def _ensure_bm25_initialized(self):
        if self.bm25_retriever is None:
            try:
                results = self.collection.find({}, limit=200)
                documents = [Document(page_content=r.get("content", ""), metadata=r.get("metadata", {})) for r in results]
                self.documents_cache = documents
                from langchain_community.retrievers import BM25Retriever
                if documents:
                    self.bm25_retriever = BM25Retriever.from_documents(documents)
                    self.bm25_retriever.k = 50  # เพิ่ม limit
                    print(f"🔧 BM25 retriever initialized with {len(documents)} documents")
            except Exception as e:
                print(f"❌ BM25 init error: {e}")
                self.bm25_retriever = None

    # Preprocess query
    def _preprocess_query_for_bm25(self, query: str) -> str:
        return query.replace("ขอข้อมูล", "").strip()

    # Text search BM25
    def _text_search(self, query: str) -> List[Document]:
        self._ensure_bm25_initialized()
        if not self.bm25_retriever:
            return []

        variants = [query, self._preprocess_query_for_bm25(query)]
        results = []
        seen = set()
        for v in variants:
            docs = self.bm25_retriever.get_relevant_documents(v)
            for d in docs:
                if d.page_content not in seen:
                    results.append(d)
                    seen.add(d.page_content)
        print(f"📊 BM25 เจอ {len(results)} documents")
        return results

    # Vector search
    def _vector_search(self, query: str) -> List[Document]:
        docs = []
        try:
            query_vector = self.embedding.embed_query(query)
            results = self.collection.find({}, sort={"$vector": query_vector}, limit=50)
            for r in results:
                doc = Document(page_content=r.get("content", ""), metadata=r.get("metadata", {}))
                doc.metadata["vector_score"] = r.get("vector_score", 0.0)  # ถ้ามี
                docs.append(doc)
            print(f"📊 Vector Search เจอ {len(docs)} documents")
        except Exception as e:
            print(f"❌ Vector search error: {e}")
        return docs

    # Extract keywords
    def _extract_search_keywords(self, query: str) -> List[str]:
        stop_words = ["ขอ", "ข้อมูล", "ข่าว", "ประกาศ", "ดู", "เกี่ยวกับ", "ใน", "ของ", "และ", "หรือ"]
        clean_query = query
        for sw in stop_words:
            clean_query = clean_query.replace(sw, " ")
        clean_query = " ".join(clean_query.split())
        keywords = [w for w in query.split() if w not in stop_words and len(w) >= 2]
        if clean_query: keywords.insert(0, clean_query)
        unique_keywords = []
        for k in keywords:
            if k not in unique_keywords:
                unique_keywords.append(k)
        print(f"📝 Keywords extracted: {unique_keywords}")
        return unique_keywords

    # Main get_relevant_documents
    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        print(f"🔍 Debug: กำลังค้นหาด้วย query: '{query}'")
        keywords = self._extract_search_keywords(query)

        # 1) Text Search
        text_docs = self._text_search(query)

        # 2) Vector Search
        vector_docs = self._vector_search(query)

        # 3) Merge with priority
        merged_docs = []
        seen_content = set()
        for doc in text_docs + vector_docs:
            if doc.page_content in seen_content:
                continue
            has_keyword = any(k.lower() in doc.page_content.lower() for k in keywords if len(k) >= 3)
            doc.metadata["priority"] = "high" if has_keyword else "normal"
            merged_docs.append(doc)
            seen_content.add(doc.page_content)

        merged_docs.sort(key=lambda d: 0 if d.metadata["priority"] == "high" else 1)
        print(f"🔄 รวมผลลัพธ์ทั้งหมด {len(merged_docs)} documents (แสดงสูงสุด 20)")

        # Debug: print top 5
        for i, d in enumerate(merged_docs[:5], 1):
            print(f"🏆 Top {i}: {d.page_content[:100]}... (priority: {d.metadata['priority']})")

        return merged_docs[:20]

retriever = BSCEntranceRetriever(collection, embedding)

# -------------------------------
# Prompt & QA Chain
# -------------------------------
PROMPT = PromptTemplate.from_template("""
บริบทต่อไปนี้คือข้อมูลประกาศ, ข่าว, และรายละเอียดเกณฑ์การรับเข้าศึกษา (คณะวิทยาการคอมพิวเตอร์ มข.)
คุณคือผู้ช่วยที่ให้ข้อมูลเกี่ยวกับการรับเข้าศึกษาระดับปริญญาตรีในคณะวิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น 
                        
สำคัญ: ตรวจสอบข้อมูลในบริบทอย่างละเอียด หากมีข้อมูลตรงกับคำถามให้นำมาตอบ

หากคำถามเกี่ยวกับวิธีการรับเข้าศึกษาระดับปริญญาตรี ให้แสดงผลแบบรายการที่ชัดเจน ดังนี้:
- ใช้หัวข้อชัดเจน เช่น "รอบที่ 1 Portfolio:"    
- แนบลิงค์ pdf หรือเว็บ (ถ้ามี) ของปี 2568 และปี 2569 ให้ครบทุกโครงการ และทุนผู้มีความาสารถดีเด่น ไม่ต้องเว้น undefin ned เพราะจะหาไม่เจอ
- แยกแต่ละรอบเป็นบรรทัดใหม่ 
- ใช้เครื่องหมาย • นำหน้ารอบย่อย
- แสดงข้อมูลที่มี เช่น เกณฑ์การคัดเลือก วิทยาการคอมพิวเตอร์ และเทคโนโลยีสารสนเทศ, วิทยาการข้อมูลและปัญญาประดิษฐ์, ระบบสารสนเทศ, เครือข่ายคอมพิวเตอร์และความมั่นคงปลอดภัยไซเบอร์, และอื่นๆ

หากคำถามเกี่ยวกับข้อมูลหรือเกณฑ์คะแนนเฉพาะของรอบใดรอบหนึ่ง เช่น รอบ 2 รอบ 3 ให้แสดงข้อมูลที่ครบถ้วน:
- เกณฑ์การคัดเลือก
- หลักสูตร วิชาที่ใช้สอบ
- คะแนนขั้นต่ำ - เอกสารประกอบการสมัคร
- ลิงก์ที่เกี่ยวข้อง (ถ้ามี)
หากไม่มีข้อมูลที่ตรงกับคำถาม ให้ตอบว่า "ขอโทษ ฉันไม่พบข้อมูลในระบบ"

---------------------
{context}
---------------------
คำถาม: {question}
คำตอบ (จัดรูปแบบให้อ่านง่ายและอ้างแหล่งข้อมูล):
""")

qa_chain = LLMChain(llm=llm, prompt=PROMPT)

# -------------------------------
# Manual QA Chain Function (for Unified Chatbot)
# -------------------------------
def manual_qa_chain(question: str) -> str:
    """
    ฟังก์ชันสำหรับ Unified Chatbot
    รับคำถาม -> ค้นหาข้อมูล -> สร้างคำตอบ
    """
    try:
        # Retrieve relevant documents
        docs = retriever.get_relevant_documents(question)
        
        if not docs:
            return "ขอโทษ ไม่พบข้อมูลการรับเข้าศึกษาที่ตรงกับคำถามของคุณ"
        
        # Build context from documents
        merged_context = "\n\n".join([f"ข้อมูล {i+1}:\n{d.page_content}" for i, d in enumerate(docs)])
        
        # Check if LLM is available
        if llm is None:
            # Fallback: Return raw context if no LLM
            return f"พบข้อมูลการรับเข้าศึกษา:\n\n{docs[0].page_content[:500]}..."
        
        # Generate answer using LLM
        response = qa_chain.run({"question": question, "context": merged_context})
        return response
        
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการค้นหาข้อมูลการรับเข้าศึกษา: {str(e)}"

# -------------------------------
# Interactive Chat Loop
# -------------------------------
if __name__ == "__main__":
    print("🎓 ระบบถาม-ตอบ BSC Entrance ChatBot พร้อมใช้งาน")
    print("พิมพ์ 'exit' เพื่อออก\n")

    while True:
        query = input("❓ คำถามของคุณ: ").strip()
        if query.lower() in ["exit", "quit", "ออก"]:
            print("👋 บอทปิดการทำงานแล้ว")
            break

        docs = retriever.get_relevant_documents(query)
        merged_context = "\n\n".join([f"ข้อมูล {i+1}:\n{d.page_content}" for i, d in enumerate(docs)])

        print("📝 CONTEXT สำหรับ LLM (Top 3 preview):")
        for i, d in enumerate(docs[:3], start=1):
            print(f"📄 Context {i}: {d.page_content[:200]}...\n")

        if llm is None:
            print("⚠️ ไม่มี API key สำหรับ LLM, ไม่สามารถสร้างคำตอบได้")
            continue

        response = qa_chain.run({"question": query, "context": merged_context})
        print("🤖 คำตอบ:", response)
        print("-"*50)
