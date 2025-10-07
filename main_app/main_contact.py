# main_contact.py - AstraDB Version สำหรับข้อมูลติดต่อ
import os
import uuid
from typing import List

from astrapy import DataAPIClient
from dotenv import load_dotenv
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

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
ASTRA_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE", "default_keyspace")
COLLECTION_NAME = "contactus_embedding"

if not ASTRA_TOKEN or not ASTRA_ENDPOINT:
    print("❌ Missing AstraDB credentials in .env")
    exit(1)

client = DataAPIClient(token=ASTRA_TOKEN)
database = client.get_database_by_api_endpoint(ASTRA_ENDPOINT)

try:
    collection = database.get_collection(COLLECTION_NAME)
    print(f"✅ Connected to collection: {COLLECTION_NAME}")
except Exception as e:
    print(f"❌ Error accessing collection: {e}")
    exit(1)

# -------------------------------
# Custom Retriever
# -------------------------------
class ContactInfoRetriever(BaseRetriever):
    def __init__(self, collection, embedding):
        super().__init__()
        self._collection = collection
        self._embedding = embedding
        self._bm25_retriever = None
        self._documents_cache = None

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        print(f"🔍 Searching contact query: {query}")

        # ดึง keywords
        query_keywords = self._extract_search_keywords(query)
        print(f"📝 Keywords used for search: {query_keywords}")
        
        # ถ้าต้องการข้อมูลทั้งหมด
        if any(word in query.lower() for word in ["ทั้งหมด", "all", "ข้อมูลติดต่อ", "contact"]):
            return self._get_comprehensive_search()

        # Step 1: Text search (BM25)
        text_results = self._text_search(query)

        # Step 2: Vector search
        vector_results = self._vector_search(query)

        # Step 3: Merge results with prioritization
        all_documents = []
        seen_content = set()

        prioritized_docs = []
        regular_docs = []

        for doc in text_results + vector_results:
            if doc.page_content not in seen_content:
                content_lower = doc.page_content.lower()
                has_exact = any(keyword.lower() in content_lower for keyword in query_keywords if len(keyword) >= 2)
                if has_exact:
                    prioritized_docs.append(doc)
                else:
                    regular_docs.append(doc)
                seen_content.add(doc.page_content)

        all_documents = prioritized_docs + regular_docs
        return all_documents[:15]

    def _get_comprehensive_search(self) -> List[Document]:
        print("🚀 Comprehensive search in contact info collection...")
        docs = []
        try:
            results = self._collection.find({}, limit=50)
            for r in results:
                docs.append(Document(page_content=r.get("content", ""), metadata=r.get("metadata", {})))
            print(f"📊 Found {len(docs)} contact documents")
        except Exception as e:
            print(f"❌ Error: {e}")
        return docs

    def _vector_search(self, query: str) -> List[Document]:
        docs = []
        try:
            query_vector = self._embedding.embed_query(query)
            results = self._collection.find({}, sort={"$vector": query_vector}, limit=10)
            for r in results:
                docs.append(Document(page_content=r.get("content", ""), metadata=r.get("metadata", {})))
        except Exception as e:
            print(f"❌ Vector search error: {e}")
        return docs

    def _ensure_bm25_initialized(self):
        if self._bm25_retriever is None:
            try:
                results = self._collection.find({}, limit=100)
                documents = [Document(page_content=r.get("content", ""), metadata=r.get("metadata", {})) for r in results]
                self._documents_cache = documents
                if documents:
                    self._bm25_retriever = BM25Retriever.from_documents(documents)
                    self._bm25_retriever.k = 10
            except Exception as e:
                print(f"❌ BM25 init error: {e}")
                self._bm25_retriever = None

    def _preprocess_query_for_bm25(self, query: str) -> str:
        return query.replace("ขอข้อมูล", "").replace("ติดต่อ", "").strip()

    def _text_search(self, query: str) -> List[Document]:
        try:
            self._ensure_bm25_initialized()
            if self._bm25_retriever is None:
                return self._fallback_keyword_search(query)
            variants = [query, self._preprocess_query_for_bm25(query)]
            seen = set()
            results = []
            for v in variants:
                docs = self._bm25_retriever.get_relevant_documents(v)
                for d in docs:
                    if d.page_content not in seen:
                        results.append(d)
                        seen.add(d.page_content)
            return results[:10]
        except Exception as e:
            print(f"❌ Text search error: {e}")
            return self._fallback_keyword_search(query)

    def _fallback_keyword_search(self, query: str) -> List[Document]:
        results = []
        try:
            keywords = self._extract_search_keywords(query)
            docs = self._collection.find({}, limit=50)
            for r in docs:
                content_lower = r.get("content", "").lower()
                if any(k.lower() in content_lower for k in keywords):
                    results.append(Document(page_content=r.get("content", ""), metadata=r.get("metadata", {})))
        except Exception as e:
            print(f"❌ Fallback search error: {e}")
        return results[:10]

    def _extract_search_keywords(self, query: str) -> List[str]:
        import re
        stop_words = ["ขอ", "ข้อมูล", "ติดต่อ", "ดู", "เกี่ยวกับ", "ใน", "ของ", "และ", "หรือ"]
        clean_query = query
        for sw in stop_words:
            clean_query = clean_query.replace(sw, " ")
        clean_query = re.sub(r'\s+', ' ', clean_query).strip()
        keywords = [clean_query] if clean_query else []
        words = query.split()
        for w in words:
            if w not in stop_words and len(w) >= 2:
                keywords.append(w)
        # Remove duplicates
        unique_keywords = []
        for k in keywords:
            if k not in unique_keywords:
                unique_keywords.append(k)
        return unique_keywords

retriever = ContactInfoRetriever(collection, embedding)

# -------------------------------
# Prompt Template
# -------------------------------
PROMPT = PromptTemplate.from_template("""
บริบทต่อไปนี้คือข้อมูลติดต่อของคณะวิทยาการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น
คุณคือผู้ช่วยที่ให้ข้อมูลติดต่อและรายละเอียดต่างๆ ของคณะวิทยาการคอมพิวเตอร์ มข.
                        
สำคัญ: ตรวจสอบข้อมูลในบริบทอย่างละเอียด หากมีข้อมูลตรงกับคำถามให้นำมาตอบ

หากคำถามเกี่ยวกับข้อมูลติดต่อ ให้แสดงผลแบบรายการที่ชัดเจน ดังนี้:
- ชื่อหน่วยงาน: วิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น
- ที่อยู่: แสดงที่อยู่ครบถ้วน
- เบอร์โทรศัพท์: แสดงเบอร์ติดต่อทั้งหมด
- Hot Line: แสดงเบอร์ Hot Line (ถ้ามี)
- อีเมล: แสดงอีเมลติดต่อ
- บุคลากร: แสดงข้อมูลบุคลากรที่เกี่ยวข้อง (ถ้ามี)

หากคำถามเกี่ยวกับบุคคลใดบุคคลหนึ่ง เช่น คณบดี รองคณบดี ให้แสดงข้อมูล:
- ตำแหน่ง
- ชื่อ-นามสกุล (ถ้ามี)
- เบอร์ติดต่อ (ถ้ามี)
- อีเมล (ถ้ามี)

หากไม่มีข้อมูลที่ตรงกับคำถาม ให้ตอบว่า "ขอโทษ ฉันไม่พบข้อมูลในระบบ"

---------------------
{context}
---------------------
คำถาม: {question}
คำตอบ (จัดรูปแบบให้อ่านง่ายและครบถ้วน):
""")

# -------------------------------
# Chat Model
# -------------------------------
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

if not openrouter_api_key:
    print("⚠️ Warning: OPENROUTER_API_KEY not found, LLM responses will not work")
    llm = None
else:
    llm = ChatOpenAI(
        model="openai/gpt-4o-2024-11-20",
        temperature=0,
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1"
    )

# -------------------------------
# Manual QA Chain
# -------------------------------
def manual_qa_chain(question: str) -> str:
    try:
        print(f"🔍 Searching contact question: {question}")
        docs = retriever.get_relevant_documents(question, run_manager=None)
        if not docs:
            return "ขอโทษ ฉันไม่พบข้อมูลในระบบ"

        # Prepare context
        context_parts = [f"ข้อมูลที่ {i+1}:\n{d.page_content}" for i, d in enumerate(docs)]
        context = "\n".join(context_parts)
        formatted_prompt = PROMPT.format(context=context, question=question)

        if llm is None:
            return "⚠️ ไม่มี API key สำหรับ LLM, ไม่สามารถสร้างคำตอบได้"

        response = llm.invoke(formatted_prompt)
        return response.content.strip()
    except Exception as e:
        print(f"❌ QA chain error: {e}")
        return "ขอโทษ เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่อีกครั้ง"

# -------------------------------
# Interactive Loop
# -------------------------------
if __name__ == "__main__":
    print("📞 ระบบถาม-ตอบข้อมูลติดต่อคณะวิทยาการคอมพิวเตอร์")
    print("พิมพ์ 'exit' เพื่อออก\n")
    while True:
        question = input("❓ ถามข้อมูลติดต่อ: ")
        if question.lower() == "exit":
            break
        answer = manual_qa_chain(question)
        print("🤖 คำตอบ:", answer)
        print("-"*50)
