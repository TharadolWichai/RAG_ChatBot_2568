import os
from typing import List

from astrapy import DataAPIClient
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from pythainlp import word_tokenize

load_dotenv()

# -------------------------------
# Embeddings & AstraDB Setup
# -------------------------------
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

ASTRA_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
COLLECTION_NAME = "researchgroup_embeddings"

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
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("⚠️ Warning: OPENAI_API_KEY not found, LLM responses will not work")
    llm = None
else:
    llm = ChatOpenAI(
        model="openai/gpt-5.1",
        temperature=0.3,
        openai_api_key=openai_api_key,
        openai_api_base="https://gen.ai.kku.ac.th/api/v1"
    )
    print("✅ OpenAI LLM initialized successfully")

# -------------------------------
# Custom Retriever with BM25 + Vector + Thai Tokenizer
# -------------------------------
class ResearchGroupRetriever(BaseRetriever):
    def __init__(self, collection, embedding):
        super().__init__()
        self._collection = collection
        self._embedding = embedding
        self._bm25_retriever = None
        self._documents_cache = None

    def _ensure_bm25_initialized(self):
        if self._bm25_retriever is None:
            try:
                results = self._collection.find({}, limit=300)
                documents = [Document(page_content=r.get("content", ""), metadata=r.get("metadata", {})) for r in results]
                self._documents_cache = documents
                from langchain_community.retrievers import BM25Retriever
                if documents:
                    self._bm25_retriever = BM25Retriever.from_documents(documents)
                    self._bm25_retriever.k = 50
                    print(f"🔧 BM25 retriever initialized with {len(documents)} documents")
            except Exception as e:
                print(f"❌ BM25 init error: {e}")
                self._bm25_retriever = None

    def _extract_keywords(self, query: str) -> List[str]:
        tokens = [w for w in word_tokenize(query, engine="newmm") if len(w.strip()) > 1]
        keywords = list(dict.fromkeys(tokens))
        print(f"📝 คำสำคัญที่ใช้ค้นหา: {keywords}")
        return keywords

    def _text_search(self, query: str) -> List[Document]:
        self._ensure_bm25_initialized()
        if not self._bm25_retriever:
            return []
        docs = self._bm25_retriever.get_relevant_documents(query)
        print(f"📚 BM25 พบ {len(docs)} เอกสารที่เกี่ยวข้อง")
        return docs

    def _vector_search(self, query: str) -> List[Document]:
        docs = []
        try:
            query_vector = self._embedding.embed_query(query)
            results = self._collection.find({}, sort={"$vector": query_vector}, limit=50)
            for r in results:
                doc = Document(page_content=r.get("content", ""), metadata=r.get("metadata", {}))
                docs.append(doc)
            print(f"📈 Vector Search พบ {len(docs)} เอกสารที่เกี่ยวข้อง")
        except Exception as e:
            print(f"❌ Vector search error: {e}")
        return docs

    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        print(f"🔍 กำลังค้นหาข้อมูลเกี่ยวกับ: '{query}'")
        keywords = self._extract_keywords(query)

        text_docs = self._text_search(query)
        vector_docs = self._vector_search(query)

        merged_docs = []
        seen = set()
        for doc in text_docs + vector_docs:
            if doc.page_content in seen:
                continue
            has_keyword = any(k.lower() in doc.page_content.lower() for k in keywords)
            doc.metadata["priority"] = "สูง" if has_keyword else "ปกติ"
            merged_docs.append(doc)
            seen.add(doc.page_content)

        merged_docs.sort(key=lambda d: 0 if d.metadata["priority"] == "สูง" else 1)
        print(f"🔄 รวมผลลัพธ์ทั้งหมด {len(merged_docs)} ชิ้น (แสดงสูงสุด 10)")

        for i, d in enumerate(merged_docs[:5], 1):
            print(f"🏆 อันดับ {i}: {d.page_content[:100]}... (priority: {d.metadata['priority']})")
                    # -------------------------------
        # 🔍 Filter เฉพาะกรณี "ขอรายชื่อกลุ่มวิจัย"
        # -------------------------------
        if any(keyword in query for keyword in ["รายชื่อ", "กลุ่มวิจัยทั้งหมด", "ชื่อกลุ่มวิจัย"]):
            print("🎯 ตรวจพบว่าคำถามต้องการเฉพาะรายชื่อกลุ่มวิจัย — กำลังกรองข้อมูลเพิ่มเติม...")
            filtered = []
            for d in merged_docs:
                text = d.page_content.lower()
                if any(kw in text for kw in [
                    "กลุ่มวิจัย", "research group", "laboratory", "lab", 
                    "data analytics", "ai", "intelligent", "computing"
                ]):
                    filtered.append(d)
            print(f"✅ กรองเหลือ {len(filtered)} เอกสารที่น่าจะเป็นชื่อกลุ่มวิจัย")
            merged_docs = filtered if filtered else merged_docs

        return merged_docs[:10]

retriever = ResearchGroupRetriever(collection, embedding)

# -------------------------------
# Prompt Template (Manual QA Chain)
# -------------------------------
PROMPT = PromptTemplate.from_template('''
บริบทต่อไปนี้คือข้อมูลของกลุ่มวิจัยจากคณะวิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น
คุณคือผู้ช่วยตอบคำถามเกี่ยวกับกลุ่มวิจัย สมาชิก นักวิจัย ผลงาน ห้องแล็บ หรือข้อมูลติดต่อ

หากถามเกี่ยวกับกลุ่มวิจัยทั้งหมด ให้สรุปกลุ่มวิจัยที่มี พร้อมแนบลิงก์หน้าเว็บ และรายละเอียดสั้นๆ
                    
หากคำถามเกี่ยวข้องกับชื่ออาจารย์หรือกลุ่มวิจัย ให้แสดงผลลัพธ์ในรูปแบบที่เข้าใจง่าย เช่น:
- ชื่อกลุ่มวิจัย: 
    - Hardware-Human Interface and Communications Lab ลิงค์ https://computing.kku.ac.th/hardware-human
    - Machine Learning and Intelligent System ลิงค์ https://computing.kku.ac.th/en/mlislab
    - Advanced Intelligent Interdisciplinary Integration (AIII)  ลิงค์ https://computing.kku.ac.th/aiii
    - Advanced GIS Technology (AGT)  ลิงค์ https://computing.kku.ac.th/agtlab
    - Advanced Smart Computing ลิงค์ https://computing.kku.ac.th/asclab
    - Natural Language and Speech Processing Laboratory  ลิงค์ https://computing.kku.ac.th/nlsplab
    - Applied Intelligence and Data Analytics (AIDA) ลิงค์ https://computing.kku.ac.th/aidalab
    - Intelligent Software Engineering Research Group (I-SERG)  ลิงค์ https://computing.kku.ac.th/i-serg
- หัวข้อการวิจัยหลัก ไม่ใช่ Research rationale
- หัวหน้ากลุ่ม: จากฐานข้อมูลที่มีในหน้าเว็บไซต์
- สมาชิก: จากฐานข้อมูลที่มีในหน้าเว็บไซต์
- รายละเอียด: <สรุปสั้น>
- หน้าเว็บ: <ลิงก์>
- ข้อมูลติดต่อ: <ถ้ามี>

หากไม่พบข้อมูลที่ตรง ให้ตอบว่า "ขอโทษค่ะ ฉันไม่พบข้อมูลของกลุ่มวิจัยนั้นในระบบ"

---------------------
{context}
---------------------
คำถาม: {question}
คำตอบ (ภาษาไทย อ่านง่ายและกระชับ):
''')

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
            return "ขอโทษ ไม่พบข้อมูลกลุ่มวิจัยที่ตรงกับคำถามของคุณ"
        
        # Build context from documents
        merged_context = "\n\n".join([f"ข้อมูล {i+1}:\n{d.page_content}" for i, d in enumerate(docs)])
        
        # Check if LLM is available
        if llm is None:
            # Fallback: Return raw context if no LLM
            return f"พบข้อมูลกลุ่มวิจัย:\n\n{docs[0].page_content[:500]}..."
        
        # Generate answer using LLM
        response = qa_chain.run({"question": question, "context": merged_context})
        return response
        
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการค้นหาข้อมูลกลุ่มวิจัย: {str(e)}"

# -------------------------------
# Interactive Chat Loop
# -------------------------------
if __name__ == "__main__":
    print("🧠 ระบบถามตอบข้อมูลกลุ่มวิจัย (ResearchGroup ChatBot)")
    print("พิมพ์ 'exit' เพื่อออก\n")

    while True:
        query = input("❓ คำถามของคุณ: ").strip()
        if query.lower() in ["exit", "quit", "ออก"]:
            print("👋 ปิดการทำงานแล้ว")
            break

        docs = retriever.get_relevant_documents(query)
        merged_context = "\n\n".join([f"ข้อมูล {i+1}:\n{d.page_content}" for i, d in enumerate(docs)])

        print("\n🧩 CONTEXT (แสดงตัวอย่าง 3 ชิ้น):")
        for i, d in enumerate(docs[:3], 1):
            print(f"📄 Context {i}: {d.page_content[:200]}...\n")

        if llm is None:
            print("⚠️ ไม่มี API key ของ OpenRouter, ไม่สามารถสร้างคำตอบได้")
            continue

        try:
            response = qa_chain.run({"question": query, "context": merged_context})
            print("🤖 คำตอบ:", response)
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดจาก LLM: {e}")

        print("-"*60)