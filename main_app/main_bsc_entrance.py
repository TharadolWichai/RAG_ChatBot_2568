import os
import re
from typing import List, Optional

from astrapy import DataAPIClient
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# -------------------------------
# Config
# -------------------------------
ASTRA_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
COLLECTION_NAME = "bsc_entrance_embedding"

# ✅ ใช้ type เป็นหลัก (ให้เหมือนหน้าอื่น)
DOC_TYPE = "bsc_entrance"

# fallback เผื่อข้อมูลเก่าไม่มี type
SOURCE_URL = "https://computing.kku.ac.th/bsc-entrance"

DEBUG = True  # ปรับเป็น False ได้

if not ASTRA_TOKEN or not ASTRA_ENDPOINT:
    print("❌ Missing AstraDB credentials in .env")
    raise SystemExit(1)

# -------------------------------
# Embeddings
# -------------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large"
)

# -------------------------------
# AstraDB Setup
# -------------------------------
client = DataAPIClient(token=ASTRA_TOKEN)
database = client.get_database_by_api_endpoint(ASTRA_ENDPOINT)
collection = database.get_collection(COLLECTION_NAME)
print(f"✅ Connected to AstraDB Collection: {COLLECTION_NAME}")

# -------------------------------
# Quick sanity check: มี type นี้จริงไหม
# -------------------------------
if DEBUG:
    try:
        sample = list(collection.find({"metadata.type": DOC_TYPE}, limit=1))
        if sample:
            print(f"🔎 DEBUG: Found docs with metadata.type='{DOC_TYPE}' ✅")
        else:
            print(f"⚠️ DEBUG: No docs with metadata.type='{DOC_TYPE}' (will fallback to source='{SOURCE_URL}')")
    except Exception as e:
        print("⚠️ DEBUG: cannot check type:", e)

# -------------------------------
# LLM Setup (OpenAI)
# -------------------------------
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("⚠️ Warning: OPENAI_API_KEY not found, LLM responses will not work")
    llm = None
else:
    llm = ChatOpenAI(
        model="openai/gpt-5.1",
        temperature=0,
        openai_api_key=openai_api_key,
        openai_api_base="https://gen.ai.kku.ac.th/api/v1"
    )
    print("✅ OpenRouter LLM initialized successfully")

# -------------------------------
# Custom Retriever (BM25 + Vector)
# -------------------------------
class BSCEntranceRetriever(BaseRetriever):
    def __init__(self, collection, embedding, doc_type: str, source_url: str, debug: bool = False):
        super().__init__()
        self._collection = collection
        self._embedding = embedding
        self._doc_type = doc_type
        self._source_url = source_url
        self._debug = debug

        self._bm25_retriever = None
        self._documents_cache = None
        self._active_filter = None  # cache filter ที่ใช้จริง

    def _detect_filter(self) -> dict:
        """
        เลือก filter ที่ใช้จริง:
        1) ถ้ามี metadata.type ใช้ type
        2) ถ้าไม่มี ใช้ source fallback
        """
        if self._active_filter is not None:
            return self._active_filter

        try:
            test = list(self._collection.find({"metadata.type": self._doc_type}, limit=1))
            if test:
                self._active_filter = {"metadata.type": self._doc_type}
            else:
                self._active_filter = {"metadata.source": self._source_url}
        except Exception:
            # fallback สุดท้าย
            self._active_filter = {"metadata.source": self._source_url}

        if self._debug:
            print(f"🧩 Active filter = {self._active_filter}")

        return self._active_filter

    def _ensure_bm25_initialized(self):
        if self._bm25_retriever is not None:
            return

        try:
            flt = self._detect_filter()

            # BM25 ไม่ควรโหลดเยอะเกินไป
            results = self._collection.find(flt, limit=1500)
            documents = [Document(page_content=r.get("content", ""), metadata=r.get("metadata", {})) for r in results]
            documents = [d for d in documents if d.page_content and d.page_content.strip()]
            self._documents_cache = documents

            from langchain_community.retrievers import BM25Retriever
            if documents:
                self._bm25_retriever = BM25Retriever.from_documents(documents)
                self._bm25_retriever.k = 15
                if self._debug:
                    print(f"🔧 BM25 initialized with {len(documents)} docs")
            else:
                self._bm25_retriever = None
                if self._debug:
                    print("⚠️ BM25: no documents to index")

        except Exception as e:
            print(f"❌ BM25 init error: {e}")
            self._bm25_retriever = None

    def _normalize_query(self, query: str) -> str:
        q = query.strip()
        q = re.sub(r"\s+", " ", q)
        return q
    def _is_weight_query(self, q: str) -> bool:
        keys = [
            "ค่าน้ำหนัก", "คะแนน", "แต่ละวิชา", "วิชาสอบ",
            "เปอร์เซ็นต์", "%", "101", "102", "103",
            "201", "202", "203", "204"
        ]
        return any(k in q for k in keys)

    def _extract_keywords(self, query: str) -> List[str]:
        stop_words = {"ขอ", "ข้อมูล", "ข่าว", "ประกาศ", "ดู", "เกี่ยวกับ", "ใน", "ของ", "และ", "หรือ", "ให้", "หน่อย"}
        tokens = [t.strip() for t in re.split(r"\s+", query) if t.strip()]
        keywords = [t for t in tokens if t not in stop_words and len(t) >= 2]

        cleaned = " ".join([t for t in tokens if t not in stop_words])
        if cleaned and cleaned not in keywords:
            keywords.insert(0, cleaned)

        uniq = []
        for k in keywords:
            if k not in uniq:
                uniq.append(k)
        return uniq

    def _text_search(self, query: str) -> List[Document]:
        self._ensure_bm25_initialized()
        if not self._bm25_retriever:
            return []
        q = self._normalize_query(query)
        return self._bm25_retriever.get_relevant_documents(q) or []

    def _vector_search(self, query: str) -> List[Document]:
        docs: List[Document] = []
        try:
            q = self._normalize_query(query)
            query_vector = self._embedding.embed_query("query: " + q)

            base_filter = {"metadata.type": "bsc_entrance"}

            # ✅ ถ้าถามเรื่องคะแนนรายวิชา
            if self._is_weight_query(q):
                flt = {**base_filter, "metadata.content_kind": "weight_table"}
                if self._debug:
                    print("🎯 Weight query detected → searching weight_table only")
            else:
                flt = base_filter

            results = self._collection.find(
                flt,
                sort={"$vector": query_vector},
                limit=25,
                include_similarity=True,
            )

            for r in results:
                sim = float(r.get("$similarity", 0.0))
                if sim < 0.15:
                    continue

                content = r.get("content", "")
                if not content or not content.strip():
                    continue

                doc = Document(page_content=content, metadata=r.get("metadata", {}))
                doc.metadata["vector_score"] = sim
                docs.append(doc)

        except Exception as e:
            print(f"❌ Vector search error: {e}")

        return docs

    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        if self._debug:
            print(f"\n🔍 Query: {query}")

        keywords = self._extract_keywords(query)
        text_docs = self._text_search(query)
        vector_docs = self._vector_search(query)

        merged: List[Document] = []
        seen = set()

        for doc in vector_docs + text_docs:
            key = doc.page_content[:200]
            if key in seen:
                continue
            seen.add(key)

            content_lower = doc.page_content.lower()
            has_kw = any(k.lower() in content_lower for k in keywords if len(k) >= 3)
            doc.metadata["priority"] = "high" if has_kw else "normal"
            merged.append(doc)

        merged.sort(
            key=lambda d: (
                0 if d.metadata.get("priority") == "high" else 1,
                -float(d.metadata.get("vector_score", 0.0)),
            )
        )

        if self._debug:
            print(f"✅ Retrieved {len(merged)} docs (top 5)")
            for i, d in enumerate(merged[:5], 1):
                print(
                    f"  {i}) score={d.metadata.get('vector_score', 0):.3f} "
                    f"prio={d.metadata.get('priority')} :: {d.page_content[:120]}..."
                )

        # ส่งให้ LLM ไม่ต้องเยอะ
        return merged[:8]


retriever = BSCEntranceRetriever(collection, embedding, DOC_TYPE, SOURCE_URL, debug=DEBUG)

# -------------------------------
# Prompt & QA Chain
# -------------------------------
PROMPT = PromptTemplate.from_template("""
บริบทต่อไปนี้คือข้อมูลประกาศ/รายละเอียดเกณฑ์การรับเข้าศึกษา (คณะวิทยาลัยการคอมพิวเตอร์ มข.)
คุณต้องตอบโดยอ้างอิงจาก "บริบท" เท่านั้น และห้ามเดาข้อมูลเพิ่มเอง

กติกา:
- ถ้าพบข้อมูลในบริบท ให้ตอบเป็นข้อ ๆ อ่านง่าย และแนบลิงก์ที่ "มีอยู่ในบริบท"
- ถ้าไม่พบข้อมูลที่ตรงคำถาม ให้ตอบว่า "ขอโทษ ฉันไม่พบข้อมูลในระบบ"
ถ้าคำถามมี “คะแนน/ค่าน้ำหนัก/รายวิชา” ต้องตอบเป็นตาราง/บูลเล็ตแยกวิชา

ถ้าคำถามเกี่ยวกับ "ค่าน้ำหนัก/คะแนนรายวิชา/แต่ละวิชา/เปอร์เซ็นต์":
- ต้องแสดงผลเป็นรายการวิชา (101-204) พร้อมเปอร์เซ็นต์
- ถ้าไม่มีข้อมูลรายวิชาในบริบท ให้ตอบว่า "ไม่พบตารางค่าน้ำหนักรายวิชาในระบบ"

---------------------
{context}
---------------------
คำถาม: {question}
คำตอบ:
""")

qa_chain = LLMChain(llm=llm, prompt=PROMPT)

# -------------------------------
# Manual QA Chain Function
# -------------------------------
def manual_qa_chain(question: str) -> str:
    try:
        docs = retriever.get_relevant_documents(question)
        if not docs:
            return "ขอโทษ ไม่พบข้อมูลการรับเข้าศึกษาที่ตรงกับคำถามของคุณ"

        merged_context = "\n\n".join(
            [f"[Doc {i+1}] (score={d.metadata.get('vector_score', 0):.3f})\n{d.page_content}" for i, d in enumerate(docs)]
        )

        if llm is None:
            return f"พบข้อมูล:\n\n{docs[0].page_content[:800]}..."

        return qa_chain.run({"question": question, "context": merged_context})

    except Exception as e:
        return f"เกิดข้อผิดพลาด: {str(e)}"


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

        response = manual_qa_chain(query)
        print("\n🤖 คำตอบ:\n", response)
        print("-" * 60)