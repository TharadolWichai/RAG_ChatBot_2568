# main_digital_services.py - Digital Services Chatbot (Enhanced with PyThaiNLP)
# ระบบถาม-ตอบเกี่ยวกับบริการดิจิตอลสำหรับนักศึกษาและบุคลากร

import sys
import os
from typing import List, Any

from astrapy import DataAPIClient
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

# PyThaiNLP for Thai text processing
try:
    from pythainlp import word_tokenize, pos_tag
    from pythainlp.corpus import thai_stopwords
    from pythainlp.util import normalize
    from rank_bm25 import BM25Okapi
    PYTHAINLP_AVAILABLE = True
    print("✅ PyThaiNLP loaded successfully")
except ImportError:
    PYTHAINLP_AVAILABLE = False
    print("⚠️ PyThaiNLP not available - falling back to basic search")

os.environ['PYTHONIOENCODING'] = 'utf-8'

load_dotenv()

# Helper function for safe Thai text output
def safe_print(text):
    """Safely print Thai text to terminal"""
    try:
        print(text)
    except UnicodeEncodeError:
        safe_text = text.encode('ascii', 'ignore').decode('ascii')
        print(f"[Thai text] {safe_text}")
    except Exception as e:
        print(f"[Output error: {e}]")

# -------------------------------
# Embeddings & AstraDB Setup
# -------------------------------
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

ASTRA_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
COLLECTION_NAME = "digital_services_embedding"

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
        model="openai/gpt-4o-mini",  # Use more capable model
        temperature=0.1,  # Lower temperature for more consistent responses
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1"
    )
    print("✅ OpenRouter LLM initialized successfully")

# -------------------------------
# Custom Retriever with BM25 + Vector + PyThaiNLP
# -------------------------------
class DigitalServicesRetriever(BaseRetriever):
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

    def _ensure_bm25_initialized(self):
        """Initialize BM25 retriever lazily"""
        if self.bm25_retriever is None:
            try:
                results = self.collection.find({}, limit=200)
                documents = [Document(page_content=r.get("content", ""), metadata=r.get("metadata", {})) for r in results]
                self.documents_cache = documents
                
                if documents:
                    self.bm25_retriever = BM25Retriever.from_documents(documents)
                    self.bm25_retriever.k = 50
                    print(f"🔧 BM25 retriever initialized with {len(documents)} documents")
            except Exception as e:
                print(f"❌ BM25 init error: {e}")
                self.bm25_retriever = None

    def _extract_keywords(self, query: str) -> List[str]:
        """แยกคำสำคัญจากคำค้นหา (with PyThaiNLP)"""
        if PYTHAINLP_AVAILABLE:
            try:
                # Normalize Thai text
                normalized_query = normalize(query)
                
                # Tokenize
                tokens = word_tokenize(normalized_query, engine="newmm")
                
                # Remove stopwords
                stopwords = thai_stopwords()
                keywords = [w for w in tokens if len(w.strip()) > 1 and w not in stopwords]
                
                safe_print(f"📝 คำสำคัญที่ใช้ค้นหา: {keywords}")
                return keywords
            except Exception as e:
                print(f"⚠️ PyThaiNLP error: {e}")
        
        # Fallback: simple split
        keywords = [w for w in query.split() if len(w) > 2]
        safe_print(f"📝 คำสำคัญที่ใช้ค้นหา: {keywords}")
        return keywords

    def _advanced_thai_search(self, query: str) -> List[Document]:
        """ค้นหาแบบ Advanced สำหรับภาษาไทย"""
        if not PYTHAINLP_AVAILABLE or not self.documents_cache:
            return []
        
        try:
            # Normalize and tokenize query
            normalized_query = normalize(query)
            query_tokens = word_tokenize(normalized_query, engine="newmm")
            stopwords = thai_stopwords()
            query_keywords = [w for w in query_tokens if w not in stopwords and len(w) > 1]
            
            if not query_keywords:
                return []
            
            # Tokenize all documents
            tokenized_docs = []
            for doc in self.documents_cache:
                normalized_content = normalize(doc.page_content)
                tokens = word_tokenize(normalized_content, engine="newmm")
                tokens = [w for w in tokens if w not in stopwords and len(w) > 1]
                tokenized_docs.append(tokens)
            
            # Create BM25 index
            bm25 = BM25Okapi(tokenized_docs)
            
            # Get scores
            scores = bm25.get_scores(query_keywords)
            
            # Create results with scores
            results = []
            for i, score in enumerate(scores):
                if score > 0:
                    doc = self.documents_cache[i]
                    doc.metadata["thai_advanced_score"] = float(score)
                    doc.metadata["search_type"] = "thai_advanced"
                    results.append((score, doc))
            
            # Sort by score
            results.sort(key=lambda x: x[0], reverse=True)
            
            return [doc for score, doc in results[:10]]
            
        except Exception as e:
            print(f"❌ Advanced Thai search error: {e}")
            return []

    def _text_search(self, query: str) -> List[Document]:
        """BM25 text search"""
        self._ensure_bm25_initialized()
        if not self.bm25_retriever:
            return []
        
        try:
            docs = self.bm25_retriever.get_relevant_documents(query)
            print(f"📚 BM25 พบ {len(docs)} เอกสารที่เกี่ยวข้อง")
            
            # Add BM25 scores with proper scoring
            for i, doc in enumerate(docs):
                # Use actual BM25 score if available, otherwise decreasing score
                actual_score = getattr(doc, 'score', None) or (50 - i)
                doc.metadata["bm25_score"] = float(actual_score)
                doc.metadata["search_type"] = "bm25"
            
            return docs
        except Exception as e:
            print(f"❌ BM25 search error: {e}")
            return []

    def _vector_search(self, query: str) -> List[Document]:
        """Vector similarity search"""
        docs = []
        try:
            query_vector = self.embedding.embed_query(query)
            results = self.collection.find({}, sort={"$vector": query_vector}, limit=50)
            
            for i, r in enumerate(results):
                doc = Document(page_content=r.get("content", ""), metadata=r.get("metadata", {}))
                # Use actual vector score if available, otherwise decreasing score
                actual_score = r.get("vector_score", None) or (50 - i)
                doc.metadata["vector_score"] = float(actual_score)
                doc.metadata["search_type"] = "vector"
                docs.append(doc)
            
            print(f"📈 Vector Search พบ {len(docs)} เอกสารที่เกี่ยวข้อง")
        except Exception as e:
            print(f"❌ Vector search error: {e}")
        
        return docs

    def _calculate_hybrid_score(self, doc: Document, bm25_score: float, vector_score: float, query: str) -> float:
        """คำนวณคะแนนรวมจากหลายๆ ปัจจัย"""
        # Normalize scores to 0-1 range
        bm25_norm = min(bm25_score / 50.0, 1.0) if bm25_score > 0 else 0.0
        vector_norm = min(vector_score / 50.0, 1.0) if vector_score > 0 else 0.0
        
        # Base hybrid score (60% BM25/Thai, 40% Vector)
        base_score = (bm25_norm * 0.6) + (vector_norm * 0.4)
        
        # Keyword match bonus
        keywords = self._extract_keywords(query)
        content_lower = doc.page_content.lower()
        keyword_matches = sum(1 for k in keywords if k.lower() in content_lower)
        
        if keyword_matches > 0:
            base_score += (keyword_matches * 0.05)  # 5% bonus per keyword
        
        # Service name exact match bonus
        service_name = doc.metadata.get('service_name', '').lower()
        if service_name and any(k.lower() in service_name for k in keywords):
            base_score += 0.1  # 10% bonus for service name match
        
        # Category relevance bonus
        category = doc.metadata.get('category', '').lower()
        query_lower = query.lower()
        if 'hosting' in query_lower and 'hosting' in category:
            base_score += 0.05
        elif 'software' in query_lower and 'software' in category:
            base_score += 0.05
        elif 'infrastructure' in query_lower and 'infrastructure' in category:
            base_score += 0.05
        
        return min(base_score, 1.0)  # Cap at 1.0

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        safe_print(f"🔍 กำลังค้นหาบริการดิจิตอล: '{query}'")
        
        # ถ้าต้องการข้อมูลทั้งหมด ให้ใช้วิธีพิเศษ
        if any(word in query.lower() for word in ["ทั้งหมด", "ทุกบริการ", "all", "รายการ", "มีบริการอะไรบ้าง", "บริการดิจิตอลทั้งหมด"]):
            safe_print("🎯 ตรวจพบคำขอข้อมูลบริการทั้งหมด - ใช้การค้นหาแบบครอบคลุม")
            return self._get_comprehensive_search()
        
        # Try multiple search strategies
        print("🔍 กำลังค้นหาด้วย AstraDB hybrid search...")
        
        # Strategy 0: Thai Advanced Search (for Thai queries)
        thai_results = []
        is_thai_query = any('\u0e00' <= char <= '\u0e7f' for char in query)  # Check if contains Thai characters
        
        if is_thai_query:
            print("🔍 เริ่ม Advanced Thai Search...")
            thai_results = self._advanced_thai_search(query)
            if thai_results:
                print(f"🎯 Advanced Thai Search: พบ {len(thai_results)} รายการ")

        # Strategy 1: Text search first for exact matches
        print("🔍 เริ่ม Text Search...")
        text_results = self._text_search(query)
        
        # Strategy 2: Vector search
        print("🧠 เริ่ม Vector Search...")
        vector_results = self._vector_search(query)
        
        # Combine results with hybrid scoring and ranking
        all_documents = []
        seen_content = set()
        
        print("🔄 รวมผลลัพธ์พร้อมคำนวณคะแนนรวม...")
        
        # Collect all unique documents with their scores
        candidate_docs = []
        
        # Add Thai advanced search results first (highest priority)
        if thai_results:
            for i, doc in enumerate(thai_results[:5]):  # Top 5 Thai matches
                if doc.page_content not in seen_content:
                    # Get Thai score (could be advanced or exact)
                    thai_score = doc.metadata.get("thai_advanced_score", 0.0)
                    search_type = doc.metadata.get("search_type", "thai_unknown")
                    
                    # Calculate proper combined score using hybrid scoring
                    doc.metadata["combined_score"] = self._calculate_hybrid_score(doc, thai_score, 0.0, query)
                    
                    # Add Thai bonus for very relevant matches
                    if thai_score > 20:  # Very high Thai relevance
                        doc.metadata["combined_score"] += 0.2
                    elif thai_score > 10:  # Medium Thai relevance  
                        doc.metadata["combined_score"] += 0.1
                    
                    doc.metadata["bm25_score"] = thai_score
                    doc.metadata["vector_score"] = 0.0
                    candidate_docs.append(doc)
                    seen_content.add(doc.page_content)
                    
                    search_type_display = "🇹🇭 Advanced"
                    service_name = doc.metadata.get('service_name', 'Unknown')
                    print(f"➕ Thai {search_type_display} #{i+1}: {service_name[:40]}... (Thai: {thai_score:.2f}, Combined: {doc.metadata['combined_score']:.4f})")
        
        # Add text search results
        for i, doc in enumerate(text_results):
            if doc.page_content not in seen_content:
                bm25_score = doc.metadata.get("bm25_score", 0.0)
                doc.metadata["combined_score"] = self._calculate_hybrid_score(doc, bm25_score, 0.0, query)
                candidate_docs.append(doc)
                seen_content.add(doc.page_content)
                service_name = doc.metadata.get('service_name', 'Unknown')
                print(f"➕ Text Search #{i+1}: {service_name[:40]}... (BM25: {bm25_score:.4f})")
        
        # Add vector search results
        for i, doc in enumerate(vector_results):
            if doc.page_content not in seen_content:
                vector_score = doc.metadata.get("vector_score", 0.0)
                doc.metadata["combined_score"] = self._calculate_hybrid_score(doc, 0.0, vector_score, query)
                candidate_docs.append(doc)
                seen_content.add(doc.page_content)
                service_name = doc.metadata.get('service_name', 'Unknown')
                print(f"➕ Vector Search #{i+1}: {service_name[:40]}... (Vector: {vector_score:.4f})")
            else:
                # Update existing document with vector score
                for existing_doc in candidate_docs:
                    if existing_doc.page_content == doc.page_content:
                        vector_score = doc.metadata.get("vector_score", 0.0)
                        bm25_score = existing_doc.metadata.get("bm25_score", 0.0)
                        existing_doc.metadata["vector_score"] = vector_score
                        existing_doc.metadata["combined_score"] = self._calculate_hybrid_score(existing_doc, bm25_score, vector_score, query)
                        service_name = existing_doc.metadata.get('service_name', 'Unknown')
                        print(f"🔄 อัปเดตคะแนน: {service_name[:40]}... (BM25: {bm25_score:.4f}, Vector: {vector_score:.4f})")
                        break
        
        # Sort by combined score (descending)
        candidate_docs.sort(key=lambda doc: doc.metadata.get("combined_score", 0), reverse=True)
        
        # Take top results
        all_documents = candidate_docs
        
        # Show final ranking with scores
        print("\n🏆 ผลลัพธ์สุดท้าย (เรียงตามคะแนนรวม):")
        print("-" * 60)
        for i, doc in enumerate(all_documents[:5], 1):
            combined_score = doc.metadata.get("combined_score", 0.0)
            bm25_score = doc.metadata.get("bm25_score", 0.0)
            vector_score = doc.metadata.get("vector_score", 0.0)
            search_type = doc.metadata.get("search_type", "unknown")
            service_name = doc.metadata.get("service_name", "Unknown")
            
            # Show search type icon
            type_icon = "🇹🇭" if "thai" in search_type else "📝" if "bm25" in search_type else "🧠" if "vector" in search_type else "❓"
            
            print(f"#{i}: {type_icon} {service_name[:50]}...")
            print(f"    🎯 Combined: {combined_score:.4f} | 📝 BM25/Thai: {bm25_score:.4f} | 🧠 Vector: {vector_score:.4f}")
            print(f"    🔍 Search Type: {search_type}")
            print("-" * 40)
        
        print(f"📊 สรุป: Text={len(text_results)}, Vector={len(vector_results)}, รวม={len(all_documents)} (unique)")
        
        return all_documents[:8]  # Return top 8 results (reduce for better LLM processing)
    
    def _get_comprehensive_search(self) -> List[Document]:
        """ค้นหาข้อมูลบริการแบบครอบคลุมทั้งหมดจาก collection"""
        print("🚀 เริ่มการค้นหาบริการดิจิตอลแบบครอบคลุมจาก digital_services_embedding collection...")
        
        all_documents = []
        
        try:
            print("🔍 ค้นหาจาก digital_services_embedding collection...")
            results = self.collection.find({}, limit=50)  # Get all service records
            
            for result in results:
                doc = Document(
                    page_content=result.get("content", ""),
                    metadata=result.get("metadata", {})
                )
                all_documents.append(doc)
            
            print(f"📊 จาก digital_services_embedding: {len(all_documents)} รายการ")
            
        except Exception as e:
            print(f"❌ Error in comprehensive search: {e}")
        
        print(f"🎯 พบข้อมูลบริการครอบคลุมรวม: {len(all_documents)} รายการ")
        return all_documents

retriever = DigitalServicesRetriever(collection, embedding)

# -------------------------------
# Prompt Template
# -------------------------------
PROMPT = PromptTemplate.from_template('''
คุณคือผู้ช่วยตอบคำถามเกี่ยวกับบริการดิจิตอลของวิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น

บริบทข้อมูลบริการดิจิตอล:
{context}

คำถาม: {question}

คำแนะนำในการตอบ:
1. อ่านบริบทข้อมูลอย่างละเอียด
2. หากพบข้อมูลที่เกี่ยวข้องกับคำถาม ให้ตอบทันที
3. ใช้ข้อมูลจากบริบทในการตอบ ไม่ต้องคิดเอง
4. หากถามเกี่ยวกับ Apple Store ให้ตอบตามข้อมูลที่มี
5. หากถามเกี่ยวกับบริการทั้งหมด ให้แสดงรายการบริการที่มี

คำตอบ (ภาษาไทย อ่านง่าย):
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
            return "ขอโทษ ไม่พบข้อมูลบริการดิจิตอลที่ตรงกับคำถามของคุณ"
        
        # Build context from documents
        merged_context = "\n\n".join([f"ข้อมูล {i+1}:\n{d.page_content}" for i, d in enumerate(docs)])
        
        # Check if LLM is available
        if llm is None:
            # Fallback: Return raw context if no LLM
            return f"พบข้อมูลบริการดิจิตอล:\n\n{docs[0].page_content[:500]}..."
        
        # Generate answer using LLM
        response = qa_chain.run({"question": question, "context": merged_context})
        return response
        
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการค้นหาข้อมูลบริการดิจิตอล: {str(e)}"

# -------------------------------
# Interactive Chat Loop
# -------------------------------
if __name__ == "__main__":
    # Fix encoding for Windows terminal (only when running as main script)
    if sys.platform == "win32":
        import codecs
        try:
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
        except:
            pass  # Already detached
    
    print("💻 ระบบถามตอบบริการดิจิตอล (Digital Services ChatBot)")
    print("📱 บริการสำหรับนักศึกษาและบุคลากร วิทยาลัยการคอมพิวเตอร์ มข.")
    print("พิมพ์ 'exit' เพื่อออก\n")
    
    # แสดงตัวอย่างคำถาม
    print("💡 ตัวอย่างคำถาม:")
    print("   - บริการดิจิตอลมีอะไรบ้าง")
    print("   - ChatGPT Plus ใช้ยังไง")
    print("   - ขอลิงก์ Web Hosting")
    print("   - Grammarly คืออะไร")
    print("   - Virtual Machine มีบริการอะไรบ้าง")
    print()

    while True:
        query = input("❓ คำถามของคุณ: ").strip()
        if query.lower() in ["exit", "quit", "ออก"]:
            print("👋 ปิดการทำงานแล้ว")
            break

        docs = retriever.get_relevant_documents(query)
        merged_context = "\n\n".join([f"ข้อมูล {i+1}:\n{d.page_content}" for i, d in enumerate(docs)])

        print("\n🧩 CONTEXT (แสดงตัวอย่าง 3 ชิ้น):")
        for i, d in enumerate(docs[:3], 1):
            service_name = d.metadata.get('service_name', 'Unknown')
            print(f"📄 Context {i} ({service_name}): {d.page_content[:150]}...\n")

        if llm is None:
            print("⚠️ ไม่มี API key ของ OpenRouter, ไม่สามารถสร้างคำตอบได้")
            continue

        try:
            response = qa_chain.run({"question": query, "context": merged_context})
            print("🤖 คำตอบ:", response)
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดจาก LLM: {e}")

        print("-"*60)
