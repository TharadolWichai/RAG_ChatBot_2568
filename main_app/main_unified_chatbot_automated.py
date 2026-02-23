# main_unified_chatbot_automated.py - Unified Chatbot with Automated Data Ingestion Integration
# รวม Hybrid Intent Classification + Dynamic Collections จาก Automated Data Ingestion

import sys
import os
import re
import json
from typing import Dict, List, Tuple, Optional, Any
from dotenv import load_dotenv

# PyThaiNLP for Thai text processing
try:
    from pythainlp import pos_tag, word_tokenize
    from pythainlp.corpus import thai_stopwords
    from pythainlp.util import normalize
    from rank_bm25 import BM25Okapi
    PYTHAINLP_AVAILABLE = True
    print("✅ PyThaiNLP loaded successfully")
except ImportError:
    PYTHAINLP_AVAILABLE = False
    print("⚠️ PyThaiNLP not available - falling back to basic search")

# OpenAI for Intent Classification
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI library not available. LLM fallback will be disabled.")

from astrapy import DataAPIClient
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
# LangChain imports
from langchain.schema import BaseRetriever, Document
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# Import Automated Data Ingestion components
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # main_app -> project root
MAIN_APP_DIR = os.path.join(PROJECT_ROOT, "main_app")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if MAIN_APP_DIR not in sys.path:
    sys.path.insert(0, MAIN_APP_DIR)

# Import Automated Data Ingestion components
try:
    from automated_data_ingestion.core.astradb_manager import AstraDBManager
    AUTOMATED_AVAILABLE = True
except ImportError:
    AUTOMATED_AVAILABLE = False
    print("⚠️ Automated Data Ingestion not available. Please ensure it's installed.")

# Import all chatbot modules (like hybrid version)
try:
    from main_allpeople import manual_qa_chain as allpeople_qa
    from main_allpeople import retriever as allpeople_retriever
    ALLPEOPLE_AVAILABLE = True
except Exception as e:
    print(f"⚠️ AllPeople chatbot not available: {e}")
    ALLPEOPLE_AVAILABLE = False

try:
    from main_contact import manual_qa_chain as contact_qa
    from main_contact import retriever as contact_retriever
    CONTACT_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Contact chatbot not available: {e}")
    CONTACT_AVAILABLE = False

try:
    from main_links import manual_qa_chain as links_qa
    from main_links import retriever as links_retriever
    LINKS_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Links chatbot not available: {e}")
    LINKS_AVAILABLE = False

try:
    from main_scholarship import manual_qa_chain as scholarship_qa
    from main_scholarship import retriever as scholarship_retriever
    SCHOLARSHIP_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Scholarship chatbot not available: {e}")
    SCHOLARSHIP_AVAILABLE = False

try:
    from main_student_club import manual_qa_chain as club_qa
    from main_student_club import retriever as club_retriever
    CLUB_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Student Club chatbot not available: {e}")
    CLUB_AVAILABLE = False

try:
    from main_students import manual_qa_chain as students_qa
    from main_students import retriever as students_retriever
    STUDENTS_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Students chatbot not available: {e}")
    STUDENTS_AVAILABLE = False

try:
    from main_researchgroup import manual_qa_chain as research_qa
    from main_researchgroup import retriever as research_retriever
    RESEARCH_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Research Group chatbot not available: {e}")
    RESEARCH_AVAILABLE = False

try:
    from main_bsc_entrance import manual_qa_chain as bsc_qa
    from main_bsc_entrance import retriever as bsc_retriever
    BSC_AVAILABLE = True
except Exception as e:
    print(f"⚠️ BSC Entrance chatbot not available: {e}")
    BSC_AVAILABLE = False

try:
    from main_digital_services import manual_qa_chain as digital_qa
    from main_digital_services import retriever as digital_retriever
    DIGITAL_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Digital Services chatbot not available: {e}")
    DIGITAL_AVAILABLE = False

try:
    from main_graduate import manual_qa_chain as graduate_qa
    from main_graduate import retriever as graduate_retriever
    GRADUATE_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Graduate Programs chatbot not available: {e}")
    GRADUATE_AVAILABLE = False

load_dotenv()

# Debug: Show which agents are available
print("\n🔍 Agent Availability Status:")
print(f"   AllPeople: {ALLPEOPLE_AVAILABLE}")
print(f"   Contact: {CONTACT_AVAILABLE}")
print(f"   Links: {LINKS_AVAILABLE}")
print(f"   Scholarship: {SCHOLARSHIP_AVAILABLE}")
print(f"   Student Club: {CLUB_AVAILABLE}")
print(f"   Students: {STUDENTS_AVAILABLE}")
print(f"   Research Group: {RESEARCH_AVAILABLE}")
print(f"   BSC Entrance: {BSC_AVAILABLE}")
print(f"   Digital Services: {DIGITAL_AVAILABLE}")
print(f"   Graduate Programs: {GRADUATE_AVAILABLE}")
print()

# ==========================================
# Dynamic Retriever Factory
# ==========================================

class DynamicAstraDBRetriever(BaseRetriever):
    """
    Dynamic Retriever ที่สามารถทำงานกับ collection ใดๆ ใน AstraDB
    รองรับทั้ง Vector Search, Text Search, และ Advanced Thai Search (Hybrid)
    """
    
    def __init__(self, collection, embedding, collection_name: str = ""):
        super().__init__()
        self._collection = collection
        self._embedding = embedding
        self._collection_name = collection_name
        self._bm25_retriever = None
        self._documents_cache = None
        self._thai_bm25 = None
        self._thai_bm25_docs = None
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """ค้นหาข้อมูลจาก AstraDB collection"""
        print(f"🔍 Searching in collection: {self._collection_name}")
        print(f"   Query: {query}")
        
        all_documents = []
        seen_content = set()
        
        # Check if query contains Thai characters
        is_thai_query = any('\u0e00' <= char <= '\u0e7f' for char in query)
        
        # Strategy 0: Advanced Thai Search (if Thai query and PyThaiNLP available)
        if is_thai_query and PYTHAINLP_AVAILABLE:
            print("   🇹🇭 Advanced Thai Search...")
            thai_results = self._advanced_thai_search(query)
            if thai_results:
                for doc in thai_results:
                    if doc.page_content not in seen_content:
                        all_documents.append(doc)
                        seen_content.add(doc.page_content)
                print(f"      ✅ Found {len(thai_results)} Thai search results")
        
        # Strategy 1: Exact Thai Keyword Search (fallback for Thai queries)
        if is_thai_query and not all_documents:
            print("   🎯 Exact Thai Keyword Search...")
            exact_results = self._exact_thai_keyword_search(query)
            if exact_results:
                for doc in exact_results:
                    if doc.page_content not in seen_content:
                        all_documents.append(doc)
                        seen_content.add(doc.page_content)
                print(f"      ✅ Found {len(exact_results)} exact keyword results")
        
        # Strategy 2: Vector Search
        try:
            print("   🧠 Vector Search...")
            query_vector = self._embedding.embed_query(query)
            vector_results = list(self._collection.find(
                {},
                sort={"$vector": query_vector},
                limit=10
            ))
            
            for i, result in enumerate(vector_results):
                doc = Document(
                    page_content=result.get("content", ""),
                    metadata=result.get("metadata", {})
                )
                doc.metadata["vector_score"] = 1.0 - (i * 0.1)
                doc.metadata["search_type"] = "vector"
                if doc.page_content not in seen_content:
                    all_documents.append(doc)
                    seen_content.add(doc.page_content)
            
            print(f"      ✅ Found {len(vector_results)} vector results")
        except Exception as e:
            print(f"      ⚠️ Vector search error: {e}")
        
        # Strategy 3: Text Search (regex)
        try:
            print("   📝 Text Search...")
            text_results = list(self._collection.find(
                {"content": {"$regex": query, "$options": "i"}},
                limit=10
            ))
            
            for result in text_results:
                doc = Document(
                    page_content=result.get("content", ""),
                    metadata=result.get("metadata", {})
                )
                doc.metadata["search_type"] = "text"
                if doc.page_content not in seen_content:
                    all_documents.append(doc)
                    seen_content.add(doc.page_content)
            
            print(f"      ✅ Found {len(text_results)} text results")
        except Exception as e:
            print(f"      ⚠️ Text search error: {e}")
        
        # If no results, try to get some documents anyway
        if not all_documents:
            print("   ⚠️ No search results, trying to get sample documents...")
            try:
                sample_results = list(self._collection.find({}, limit=5))
                for result in sample_results:
                    doc = Document(
                        page_content=result.get("content", ""),
                        metadata=result.get("metadata", {})
                    )
                    all_documents.append(doc)
                print(f"      ✅ Got {len(sample_results)} sample documents")
            except Exception as e:
                print(f"      ❌ Error getting samples: {e}")
        
        print(f"   📊 Total unique documents: {len(all_documents)}")
        return all_documents[:10]  # Return top 10
    
    def _advanced_thai_search(self, query: str) -> List[Document]:
        """Advanced Thai search using PyThaiNLP"""
        if not PYTHAINLP_AVAILABLE:
            return []
        
        try:
            # 1. Normalize and tokenize query
            normalized_query = normalize(query)
            query_tokens = word_tokenize(normalized_query, engine='newmm')
            
            # 2. POS tagging to get meaningful words
            pos_tags = pos_tag(query_tokens, engine='perceptron')
            stopwords = thai_stopwords()
            
            # 3. Extract meaningful tokens (nouns, verbs, adjectives, proper nouns)
            meaningful_tokens = []
            for word, pos in pos_tags:
                if (pos in ['NOUN', 'VERB', 'ADJ', 'PROPN'] and 
                    word not in stopwords and 
                    len(word.strip()) > 1):
                    meaningful_tokens.append(word)
            
            # If no meaningful tokens, use original tokens
            if not meaningful_tokens:
                meaningful_tokens = [token for token in query_tokens if len(token.strip()) > 1]
            
            print(f"      🔤 Tokenized: {query_tokens}")
            print(f"      🎯 Meaningful: {meaningful_tokens}")
            
            # 4. Search in documents
            all_results = list(self._collection.find({}, limit=100))
            matched_docs = []
            
            for result in all_results:
                content = result.get("content", "")
                content_lower = content.lower()
                
                # Check if any meaningful token appears in content
                match_score = 0
                for token in meaningful_tokens:
                    if token in content_lower:
                        match_score += 1
                
                if match_score > 0:
                    doc = Document(
                        page_content=content,
                        metadata=result.get("metadata", {})
                    )
                    doc.metadata["thai_match_score"] = match_score
                    doc.metadata["search_type"] = "thai_advanced"
                    matched_docs.append((match_score, doc))
            
            # Sort by match score (descending)
            matched_docs.sort(key=lambda x: x[0], reverse=True)
            
            # Return top documents
            return [doc for _, doc in matched_docs[:10]]
            
        except Exception as e:
            print(f"      ⚠️ Advanced Thai search error: {e}")
            return []
    
    def _exact_thai_keyword_search(self, query: str) -> List[Document]:
        """Exact keyword search for Thai queries"""
        try:
            # Extract keywords from query
            keywords = self._extract_keywords(query)
            
            if not keywords:
                return []
            
            print(f"      🔑 Keywords: {keywords}")
            
            # Search for documents containing keywords
            all_results = list(self._collection.find({}, limit=100))
            matched_docs = []
            
            for result in all_results:
                content = result.get("content", "").lower()
                
                # Check if any keyword appears in content
                match_count = sum(1 for keyword in keywords if keyword.lower() in content)
                
                if match_count > 0:
                    doc = Document(
                        page_content=result.get("content", ""),
                        metadata=result.get("metadata", {})
                    )
                    doc.metadata["keyword_match_count"] = match_count
                    doc.metadata["search_type"] = "thai_exact"
                    matched_docs.append((match_count, doc))
            
            # Sort by match count (descending)
            matched_docs.sort(key=lambda x: x[0], reverse=True)
            
            return [doc for _, doc in matched_docs[:10]]
            
        except Exception as e:
            print(f"      ⚠️ Exact keyword search error: {e}")
            return []
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query"""
        keywords = []
        
        if PYTHAINLP_AVAILABLE:
            try:
                # Tokenize and get meaningful words
                tokens = word_tokenize(query, engine='newmm')
                stopwords = thai_stopwords()
                
                for token in tokens:
                    if token not in stopwords and len(token.strip()) > 1:
                        keywords.append(token)
            except:
                pass
        
        # Fallback: extract Thai names and important words
        import re
        thai_name_pattern = r'[ก-๙]{3,15}'
        potential_names = re.findall(thai_name_pattern, query)
        for name in potential_names:
            if len(name) >= 3 and name not in keywords:
                keywords.append(name)
        
        # Add original query if no keywords found
        if not keywords:
            keywords = [query.strip()]
        
        return keywords


def create_retriever_from_collection(collection_name: str, astradb_manager: AstraDBManager) -> Optional[DynamicAstraDBRetriever]:
    """
    สร้าง retriever จาก collection name แบบ dynamic
    
    Args:
        collection_name: ชื่อ collection ใน AstraDB
        astradb_manager: AstraDBManager instance
        
    Returns:
        DynamicAstraDBRetriever หรือ None ถ้าไม่พบ collection
    """
    try:
        # Get collection
        collection = astradb_manager.get_or_create_collection(
            collection_name,
            dimension=384,  # all-MiniLM-L6-v2 dimension
            auto_create=False
        )
        
        # Get embedding model
        embedding = astradb_manager.embedding_model
        
        # Create retriever
        retriever = DynamicAstraDBRetriever(
            collection=collection,
            embedding=embedding,
            collection_name=collection_name
        )
        
        print(f"✅ Created retriever for collection: {collection_name}")
        return retriever
        
    except Exception as e:
        print(f"⚠️ Failed to create retriever for {collection_name}: {e}")
        return None


def create_qa_chain_with_logging(retriever: BaseRetriever, collection_name: str) -> callable:
    """
    สร้าง QA chain จาก retriever
    
    Args:
        retriever: BaseRetriever instance
        collection_name: ชื่อ collection (สำหรับ context)
        
    Returns:
        QA function ที่รับ question และ return answer
    """
    # Initialize LLM
    try:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        model_name = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
        
        llm = ChatOpenAI(
            openai_api_key=api_key,
            openai_api_base=base_url,
            model_name=model_name,
            temperature=0.1
        )
    except Exception as e:
        print(f"⚠️ Failed to initialize LLM: {e}")
        return lambda q: f"Error: LLM not available. Collection: {collection_name}"
    
    # Create prompt template (improved version)
    prompt_template = f"""คุณเป็นผู้ช่วยตอบคำถามเกี่ยวกับข้อมูลจาก {collection_name} ของวิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น

**คำแนะนำในการตอบ:**
1. อ่านบริบทข้อมูลอย่างละเอียด
2. หากพบข้อมูลที่เกี่ยวข้องกับคำถาม ให้ตอบทันทีโดยใช้ข้อมูลจากบริบท
3. ใช้ข้อมูลจากบริบทในการตอบ ไม่ต้องคิดเอง
4. หากไม่พบข้อมูลที่เกี่ยวข้อง ให้ตอบว่า "ขอโทษ ไม่พบข้อมูลที่ตรงกับคำถามของคุณในระบบ"
5. ตอบเป็นภาษาไทย อ่านง่ายและกระชับ

**บริบทข้อมูล:**
{{context}}

**คำถาม:** {{question}}

**คำตอบ:**"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create LLM chain
    qa_chain = LLMChain(llm=llm, prompt=PROMPT)
    
    def qa_function(question: str) -> str:
        """QA function wrapper with detailed logging"""
        print(f"\n{'='*60}")
        print(f"🔍 เริ่มค้นหาข้อมูลสำหรับคำถาม: '{question}'")
        print(f"📦 Collection: {collection_name}")
        print(f"{'='*60}\n")
        
        try:
            # Step 1: Retrieve relevant documents
            print("📥 Step 1: กำลังค้นหาเอกสารที่เกี่ยวข้อง...")
            docs = retriever.get_relevant_documents(question)
            
            if not docs:
                print("❌ ไม่พบเอกสารใดๆ")
                return f"ขอโทษ ไม่พบข้อมูลใน collection {collection_name} ที่ตรงกับคำถามของคุณ"
            
            print(f"✅ พบเอกสาร {len(docs)} เอกสาร")
            
            # Step 2: Show retrieved documents
            print(f"\n📄 เอกสารที่ค้นหาได้:")
            for i, doc in enumerate(docs[:5], 1):
                content_preview = doc.page_content[:100].replace('\n', ' ')
                search_type = doc.metadata.get("search_type", "unknown")
                print(f"   {i}. [{search_type}] {content_preview}...")
            
            # Step 3: Filter relevant documents
            print(f"\n🔍 Step 2: กำลังกรองเอกสารที่เกี่ยวข้อง...")
            question_lower = question.lower()
            question_keywords = set(question_lower.split())
            
            relevant_docs = []
            for doc in docs:
                doc_content_lower = doc.page_content.lower()
                # Check if any keyword from question appears in document
                if any(keyword in doc_content_lower for keyword in question_keywords if len(keyword) > 2):
                    relevant_docs.append(doc)
            
            # If no relevant docs found, use all docs anyway (might be semantic match)
            if not relevant_docs:
                print(f"⚠️ ไม่พบ keyword match - ใช้เอกสารทั้งหมด (อาจเป็น semantic match)")
                relevant_docs = docs[:5]
            else:
                print(f"✅ พบเอกสารที่เกี่ยวข้อง {len(relevant_docs)} เอกสาร")
                relevant_docs = relevant_docs[:5]  # Limit to top 5
            
            # Step 4: Build context
            print(f"\n📝 Step 3: กำลังสร้าง context จากเอกสาร...")
            merged_context = "\n\n".join([f"ข้อมูล {i+1}:\n{d.page_content}" for i, d in enumerate(relevant_docs)])
            print(f"✅ Context length: {len(merged_context)} characters")
            print(f"   ใช้เอกสาร: {len(relevant_docs)} เอกสาร")
            
            # Show context preview
            print(f"\n📋 Context Preview (200 chars):")
            print(f"   {merged_context[:200]}...")
            
            # Step 5: Generate answer using LLM
            print(f"\n🤖 Step 4: กำลังเรียก LLM เพื่อสร้างคำตอบ...")
            print(f"   Model: {model_name}")
            print(f"   Question: {question}")
            
            try:
                input_dict = {"question": question, "context": merged_context}
                response = qa_chain.invoke(input_dict)
                
                # Extract text from response if it's a dict
                if isinstance(response, dict):
                    answer = response.get("text", str(response))
                else:
                    answer = str(response)
                
                print(f"✅ LLM สร้างคำตอบสำเร็จ (length: {len(answer)} chars)")
                
            except Exception as chain_error:
                print(f"❌ Chain invoke error: {chain_error}")
                # Fallback: try run method
                try:
                    print(f"🔄 ลองใช้ run method...")
                    answer = qa_chain.run({"question": question, "context": merged_context})
                    print(f"✅ LLM สร้างคำตอบสำเร็จ (fallback method)")
                except Exception as run_error:
                    print(f"❌ Chain run error: {run_error}")
                    return f"เกิดข้อผิดพลาดในการเรียก LLM: {str(run_error)}"
            
            # Step 6: Check answer quality
            print(f"\n📊 Step 5: กำลังตรวจสอบคุณภาพคำตอบ...")
            no_info_phrases = [
                "ไม่มีในข้อมูล", "ไม่พบข้อมูล", "ไม่มีข้อมูล", 
                "no data", "not found", "ไม่มีในบริบท"
            ]
            
            if any(phrase in answer.lower() for phrase in no_info_phrases):
                print(f"⚠️ LLM บอกว่าไม่พบข้อมูล แต่มีเอกสาร {len(docs)} เอกสาร")
                # If LLM says no info, but we have docs, show a sample
                if relevant_docs:
                    sample_content = relevant_docs[0].page_content[:300]
                    answer = f"{answer}\n\n**หมายเหตุ:** พบเอกสารที่เกี่ยวข้อง แต่ข้อมูลอาจไม่ตรงกับคำถามของคุณ\n\nตัวอย่างข้อมูลที่พบ:\n{sample_content}..."
            else:
                print(f"✅ คำตอบดูดี (มีข้อมูล)")
            
            # Add source information
            if docs:
                answer += f"\n\n(พบข้อมูลจาก {len(docs)} เอกสาร)"
            
            print(f"\n{'='*60}")
            print(f"✅ เสร็จสิ้น - คำตอบพร้อมแล้ว")
            print(f"{'='*60}\n")
            
            return answer
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"\n❌ เกิดข้อผิดพลาด: {e}")
            print(f"Traceback:\n{error_details}")
            return f"เกิดข้อผิดพลาด: {str(e)}\n\nDebug: {error_details[:500]}"
    
    return qa_function

# ==========================================
# Hybrid Intent Classification System
# ==========================================

class HybridIntentClassifier:
    """
    ระบบจำแนกประเภทคำถามแบบ Hybrid
    - ลอง Rule-Based ก่อน (เร็ว, ไม่เสียค่าใช้จ่าย)
    - ถ้าความมั่นใจต่ำ → ใช้ LLM ช่วย (แม่นยำ แต่เสียค่าใช้จ่าย)
    """
    
    def __init__(self, collection_mapping: Dict[str, str]):
        """
        Args:
            collection_mapping: Dict mapping intent keys to collection names
                Example: {"allpeople": "allpeople_embedding", "contact": "contact_embedding"}
        """
        self.collection_mapping = collection_mapping
        
        # Rule-based patterns
        self.intent_patterns = {
            "allpeople": {
                "keywords": [
                    "อาจารย์", "ผู้ช่วย", "รอง", "ศาสตราจารย์", "อ.", "ดร.", "หัวหน้า",
                    "บุคลากร", "คณาจารย์", "ผู้สอน", "professor", "lecturer", "faculty",
                    "staff", "teacher", "อาจารย์ประจำ", "สายวิชาการ"
                ],
                "patterns": [
                    r'อาจารย์.*',
                    r'ผู้ช่วยศาสตราจารย์.*',
                    r'รองศาสตราจารย์.*',
                    r'ศาสตราจารย์.*',
                    r'.*หัวหน้า.*',
                    r'.*บุคลากร.*',
                    r'.*คณาจารย์.*'
                ]
            },
            "contact": {
                "keywords": [
                    "ติดต่อ", "โทร", "อีเมล", "email", "เบอร์", "โทรศัพท์", "ที่อยู่",
                    "contact", "address", "phone", "hotline", "แฟกซ์", "fax",
                    "ต่อ", "เบอร์โทร"
                ],
                "patterns": [
                    r'.*ติดต่อ.*',
                    r'.*โทร.*',
                    r'.*อีเมล.*',
                    r'.*เบอร์.*',
                    r'.*ที่อยู่.*',
                    r'\d{3}-\d+',
                    r'.*@.*\..*'
                ]
            },
            "links": {
                "keywords": [
                    "ลิงก์", "ระบบ", "link", "url", "เว็บไซต์", "website", "หน้าเว็บ",
                    "จอง", "booking", "reservation", "ห้องประชุม", "ห้องแล็บ",
                    "แบบฟอร์ม", "form", "ดาวน์โหลด", "download", "อัปโหลด", "upload"
                ],
                "patterns": [
                    r'.*ลิงก์.*',
                    r'.*ระบบ.*',
                    r'.*จอง.*',
                    r'.*แบบฟอร์ม.*',
                    r'https?://.*'
                ]
            },
            "scholarship": {
                "keywords": [
                    "ทุน", "ทุนการศึกษา", "ทุนวิจัย", "scholarship", "grant", "funding",
                    "ทุนนานาชาติ", "ทุน asean", "ทุน gms", "ทุนส่งเสริม",
                    "คุณสมบัติทุน", "เงื่อนไขทุน", "ผลประโยชน์ทุน"
                ],
                "patterns": [
                    r'.*ทุน.*',
                    r'.*scholarship.*',
                    r'.*grant.*',
                    r'.*funding.*'
                ]
            },
            "student_club": {
                "keywords": [
                    "สโมสร", "สโมสรนักศึกษา", "คณะกรรมการ", "ประธาน", "รองประธาน",
                    "student club", "club", "กรรมการ", "เลขานุการ", "เหรัญญิก",
                    "ความเป็นมา", "ประวัติสโมสร"
                ],
                "patterns": [
                    r'.*สโมสร.*',
                    r'.*คณะกรรมการ.*',
                    r'.*ประธาน.*',
                    r'.*student.*club.*'
                ]
            },
            "students": {
                "keywords": [
                    "นักศึกษา", "student", "โครงงาน", "project", "ฝึกงาน", "internship",
                    "สหกิจ", "co-op", "วิทยานิพนธ์", "thesis", "ตารางสอน", "schedule",
                    "ลงทะเบียน", "registration", "เกรด", "grade", "ผลการเรียน"
                ],
                "patterns": [
                    r'.*นักศึกษา.*',
                    r'.*โครงงาน.*',
                    r'.*ฝึกงาน.*',
                    r'.*สหกิจ.*',
                    r'.*วิทยานิพนธ์.*',
                    r'.*ลงทะเบียน.*'
                ]
            },
            "research": {
                "keywords": [
                    "กลุ่มวิจัย", "research group", "lab", "laboratory", "ห้องแล็บ",
                    "นักวิจัย", "researcher", "งานวิจัย", "research", "ผลงานวิจัย",
                    "AIDA", "AIII", "AGT", "ASC", "NLSP", "I-SERG", "MLISLAB"
                ],
                "patterns": [
                    r'.*กลุ่มวิจัย.*',
                    r'.*research.*group.*',
                    r'.*lab.*',
                    r'.*ห้องแล็บ.*',
                    r'.*งานวิจัย.*'
                ]
            },
            "bsc_entrance": {
                "keywords": [
                    "รับเข้า", "สมัคร", "admission", "entrance", "รอบ", "โควตา",
                    "tcas", "portfolio", "เกณฑ์", "คะแนน", "หลักสูตร", "ปริญญาตรี",
                    "undergraduate", "รับสมัคร", "สอบเข้า"
                ],
                "patterns": [
                    r'.*รับเข้า.*',
                    r'.*สมัคร.*',
                    r'.*admission.*',
                    r'.*รอบ.*\d+.*',
                    r'.*tcas.*',
                    r'.*portfolio.*',
                    r'.*โควตา.*'
                ]
            },
            "digital_services": {
                "keywords": [
                    "บริการดิจิตอล", "digital service", "web hosting", "โฮสติ้ง",
                    "virtual machine", "vm", "เครื่องเสมือน", "apple store", "google play",
                    "grammarly", "chatgpt plus", "ai server", "snapdrop", "แชร์ไฟล์",
                    "บริการ", "ส่วนที่", "ข้อตกลง", "เทคโนโลยี"
                ],
                "patterns": [
                    r'.*บริการ.*ดิจิตอล.*',
                    r'.*digital.*service.*',
                    r'.*web.*hosting.*',
                    r'.*virtual.*machine.*',
                    r'.*apple.*store.*',
                    r'.*google.*play.*',
                    r'.*grammarly.*',
                    r'.*chatgpt.*plus.*',
                    r'.*snapdrop.*',
                    r'.*ส่วนที่.*\d+.*'
                ]
            },
            "graduate": {
                "keywords": [
                    "บัณฑิตศึกษา", "graduate", "ปริญญาโท", "master", "มหาบัณฑิต", "ป.โท",
                    "ปริญญาเอก", "phd", "ph.d", "ดุษฎีบัณฑิต", "ป.เอก", "doctoral",
                    "หลักสูตรโท", "หลักสูตรเอก", "สมัครโท", "สมัครเอก", "คุณสมบัติโท",
                    "คุณสมบัติเอก", "ค่าเทอมโท", "ค่าเทอมเอก", "อาจารย์ที่ปรึกษา"
                ],
                "patterns": [
                    r'.*บัณฑิตศึกษา.*',
                    r'.*ปริญญาโท.*',
                    r'.*ป\.โท.*',
                    r'.*master.*',
                    r'.*ปริญญาเอก.*',
                    r'.*ป\.เอก.*',
                    r'.*phd.*',
                    r'.*ph\.d.*',
                    r'.*doctoral.*',
                    r'.*หลักสูตร.*โท.*',
                    r'.*หลักสูตร.*เอก.*'
                ]
            }
        }
        
        # LLM descriptions (for fallback)
        self.llm_intent_descriptions = {
            "allpeople": {
                "name": "อาจารย์และบุคลากร",
                "description": "ข้อมูลเกี่ยวกับอาจารย์, ผู้ช่วยศาสตราจารย์, รองศาสตราจารย์, ศาสตราจารย์, บุคลากร, คณาจารย์, หัวหน้าภาควิชา",
                "examples": ["อาจารย์สมชาย", "ผศ.ดร.สมหญิง", "หัวหน้าภาควิชา"]
            },
            "contact": {
                "name": "ข้อมูลติดต่อ",
                "description": "ข้อมูลติดต่อหน่วยงาน, เบอร์โทรศัพท์, อีเมล, ที่อยู่, แฟกซ์, Hot Line",
                "examples": ["ติดต่อวิทยาลัย", "เบอร์โทรศัพท์", "อีเมล"]
            },
            "links": {
                "name": "ลิงก์และระบบ",
                "description": "ลิงก์ระบบต่างๆ, การจองห้องประชุม, จองห้องแล็บ, แบบฟอร์ม, ระบบจัดการเอกสาร",
                "examples": ["ลิงก์จองห้องประชุม", "แบบฟอร์มลาพักผ่อน", "ดาวน์โหลดแบบฟอร์ม"]
            },
            "scholarship": {
                "name": "ทุนการศึกษา",
                "description": "ทุนการศึกษา, ทุนวิจัย, ทุนนานาชาติ, ทุน ASEAN, ทุน GMS, คุณสมบัติทุน",
                "examples": ["ทุนการศึกษา", "ทุนวิจัย", "ทุนนานาชาติ"]
            },
            "student_club": {
                "name": "สโมสรนักศึกษา",
                "description": "สโมสรนักศึกษา, คณะกรรมการสโมสร, ประธานสโมสร, กิจกรรมสโมสร",
                "examples": ["ประธานสโมสร", "คณะกรรมการสโมสร", "กิจกรรมสโมสร"]
            },
            "students": {
                "name": "ลิงก์บริการนักศึกษา",
                "description": "บริการสำหรับนักศึกษา, ลิงก์โครงงาน, วิทยานิพนธ์, ลงทะเบียน, ตารางสอน",
                "examples": ["ลิงก์โครงงานนักศึกษา", "ลิงก์ลงทะเบียน", "ตารางสอน"]
            },
            "research": {
                "name": "กลุ่มวิจัย",
                "description": "ข้อมูลกลุ่มวิจัย, ห้องแล็บ, นักวิจัย, AIDA Lab, AIII Lab, AGT Lab",
                "examples": ["กลุ่มวิจัย AIDA", "ห้องแล็บ AI", "รายชื่อกลุ่มวิจัย"]
            },
            "bsc_entrance": {
                "name": "การรับเข้าศึกษา",
                "description": "การรับเข้าศึกษาระดับปริญญาตรี, รอบ Portfolio, TCAS, โควตา, เกณฑ์คะแนน",
                "examples": ["รอบ Portfolio", "เกณฑ์รับเข้า", "TCAS รอบ 3"]
            },
            "digital_services": {
                "name": "บริการดิจิตอล",
                "description": "บริการดิจิตอล, Web Hosting, Virtual Machine, Apple Store, Google Play, Grammarly, ChatGPT Plus",
                "examples": ["Web Hosting", "Virtual Machine", "Apple Store", "Grammarly"]
            },
            "graduate": {
                "name": "หลักสูตรบัณฑิตศึกษา",
                "description": "ข้อมูลหลักสูตรบัณฑิตศึกษา, ปริญญาโท (Master), ปริญญาเอก (Ph.D.), คุณสมบัติผู้สมัคร, ค่าใช้จ่าย, อาจารย์ที่ปรึกษา",
                "examples": ["ปริญญาโท", "ปริญญาเอก", "หลักสูตรโท"]
            }
        }
        
        # Initialize LLM if available
        self.llm_client = None
        if OPENAI_AVAILABLE:
            try:
                api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
                if api_key:
                    base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
                    self.llm_client = OpenAI(api_key=api_key, base_url=base_url)
                    self.llm_model = "openai/gpt-4o-mini"
                    print("✅ LLM fallback initialized successfully!")
            except Exception as e:
                print(f"⚠️ LLM fallback initialization failed: {e}")
    
    def classify(self, query: str) -> Tuple[str, float, str, str]:
        """
        จำแนกประเภทคำถามแบบ Hybrid
        Returns: (intent_name, confidence_score, method_used, reason)
        """
        
        # Step 1: Try Rule-Based first
        rule_intent, rule_confidence = self._rule_based_classify(query)
        
        HIGH_CONFIDENCE_THRESHOLD = 7.0
        
        if rule_confidence >= HIGH_CONFIDENCE_THRESHOLD:
            print(f"   ✅ Rule-Based มั่นใจสูง (คะแนน: {rule_confidence:.2f})")
            return rule_intent, rule_confidence, "rule_based", "High confidence from keyword/pattern matching"
        
        # Step 2: Low confidence - Use LLM fallback if available
        if self.llm_client and rule_confidence < HIGH_CONFIDENCE_THRESHOLD:
            print(f"   ⚠️ Rule-Based มั่นใจต่ำ (คะแนน: {rule_confidence:.2f}) → ใช้ LLM ช่วย")
            llm_intent, llm_confidence, llm_reason = self._llm_classify(query)
            
            if llm_intent != "unknown" and llm_confidence >= 0.4:
                print(f"   ✅ LLM ให้คำแนะนำ: {llm_intent} (มั่นใจ: {llm_confidence:.2f})")
                return llm_intent, llm_confidence, "llm_fallback", llm_reason
            elif llm_intent != "unknown":
                print(f"   🔄 LLM มั่นใจต่ำแต่ยังใช้ได้: {llm_intent} (มั่นใจ: {llm_confidence:.2f})")
                return llm_intent, max(llm_confidence, 0.3), "llm_low_conf", llm_reason
            else:
                print(f"   🔄 LLM ไม่แน่ใจ → ใช้ Rule-Based แทน: {rule_intent} (คะแนน: {rule_confidence:.2f})")
                return rule_intent, rule_confidence, "rule_fallback", "LLM uncertain, using rule-based result"
        
        # Step 3: No LLM available or rule-based result is ok
        if rule_intent != "unknown":
            return rule_intent, rule_confidence, "rule_based", "LLM not available, using rule-based"
        
        return "unknown", 0.0, "rule_based", "No match found"
    
    def _rule_based_classify(self, query: str) -> Tuple[str, float]:
        """Rule-based classification"""
        query_lower = query.lower()
        
        # Tokenize with PyThaiNLP if available
        if PYTHAINLP_AVAILABLE:
            tokens = word_tokenize(query_lower, engine='newmm')
        else:
            tokens = query_lower.split()
        
        # Calculate scores for each intent
        intent_scores = {}
        
        for intent, config in self.intent_patterns.items():
            score = 0.0
            matched_keywords = 0
            
            # 1. Keyword matching
            for keyword in config["keywords"]:
                if keyword.lower() in query_lower:
                    matched_keywords += 1
                    score += 3.0
                elif any(keyword.lower() in token.lower() for token in tokens):
                    matched_keywords += 1
                    score += 2.0
            
            # 2. Pattern matching
            pattern_matches = 0
            for pattern in config["patterns"]:
                if re.search(pattern, query_lower):
                    pattern_matches += 1
                    score += 2.0
            
            if matched_keywords > 0 or pattern_matches > 0:
                # Bonus for multiple matches
                if matched_keywords >= 2:
                    score *= 1.5
                intent_scores[intent] = score
            else:
                intent_scores[intent] = 0.0
        
        # Get best match
        if not intent_scores:
            return "unknown", 0.0
        
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        
        if best_intent[1] < 5.0:
            return "unknown", best_intent[1]
        
        return best_intent
    
    def _llm_classify(self, query: str) -> Tuple[str, float, str]:
        """LLM-based classification (fallback)"""
        if not self.llm_client:
            return "unknown", 0.0, "LLM not available"
        
        try:
            prompt = self._build_llm_prompt(query)
            
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an intent classifier. Respond ONLY with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=150,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content.strip()
            result = json.loads(result_text)
            
            intent = result.get("intent", "unknown")
            confidence = float(result.get("confidence", 0.0))
            reason = result.get("reason", "No reason provided")
            
            # Validate intent
            if intent not in self.llm_intent_descriptions and intent != "unknown":
                intent = "unknown"
                confidence = 0.0
            
            return intent, confidence, reason
            
        except Exception as e:
            print(f"   ❌ LLM Error: {e}")
            return "unknown", 0.0, f"Error: {str(e)}"
    
    def _build_llm_prompt(self, query: str) -> str:
        """Build prompt for LLM"""
        intent_list = []
        for intent_key, intent_info in self.llm_intent_descriptions.items():
            intent_list.append(
                f"- **{intent_key}** ({intent_info['name']}): {intent_info['description']}"
            )
        
        intents_text = "\n".join(intent_list)
        
        prompt = f"""คุณเป็น Intent Classifier สำหรับระบบ Chatbot ของวิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น

มี categories ดังนี้:

{intents_text}

คำถามจากผู้ใช้: "{query}"

วิเคราะห์และตอบเป็น JSON:
{{
  "intent": "intent_key หรือ unknown",
  "confidence": 0.0-1.0,
  "reason": "เหตุผลสั้นๆ"
}}

**สำคัญ:** ตอบเป็น JSON เท่านั้น"""
        
        return prompt

# ==========================================
# Unified Chatbot with Automated Integration
# ==========================================

class UnifiedChatbotAutomated:
    """
    Unified Chatbot ที่ทำงานร่วมกับ Automated Data Ingestion
    - ดึง collections แบบ dynamic จาก AstraDB
    - สร้าง retrievers และ QA chains แบบ dynamic
    - ใช้ Hybrid Intent Classification
    """
    
    def __init__(self, collection_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize Unified Chatbot
        
        Args:
            collection_mapping: Optional dict mapping intent keys to collection names
                If None, will auto-detect from available collections
                Example: {"allpeople": "allpeople_embedding", "contact": "contact_embedding"}
        """
        # Initialize AstraDB Manager
        if not AUTOMATED_AVAILABLE:
            raise Exception("Automated Data Ingestion not available. Please install it.")
        
        self.astradb_manager = AstraDBManager()
        
        # Get available collections
        print("\n🔍 Discovering available collections...")
        available_collections = self.astradb_manager.list_collections()
        print(f"   Found {len(available_collections)} collections: {available_collections}")
        
        # Build collection mapping
        if collection_mapping is None:
            collection_mapping = self._build_default_collection_mapping(available_collections)
        
        self.collection_mapping = collection_mapping
        print(f"\n📋 Collection Mapping:")
        for intent, collection in collection_mapping.items():
            print(f"   {intent} → {collection}")
        
        # Initialize hybrid classifier
        self.classifier = HybridIntentClassifier(collection_mapping)
        
        # Map intents to chatbot configs (will be populated dynamically)
        self.chatbot_map = {}
        
        # Build chatbot map from collections
        self._build_chatbot_map()
        
        print(f"\n✅ Unified Chatbot (Automated) initialized with {len(self.chatbot_map)} agents")
        for intent, config in self.chatbot_map.items():
            print(f"   {config['icon']} {config['name']} ({config['collection']})")
    
    def _build_default_collection_mapping(self, available_collections: List[str]) -> Dict[str, str]:
        """Build default collection mapping from available collections"""
        # Default mapping patterns
        default_patterns = {
            "allpeople": ["allpeople", "faculty", "staff"],
            "contact": ["contact", "services"],
            "links": ["links", "services"],
            "scholarship": ["scholarship"],
            "student_club": ["student_club", "club"],
            "students": ["students", "student"],
            "research": ["research", "researchgroup"],
            "bsc_entrance": ["bsc", "entrance", "admission"],
            "digital_services": ["digital", "services"],
            "graduate": ["graduate"]
        }
        
        mapping = {}
        
        for intent, patterns in default_patterns.items():
            for collection in available_collections:
                collection_lower = collection.lower()
                # Check if collection name matches any pattern
                if any(pattern in collection_lower for pattern in patterns):
                    # Prefer collections with "_embedding" suffix
                    if "_embedding" in collection_lower:
                        mapping[intent] = collection
                        break
        
        return mapping
    
    def _build_chatbot_map(self):
        """Build chatbot map - ใช้ retriever จากแต่ละ module เหมือน hybrid version"""
        # Intent metadata
        intent_metadata = {
            "allpeople": {"name": "อาจารย์และบุคลากร", "icon": "👨‍🏫", "module_retriever": allpeople_retriever if ALLPEOPLE_AVAILABLE else None, "module_qa": allpeople_qa if ALLPEOPLE_AVAILABLE else None},
            "contact": {"name": "ข้อมูลติดต่อ", "icon": "📞", "module_retriever": contact_retriever if CONTACT_AVAILABLE else None, "module_qa": contact_qa if CONTACT_AVAILABLE else None},
            "links": {"name": "ลิงก์และระบบ", "icon": "🔗", "module_retriever": links_retriever if LINKS_AVAILABLE else None, "module_qa": links_qa if LINKS_AVAILABLE else None},
            "scholarship": {"name": "ทุนการศึกษา", "icon": "🎓", "module_retriever": scholarship_retriever if SCHOLARSHIP_AVAILABLE else None, "module_qa": scholarship_qa if SCHOLARSHIP_AVAILABLE else None},
            "student_club": {"name": "สโมสรนักศึกษา", "icon": "🎭", "module_retriever": club_retriever if CLUB_AVAILABLE else None, "module_qa": club_qa if CLUB_AVAILABLE else None},
            "students": {"name": "ลิงก์นักศึกษา", "icon": "📚", "module_retriever": students_retriever if STUDENTS_AVAILABLE else None, "module_qa": students_qa if STUDENTS_AVAILABLE else None},
            "research": {"name": "กลุ่มวิจัย", "icon": "🔬", "module_retriever": research_retriever if RESEARCH_AVAILABLE else None, "module_qa": research_qa if RESEARCH_AVAILABLE else None},
            "bsc_entrance": {"name": "การรับเข้าศึกษา", "icon": "🎓", "module_retriever": bsc_retriever if BSC_AVAILABLE else None, "module_qa": bsc_qa if BSC_AVAILABLE else None},
            "digital_services": {"name": "บริการดิจิตอล", "icon": "💻", "module_retriever": digital_retriever if DIGITAL_AVAILABLE else None, "module_qa": digital_qa if DIGITAL_AVAILABLE else None},
            "graduate": {"name": "หลักสูตรบัณฑิตศึกษา", "icon": "🎓", "module_retriever": graduate_retriever if GRADUATE_AVAILABLE else None, "module_qa": graduate_qa if GRADUATE_AVAILABLE else None}
        }
        
        for intent, collection_name in self.collection_mapping.items():
            try:
                metadata = intent_metadata.get(intent, {"name": intent, "icon": "📦"})
                
                # ใช้ retriever จาก module ถ้ามี (เหมือน hybrid version)
                if metadata.get("module_retriever") and metadata.get("module_qa"):
                    print(f"✅ Using module retriever for {intent}")
                    self.chatbot_map[intent] = {
                        "name": metadata["name"],
                        "icon": metadata["icon"],
                        "collection": collection_name,
                        "qa_function": metadata["module_qa"],
                        "retriever": metadata["module_retriever"]
                    }
                else:
                    # Fallback: สร้าง dynamic retriever
                    print(f"⚠️ Module retriever not available for {intent}, using dynamic retriever")
                    retriever = create_retriever_from_collection(collection_name, self.astradb_manager)
                    if not retriever:
                        print(f"⚠️ Skipping {intent}: Failed to create retriever")
                        continue
                    
                    # Create QA chain with detailed logging
                    qa_function = create_qa_chain_with_logging(retriever, collection_name)
                    
                    # Add to chatbot map
                    self.chatbot_map[intent] = {
                        "name": metadata["name"],
                        "icon": metadata["icon"],
                        "collection": collection_name,
                        "qa_function": qa_function,
                        "retriever": retriever
                    }
                
            except Exception as e:
                print(f"⚠️ Failed to setup {intent} ({collection_name}): {e}")
                import traceback
                traceback.print_exc()
    
    def answer(self, question: str) -> str:
        """ตอบคำถามโดยใช้ Hybrid Classification (with detailed logging)"""
        
        # Step 1: Hybrid Intent Classification
        print(f"\n{'='*60}")
        print(f"🔀 กำลังวิเคราะห์คำถามด้วย Hybrid Classification...")
        print(f"   คำถาม: '{question}'")
        print(f"{'='*60}")
        
        intent, confidence, method, reason = self.classifier.classify(question)
        
        print(f"\n🎯 Hybrid Intent Classification Result:")
        print(f"   ประเภท: {intent}")
        print(f"   ความมั่นใจ: {confidence:.2f}")
        print(f"   วิธีการ: {method}")
        print(f"   เหตุผล: {reason}")
        
        # Step 2: Route to appropriate chatbot
        if intent == "unknown":
            print(f"\n❓ ไม่แน่ใจประเภทคำถาม - จะค้นหาจากทุก Agent")
            return self._multi_agent_search(question)
        
        if intent not in self.chatbot_map:
            print(f"\n⚠️ Agent '{intent}' ไม่พร้อมใช้งาน - จะค้นหาจากทุก Agent")
            return self._multi_agent_search(question)
        
        # Step 3: Use specific chatbot
        chatbot_config = self.chatbot_map[intent]
        print(f"\n{'='*60}")
        print(f"➡️  เลือก Agent: {chatbot_config['icon']} {chatbot_config['name']}")
        print(f"📦 Collection: {chatbot_config['collection']}")
        print(f"{'='*60}\n")
        
        try:
            # Call QA function (which has detailed logging inside)
            answer = chatbot_config["qa_function"](question)
            return f"{chatbot_config['icon']} [{chatbot_config['name']}]\n\n{answer}"
        except Exception as e:
            print(f"\n❌ Error from {chatbot_config['name']}: {e}")
            import traceback
            traceback.print_exc()
            return f"ขอโทษ เกิดข้อผิดพลาดจาก Agent {chatbot_config['name']}"
    
    def _multi_agent_search(self, question: str) -> str:
        """ค้นหาจากทุก Agent และรวมผลลัพธ์"""
        print("🔍 กำลังค้นหาจากทุก Agent...\n")
        print(f"📊 Total agents to search: {len(self.chatbot_map)}")
        
        results = []
        
        for i, (intent, config) in enumerate(self.chatbot_map.items(), 1):
            print(f"\n{'='*60}")
            print(f"🔸 Agent {i}/{len(self.chatbot_map)}: {config['icon']} {config['name']} ({config['collection']})")
            print(f"{'='*60}")
            try:
                answer = config["qa_function"](question)
                
                # Check if answer is meaningful
                is_meaningful = len(answer.strip()) > 50 and not any(
                    phrase in answer.lower() for phrase in [
                        "ไม่พบข้อมูล", "ไม่มีข้อมูล", "no data", "not found",
                        "ขอโทษ", "sorry", "ไม่สามารถ", "error"
                    ]
                )
                
                if answer and is_meaningful:
                    results.append({
                        "agent": config["name"],
                        "icon": config["icon"],
                        "answer": answer
                    })
                    print(f"   ✅ พบข้อมูลที่มีความหมาย!")
                else:
                    print(f"   ⚪ ไม่พบข้อมูลที่มีความหมาย")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Combine results
        if not results:
            return "ขอโทษ ไม่พบข้อมูลที่ตรงกับคำถามของคุณในระบบ"
        
        if len(results) == 1:
            result = results[0]
            return f"{result['icon']} [{result['agent']}]\n\n{result['answer']}"
        
        # Multiple results
        combined = "พบข้อมูลจากหลาย Agent:\n\n"
        for i, result in enumerate(results, 1):
            combined += f"{result['icon']} **{result['agent']}**\n"
            combined += f"{result['answer']}\n\n"
            if i < len(results):
                combined += f"{'-'*60}\n\n"
        
        return combined
    
    def answer_with_contexts(self, question: str) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        ตอบคำถามพร้อม contexts และ classification info
        สำหรับการประเมินผลด้วย RAGAS
        
        Args:
            question: คำถามจากผู้ใช้
            
        Returns:
            tuple: (answer, contexts, classification_info)
            - answer: คำตอบจากโมเดล (str)
            - contexts: รายการ contexts ที่ retrieve มา (List[str])
            - classification_info: ข้อมูล classification (Dict)
        """
        print(f"\n{'='*60}")
        print(f"🔀 [WITH CONTEXTS] กำลังวิเคราะห์คำถามด้วย Hybrid Classification...")
        print(f"   คำถาม: '{question}'")
        print(f"{'='*60}")
        
        # Step 1: Hybrid Intent Classification
        intent, confidence, method, reason = self.classifier.classify(question)
        
        classification_info = {
            "intent": intent,
            "confidence": float(confidence),
            "method": method,
            "reason": reason
        }
        
        print(f"\n🎯 Hybrid Intent Classification Result:")
        print(f"   ประเภท: {intent}")
        print(f"   ความมั่นใจ: {confidence:.2f}")
        print(f"   วิธีการ: {method}")
        print(f"   เหตุผล: {reason}")
        
        # Step 2: Get contexts and answer based on intent
        contexts = []
        
        if intent == "unknown" or intent not in self.chatbot_map:
            print(f"\n❓ ไม่แน่ใจประเภทคำถาม - จะค้นหาจากทุก Agent และรวม contexts")
            
            # Multi-agent search - collect contexts from all agents
            for intent_key, config in self.chatbot_map.items():
                try:
                    if "retriever" in config and config["retriever"]:
                        print(f"   📚 ดึง contexts จาก {config['name']}...")
                        docs = config["retriever"].get_relevant_documents(question)
                        agent_contexts = [doc.page_content for doc in docs]
                        contexts.extend(agent_contexts)
                        print(f"      ✅ ได้ {len(agent_contexts)} contexts")
                except Exception as e:
                    print(f"      ⚠️ Error: {e}")
            
            # Get answer
            answer = self._multi_agent_search(question)
            
            # Remove duplicate contexts while preserving order
            seen = set()
            unique_contexts = []
            for ctx in contexts:
                if ctx not in seen:
                    unique_contexts.append(ctx)
                    seen.add(ctx)
            contexts = unique_contexts
            
            print(f"\n📊 รวม contexts จากทุก agent: {len(contexts)} contexts (unique)")
            
        else:
            # Specific agent - get contexts from that agent
            chatbot_config = self.chatbot_map[intent]
            print(f"\n{'='*60}")
            print(f"➡️  เลือก Agent: {chatbot_config['icon']} {chatbot_config['name']}")
            print(f"📦 Collection: {chatbot_config['collection']}")
            print(f"{'='*60}\n")
            
            try:
                # Get contexts if retriever available
                if "retriever" in chatbot_config and chatbot_config["retriever"]:
                    print(f"📚 กำลังดึง contexts จาก {chatbot_config['name']}...")
                    docs = chatbot_config["retriever"].get_relevant_documents(question)
                    contexts = [doc.page_content for doc in docs]
                    print(f"   ✅ ได้ {len(contexts)} contexts")
                else:
                    print(f"   ⚠️ Agent นี้ไม่มี retriever")
                
                # Get answer
                answer = chatbot_config["qa_function"](question)
                answer = f"{chatbot_config['icon']} [{chatbot_config['name']}]\n\n{answer}"
                
            except Exception as e:
                print(f"\n❌ Error from {chatbot_config['name']}: {e}")
                import traceback
                traceback.print_exc()
                answer = f"ขอโทษ เกิดข้อผิดพลาดจาก Agent {chatbot_config['name']}: {str(e)}"
                contexts = []
        
        # Ensure we have at least some context
        if not contexts:
            contexts = [f"No contexts retrieved for question: {question}"]
            print(f"   ⚠️ ไม่มี contexts - ใช้ fallback message")
        
        print(f"\n✅ เสร็จสิ้น:")
        print(f"   - Answer length: {len(answer)} chars")
        print(f"   - Total contexts: {len(contexts)}")
        print(f"   - Classification: {intent} ({confidence:.2f})")
        
        return answer, contexts, classification_info
    
    def show_help(self):
        """แสดงคำแนะนำการใช้งาน"""
        print("\n" + "="*60)
        print("📖 คำแนะนำการใช้งาน Unified Chatbot (Automated Version)")
        print("="*60)
        print("\n🔀 ระบบใช้ Hybrid Classification (Rule-Based + LLM)!")
        print("   - ลอง Rule-Based ก่อน (เร็ว, ฟรี)")
        print("   - ถ้าไม่มั่นใจ → ใช้ LLM ช่วย (แม่นยำ)")
        print("\n📦 Collections ที่พร้อมใช้งาน:\n")
        
        for intent, config in self.chatbot_map.items():
            print(f"{config['icon']} {config['name']} ({config['collection']})")
        
        print("\n💡 ตัวอย่างคำถาม:")
        print("   - อาจารย์สมชาย → อาจารย์และบุคลากร")
        print("   - ติดต่อวิทยาลัย → ข้อมูลติดต่อ")
        print("   - ลิงก์จองห้องประชุม → ลิงก์และระบบ")
        print("   - ทุนการศึกษา → ทุนการศึกษา")
        print("="*60)

# ==========================================
# Main Program
# ==========================================

def main():
    """Main function สำหรับ Unified Chatbot (Automated Version)"""
    
    # Fix encoding for Windows terminal
    if sys.platform == "win32":
        import codecs
        try:
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
        except:
            pass
    
    print("\n" + "="*60)
    print("🤖 Unified RAG Chatbot (Automated Version) - วิทยาลัยการคอมพิวเตอร์ มข.")
    print("="*60)
    print("🔀 ใช้ Hybrid Classification + Dynamic Collections!")
    print()
    
    # Initialize unified chatbot
    try:
        # Optional: Provide custom collection mapping
        # collection_mapping = {
        #     "allpeople": "allpeople_embedding",
        #     "contact": "services_embedding",
        #     "links": "services_embedding",
        #     ...
        # }
        # chatbot = UnifiedChatbotAutomated(collection_mapping=collection_mapping)
        
        chatbot = UnifiedChatbotAutomated()  # Auto-detect collections
    except Exception as e:
        print(f"\n❌ Error initializing chatbot: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Show initial help
    chatbot.show_help()
    
    # Main loop
    while True:
        print("\n" + "-"*60)
        try:
            question = input("\n❓ ถามมาเลย Automated Version (หรือพิมพ์ 'help'): ").strip()
        except EOFError:
            print("\n👋 ออกจากโปรแกรม")
            break
        
        if not question:
            continue
        
        # Handle special commands
        if question.lower() in ['exit', 'quit', 'ออก', 'จบ']:
            print("\n👋 ขอบคุณที่ใช้บริการ!")
            break
        
        if question.lower() in ['help', 'ช่วยเหลือ', 'คำแนะนำ']:
            chatbot.show_help()
            continue
        
        if question.lower() in ['agents', 'รายการ', 'agent', 'collections']:
            print("\n📋 รายการ Agent และ Collections:")
            for intent, config in chatbot.chatbot_map.items():
                print(f"   {config['icon']} {config['name']} → {config['collection']}")
            continue
        
        # Get answer
        try:
            print()
            answer = chatbot.answer(question)
            print("\n" + "="*60)
            print("🤖 คำตอบ:")
            print("="*60)
            print(answer)
        except Exception as e:
            print(f"\n❌ เกิดข้อผิดพลาด: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

