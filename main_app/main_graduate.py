# main_graduate.py - AstraDB Version สำหรับ graduate_data.py (หลักสูตรบัณฑิตศึกษา)

import sys
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
import os
from dotenv import load_dotenv
from typing import List, Optional
from astrapy import DataAPIClient

# PyThaiNLP imports for advanced Thai processing
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

# ✅ Helper function for safe Thai text output
def safe_print(text):
    """Safely print Thai text to terminal"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback for terminals that don't support Thai
        safe_text = text.encode('ascii', 'ignore').decode('ascii')
        print(f"[Thai text] {safe_text}")
    except Exception as e:
        print(f"[Output error: {e}]")

# ✅ เตรียม embedding
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize AstraDB client
token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
keyspace = os.getenv("ASTRA_DB_KEYSPACE", "default_keyspace")

if not token or not api_endpoint:
    print("❌ Error: Missing AstraDB credentials in .env file")
    exit(1)

client = DataAPIClient(token=token)
database = client.get_database_by_api_endpoint(api_endpoint)

# Get graduate collection
try:
    collection = database.get_collection("graduate_embedding")
    print(f"✅ Connected to collection: graduate_embedding")
except Exception as e:
    print(f"❌ Error accessing collection: {e}")
    exit(1)

# ✅ สร้าง Custom Retriever สำหรับ AstraDB (Graduate Programs Collection)
class GraduateRetriever(BaseRetriever):
    def __init__(self, collection, embedding):
        super().__init__()
        self._collection = collection
        self._embedding = embedding
        self._bm25_retriever = None
        self._documents_cache = None
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        safe_print(f"🔍 Debug: กำลังค้นหาหลักสูตรบัณฑิตศึกษาด้วย query: '{query}'")
        
        # ตรวจสอบว่าเป็นคำถามเกี่ยวกับหลักสูตรเฉพาะหรือไม่
        query_lower = query.lower()
        filter_category = None
        
        # ตรวจสอบว่าเป็นคำถามเกี่ยวกับปริญญาโท
        if any(word in query_lower for word in ["โท", "master", "ป.โท", "มหาบัณฑิต"]):
            filter_category = "master"
            safe_print("🎯 ตรวจพบคำค้นหาเกี่ยวกับปริญญาโท")
        
        # ตรวจสอบว่าเป็นคำถามเกี่ยวกับปริญญาเอก
        elif any(word in query_lower for word in ["เอก", "phd", "ph.d", "ป.เอก", "ดุษฎีบัณฑิต"]):
            filter_category = "phd"
            safe_print("🎯 ตรวจพบคำค้นหาเกี่ยวกับปริญญาเอก")
        
        # ถ้าต้องการข้อมูลทั้งหมด
        if any(word in query_lower for word in ["ทั้งหมด", "all", "รายการ", "มีหลักสูตรอะไรบ้าง"]):
            safe_print("🎯 ตรวจพบคำขอข้อมูลหลักสูตรทั้งหมด")
            return self._get_comprehensive_search(filter_category)
        
        # Try multiple search strategies
        print("🔍 กำลังค้นหาด้วย AstraDB hybrid search...")
        
        # Strategy 0: Thai Advanced Search
        thai_results = []
        is_thai_query = any('\u0e00' <= char <= '\u0e7f' for char in query)
        
        if is_thai_query:
            print("🔍 เริ่ม Advanced Thai Search...")
            thai_results = self._advanced_thai_search(query, filter_category)
            if thai_results:
                print(f"🎯 Advanced Thai Search: พบ {len(thai_results)} รายการ")
        
        # Strategy 1: Text search
        print("🔍 เริ่ม Text Search...")
        text_results = self._text_search(query, filter_category)
        
        # Strategy 2: Vector search
        print("🧠 เริ่ม Vector Search...")
        vector_results = self._vector_search(query, filter_category)
        
        # Combine results with hybrid scoring
        all_documents = []
        seen_content = set()
        candidate_docs = []
        
        print("🔄 รวมผลลัพธ์พร้อมคำนวณคะแนนรวม...")
        
        # Add Thai advanced search results first
        if thai_results:
            for i, doc in enumerate(thai_results[:5]):
                if doc.page_content not in seen_content:
                    thai_score = doc.metadata.get("thai_advanced_score", 0.0) or doc.metadata.get("thai_exact_score", 0.0)
                    search_type = doc.metadata.get("search_type", "thai_unknown")
                    
                    doc.metadata["combined_score"] = self._calculate_hybrid_score(doc, thai_score, 0.0, query)
                    
                    if thai_score > 20:
                        doc.metadata["combined_score"] += 0.2
                    elif thai_score > 10:
                        doc.metadata["combined_score"] += 0.1
                    
                    doc.metadata["bm25_score"] = thai_score
                    doc.metadata["vector_score"] = 0.0
                    candidate_docs.append(doc)
                    seen_content.add(doc.page_content)
                    
                    search_type_display = "🇹🇭 Advanced" if search_type == "thai_advanced" else "🎯 Exact"
                    program_name = doc.metadata.get('program_name', 'Unknown')
                    print(f"➕ Thai {search_type_display} #{i+1}: {program_name[:40]}... (Thai: {thai_score:.2f}, Combined: {doc.metadata['combined_score']:.4f})")
        
        # Add text search results
        for i, doc in enumerate(text_results):
            if doc.page_content not in seen_content:
                bm25_score = doc.metadata.get("bm25_score", 0.0)
                doc.metadata["combined_score"] = self._calculate_hybrid_score(doc, bm25_score, 0.0, query)
                candidate_docs.append(doc)
                seen_content.add(doc.page_content)
                program_name = doc.metadata.get('program_name', 'Unknown')
                print(f"➕ Text Search #{i+1}: {program_name[:40]}... (BM25: {bm25_score:.4f})")
        
        # Add vector search results
        for i, doc in enumerate(vector_results):
            if doc.page_content not in seen_content:
                vector_score = doc.metadata.get("vector_score", 0.0)
                doc.metadata["combined_score"] = self._calculate_hybrid_score(doc, 0.0, vector_score, query)
                candidate_docs.append(doc)
                seen_content.add(doc.page_content)
                program_name = doc.metadata.get('program_name', 'Unknown')
                print(f"➕ Vector Search #{i+1}: {program_name[:40]}... (Vector: {vector_score:.4f})")
            else:
                for existing_doc in candidate_docs:
                    if existing_doc.page_content == doc.page_content:
                        vector_score = doc.metadata.get("vector_score", 0.0)
                        bm25_score = existing_doc.metadata.get("bm25_score", 0.0)
                        existing_doc.metadata["vector_score"] = vector_score
                        existing_doc.metadata["combined_score"] = self._calculate_hybrid_score(existing_doc, bm25_score, vector_score, query)
                        program_name = existing_doc.metadata.get('program_name', 'Unknown')
                        print(f"🔄 อัปเดตคะแนน: {program_name[:40]}... (BM25: {bm25_score:.4f}, Vector: {vector_score:.4f})")
                        break
        
        # Sort by combined score
        candidate_docs.sort(key=lambda doc: doc.metadata.get("combined_score", 0), reverse=True)
        all_documents = candidate_docs
        
        # Show final ranking
        print("\n🏆 ผลลัพธ์สุดท้าย (เรียงตามคะแนนรวม):")
        print("-" * 60)
        for i, doc in enumerate(all_documents[:5], 1):
            combined_score = doc.metadata.get("combined_score", 0.0)
            bm25_score = doc.metadata.get("bm25_score", 0.0)
            vector_score = doc.metadata.get("vector_score", 0.0)
            search_type = doc.metadata.get("search_type", "unknown")
            program_name = doc.metadata.get("program_name", "Unknown")
            category = doc.metadata.get("category", "Unknown")
            
            type_icon = "🇹🇭" if "thai" in search_type else "📝" if "bm25" in search_type else "🧠" if "vector" in search_type else "❓"
            category_icon = "🎓" if category == "master" else "🔬" if category == "phd" else "📚"
            
            print(f"#{i}: {type_icon}{category_icon} {program_name[:50]}...")
            print(f"    🎯 Combined: {combined_score:.4f} | 📝 BM25/Thai: {bm25_score:.4f} | 🧠 Vector: {vector_score:.4f}")
            print(f"    🔍 Search Type: {search_type} | Category: {category}")
            print("-" * 40)
        
        print(f"📊 สรุป: Text={len(text_results)}, Vector={len(vector_results)}, รวม={len(all_documents)} (unique)")
        
        return all_documents[:15]
    
    def _get_comprehensive_search(self, filter_category: Optional[str] = None) -> List[Document]:
        """ค้นหาข้อมูลหลักสูตรแบบครอบคลุม"""
        print(f"🚀 เริ่มการค้นหาหลักสูตรแบบครอบคลุมจาก graduate_embedding collection...")
        if filter_category:
            print(f"   🔍 กรองเฉพาะ category: {filter_category}")
        
        all_documents = []
        
        try:
            query_filter = {"metadata.category": filter_category} if filter_category else {}
            results = self._collection.find(query_filter, limit=50)
            
            for result in results:
                doc = Document(
                    page_content=result.get("content", ""),
                    metadata=result.get("metadata", {})
                )
                all_documents.append(doc)
            
            print(f"📊 จาก graduate_embedding: {len(all_documents)} รายการ")
            
        except Exception as e:
            print(f"❌ Error in comprehensive search: {e}")
        
        print(f"🎯 พบข้อมูลหลักสูตรครอบคลุมรวม: {len(all_documents)} รายการ")
        return all_documents
    
    def _vector_search(self, query: str, filter_category: Optional[str] = None) -> List[Document]:
        """ค้นหาแบบ vector search"""
        all_documents = []
        
        try:
            print(f"🧠 Vector Search: กำลังสร้าง embedding สำหรับ query: '{query}'")
            
            query_vector = self._embedding.embed_query(query)
            print(f"📊 Vector Search: สร้าง embedding แล้ว (dimension: {len(query_vector)})")
            
            # Apply category filter if specified
            query_filter = {"metadata.category": filter_category} if filter_category else {}
            
            results = self._collection.find(
                query_filter,
                sort={"$vector": query_vector},
                limit=10,
                include_similarity=True
            )
            
            for result in results:
                similarity_score = result.get("$similarity", 0.0)
                metadata = result.get("metadata", {}).copy()
                metadata["vector_score"] = similarity_score
                metadata["search_type"] = "vector"
                
                doc = Document(
                    page_content=result.get("content", ""),
                    metadata=metadata
                )
                all_documents.append(doc)
            
            print(f"🧠 Vector search: พบ {len(all_documents)} documents")
            
            if all_documents:
                print("   🏆 Top Vector Matches:")
                for i, doc in enumerate(all_documents[:3]):
                    score = doc.metadata.get("vector_score", 0.0)
                    program_name = doc.metadata.get("program_name", "Unknown")
                    print(f"   #{i+1}: {program_name[:40]}... (score: {score:.4f})")
            
            return all_documents
            
        except Exception as e:
            print(f"❌ Error in vector search: {e}")
            return []
    
    def _ensure_bm25_initialized(self):
        """Initialize BM25 retriever"""
        if self._bm25_retriever is None:
            print("🔧 Initializing Enhanced BM25 retriever with Thai support...")
            try:
                results = self._collection.find({}, limit=100)
                documents = []
                
                for result in results:
                    doc = Document(
                        page_content=result.get("content", ""),
                        metadata=result.get("metadata", {})
                    )
                    documents.append(doc)
                
                self._documents_cache = documents
                print(f"📚 Loaded {len(documents)} graduate program documents for BM25")
                
                if documents and PYTHAINLP_AVAILABLE:
                    self._create_thai_bm25(documents)
                    print("✅ Enhanced Thai BM25 initialized successfully")
                elif documents:
                    self._bm25_retriever = BM25Retriever.from_documents(documents)
                    self._bm25_retriever.k = 10
                    print("✅ Standard BM25 initialized successfully")
                else:
                    print("⚠️ No documents found for BM25 initialization")
                    
            except Exception as e:
                print(f"❌ Error initializing BM25: {e}")
                self._bm25_retriever = None
    
    def _create_thai_bm25(self, documents: List[Document]):
        """สร้าง BM25 ที่ใช้ Thai tokenization"""
        try:
            tokenized_docs = []
            
            for doc in documents:
                content = doc.page_content
                normalized = normalize(content)
                tokens = word_tokenize(normalized, engine='newmm')
                
                stopwords = thai_stopwords()
                filtered_tokens = []
                
                for token in tokens:
                    token_clean = token.strip()
                    if (len(token_clean) > 1 and 
                        token_clean not in stopwords and
                        not token_clean.isspace()):
                        filtered_tokens.append(token_clean.lower())
                
                tokenized_docs.append(filtered_tokens)
            
            self._thai_bm25 = BM25Okapi(tokenized_docs)
            self._thai_bm25_docs = documents
            
            print(f"🇹🇭 Thai BM25 created with {len(tokenized_docs)} tokenized documents")
            
        except Exception as e:
            print(f"❌ Error creating Thai BM25: {e}")
            self._bm25_retriever = BM25Retriever.from_documents(documents)
            self._bm25_retriever.k = 10
    
    def _text_search(self, query: str, filter_category: Optional[str] = None) -> List[Document]:
        """Enhanced BM25 text search"""
        try:
            self._ensure_bm25_initialized()
            
            if hasattr(self, '_thai_bm25') and PYTHAINLP_AVAILABLE:
                return self._thai_bm25_search(query, filter_category)
            elif self._bm25_retriever is not None:
                return self._standard_bm25_search(query, filter_category)
            else:
                print("⚠️ BM25 not available, falling back to keyword search")
                return self._fallback_keyword_search(query, filter_category)
                
        except Exception as e:
            print(f"❌ Error in text search: {e}")
            return self._fallback_keyword_search(query, filter_category)
    
    def _thai_bm25_search(self, query: str, filter_category: Optional[str] = None) -> List[Document]:
        """ค้นหาด้วย Thai BM25"""
        try:
            print(f"🇹🇭 Thai BM25 Search: กำลังค้นหาด้วย query: '{query}'")
            
            normalized_query = normalize(query)
            query_tokens = word_tokenize(normalized_query, engine='newmm')
            
            stopwords = thai_stopwords()
            filtered_tokens = []
            
            for token in query_tokens:
                token_clean = token.strip().lower()
                if (len(token_clean) > 1 and 
                    token_clean not in stopwords and
                    not token_clean.isspace()):
                    filtered_tokens.append(token_clean)
            
            if not filtered_tokens:
                filtered_tokens = [token.lower() for token in query_tokens if len(token.strip()) > 1]
            
            print(f"   🔤 Query tokens: {query_tokens}")
            print(f"   🎯 Filtered tokens: {filtered_tokens}")
            
            bm25_scores = self._thai_bm25.get_scores(filtered_tokens)
            
            results = []
            for i, score in enumerate(bm25_scores):
                if score > 0:
                    doc = self._thai_bm25_docs[i]
                    
                    # Apply category filter
                    if filter_category and doc.metadata.get("category") != filter_category:
                        continue
                    
                    doc.metadata = doc.metadata.copy()
                    doc.metadata["bm25_score"] = score
                    doc.metadata["search_type"] = "thai_bm25"
                    doc.metadata["query_tokens"] = filtered_tokens
                    
                    results.append(doc)
            
            results.sort(key=lambda doc: doc.metadata.get("bm25_score", 0), reverse=True)
            
            print(f"📝 Thai BM25 search: พบ {len(results)} documents")
            
            if results:
                print("   🏆 Top Thai BM25 Matches:")
                for i, doc in enumerate(results[:3]):
                    score = doc.metadata.get("bm25_score", 0.0)
                    program_name = doc.metadata.get('program_name', 'Unknown')
                    print(f"   #{i+1}: {program_name[:40]}... (Thai BM25: {score:.4f})")
            
            return results[:10]
            
        except Exception as e:
            print(f"❌ Error in Thai BM25 search: {e}")
            return self._standard_bm25_search(query, filter_category)
    
    def _standard_bm25_search(self, query: str, filter_category: Optional[str] = None) -> List[Document]:
        """Standard BM25 search"""
        try:
            print(f"🔍 Standard BM25 Search: กำลังค้นหาด้วย query: '{query}'")
            
            all_bm25_results = []
            seen_content = set()
            
            variant_results = self._bm25_retriever.get_relevant_documents(query)
            scored_results = self._calculate_bm25_scores(variant_results, query)
            
            for doc, bm25_score in scored_results:
                # Apply category filter
                if filter_category and doc.metadata.get("category") != filter_category:
                    continue
                
                if doc.page_content not in seen_content:
                    doc.metadata = doc.metadata.copy()
                    doc.metadata["bm25_score"] = bm25_score
                    doc.metadata["search_type"] = "standard_bm25"
                    
                    all_bm25_results.append(doc)
                    seen_content.add(doc.page_content)
            
            all_bm25_results.sort(key=lambda doc: doc.metadata.get("bm25_score", 0), reverse=True)
            
            print(f"📝 Standard BM25 search: พบ {len(all_bm25_results)} documents")
            
            if all_bm25_results:
                print("   🏆 Top BM25 Matches:")
                for i, doc in enumerate(all_bm25_results[:3]):
                    score = doc.metadata.get("bm25_score", 0.0)
                    program_name = doc.metadata.get("program_name", "Unknown")
                    print(f"   #{i+1}: {program_name[:40]}... (BM25: {score:.4f})")
            
            return all_bm25_results[:10]
            
        except Exception as e:
            print(f"❌ Error in BM25 search: {e}")
            return self._fallback_keyword_search(query, filter_category)
    
    def _advanced_thai_search(self, query: str, filter_category: Optional[str] = None) -> List[Document]:
        """ค้นหาขั้นสูงด้วย PyThaiNLP"""
        if not PYTHAINLP_AVAILABLE:
            return self._exact_thai_keyword_search(query, filter_category)
        
        try:
            print(f"🇹🇭 Advanced Thai Search with PyThaiNLP: '{query}'")
            
            normalized_query = normalize(query)
            query_tokens = word_tokenize(normalized_query, engine='newmm')
            pos_tags = pos_tag(query_tokens, engine='perceptron')
            stopwords = thai_stopwords()
            
            meaningful_tokens = []
            for word, pos in pos_tags:
                if (pos in ['NOUN', 'VERB', 'ADJ', 'PROPN'] and 
                    word not in stopwords and 
                    len(word.strip()) > 1):
                    meaningful_tokens.append(word)
            
            if not meaningful_tokens:
                meaningful_tokens = [token for token in query_tokens if len(token.strip()) > 1]
            
            print(f"   🔤 Tokenized: {query_tokens}")
            print(f"   🎯 Meaningful: {meaningful_tokens}")
            
            query_filter = {"metadata.category": filter_category} if filter_category else {}
            all_results = list(self._collection.find(query_filter, limit=100))
            matched_docs = []
            
            for result in all_results:
                content = result.get("content", "")
                metadata = result.get("metadata", {})
                
                match_score = self._calculate_thai_match_score(
                    meaningful_tokens, query_tokens, content, metadata
                )
                
                if match_score > 0:
                    doc = Document(
                        page_content=content,
                        metadata={
                            **metadata,
                            "thai_advanced_score": match_score,
                            "thai_tokens": meaningful_tokens,
                            "search_type": "thai_advanced"
                        }
                    )
                    matched_docs.append((doc, match_score))
                    program_name = metadata.get('program_name', 'Unknown')
                    print(f"  ✅ Match: {program_name[:40]}... (score: {match_score:.2f})")
            
            matched_docs.sort(key=lambda x: x[1], reverse=True)
            result_docs = [doc for doc, score in matched_docs[:10]]
            print(f"🎯 Advanced Thai Search: Found {len(result_docs)} matches")
            
            return result_docs
            
        except Exception as e:
            print(f"❌ Error in advanced Thai search: {e}")
            return self._exact_thai_keyword_search(query, filter_category)
    
    def _calculate_thai_match_score(self, meaningful_tokens: List[str], all_tokens: List[str], 
                                   content: str, metadata: dict) -> float:
        """คำนวณคะแนนการ match แบบขั้นสูงสำหรับภาษาไทย"""
        score = 0.0
        
        content_lower = content.lower()
        program_name = metadata.get('program_name', '').lower()
        
        # Exact token matching in program name (highest priority)
        for token in meaningful_tokens:
            token_lower = token.lower()
            if token_lower in program_name:
                score += 15.0
        
        # Exact token matching in content
        for token in meaningful_tokens:
            if token.lower() in content_lower:
                score += 5.0
        
        # Partial matching
        for token in all_tokens:
            if len(token) > 2:
                token_lower = token.lower()
                if token_lower in program_name:
                    score += 8.0
                elif token_lower in content_lower:
                    score += 2.0
        
        # Graduate program-specific semantic bonus
        program_patterns = {
            'โท': ['โท', 'master', 'มหาบัณฑิต'],
            'เอก': ['เอก', 'phd', 'ดุษฎีบัณฑิต'],
            'บัณฑิต': ['บัณฑิต', 'graduate'],
            'หลักสูตร': ['หลักสูตร', 'program', 'curriculum'],
            'วิทยาการคอมพิวเตอร์': ['วิทยาการคอมพิวเตอร์', 'computer science'],
        }
        
        for token in meaningful_tokens:
            if token in program_patterns:
                related_words = program_patterns[token]
                for word in related_words:
                    if word in content_lower:
                        score += 3.0
        
        return score
    
    def _exact_thai_keyword_search(self, query: str, filter_category: Optional[str] = None) -> List[Document]:
        """ค้นหาแบบ exact match สำหรับคำสำคัญภาษาไทย"""
        try:
            thai_keyword_mappings = {
                "โท": ["โท", "master", "มหาบัณฑิต"],
                "เอก": ["เอก", "phd", "ดุษฎีบัณฑิต"],
                "บัณฑิต": ["บัณฑิต", "graduate"],
                "หลักสูตร": ["หลักสูตร", "program"],
                "วิทยาการคอมพิวเตอร์": ["วิทยาการคอมพิวเตอร์", "computer science"],
            }
            
            query_filter = {"metadata.category": filter_category} if filter_category else {}
            all_results = list(self._collection.find(query_filter, limit=100))
            matched_docs = []
            
            query_lower = query.lower().strip()
            target_keywords = thai_keyword_mappings.get(query_lower, [query_lower])
            
            print(f"🔍 Thai Exact Search: '{query}' → looking for keywords: {target_keywords}")
            
            for result in all_results:
                content = result.get("content", "").lower()
                metadata = result.get("metadata", {})
                
                match_score = 0.0
                matches = []
                
                for keyword in target_keywords:
                    if keyword in content:
                        match_score += 5.0
                        matches.append(f"content:{keyword}")
                
                if match_score > 0:
                    doc = Document(
                        page_content=result.get("content", ""),
                        metadata={
                            **metadata,
                            "thai_exact_score": match_score,
                            "thai_matches": matches,
                            "search_type": "thai_exact"
                        }
                    )
                    matched_docs.append((doc, match_score))
                    program_name = metadata.get('program_name', 'Unknown')
                    print(f"  ✅ Match: {program_name[:40]}... (score: {match_score:.2f})")
            
            matched_docs.sort(key=lambda x: x[1], reverse=True)
            result_docs = [doc for doc, score in matched_docs[:10]]
            print(f"🎯 Thai Exact Search: Found {len(result_docs)} matches")
            
            return result_docs
            
        except Exception as e:
            print(f"❌ Error in Thai exact search: {e}")
            return []
    
    def _calculate_bm25_scores(self, documents: List[Document], query: str) -> List[tuple]:
        """คำนวณคะแนน BM25 สำหรับ documents"""
        try:
            import re
            from collections import Counter
            
            query_terms = re.findall(r'\w+', query.lower())
            if not query_terms:
                return [(doc, 0.0) for doc in documents]
            
            scored_docs = []
            
            for doc in documents:
                content_lower = doc.page_content.lower()
                content_terms = re.findall(r'\w+', content_lower)
                
                if not content_terms:
                    scored_docs.append((doc, 0.0))
                    continue
                
                term_freq = Counter(content_terms)
                doc_length = len(content_terms)
                
                score = 0.0
                for term in query_terms:
                    if term in term_freq:
                        tf = term_freq[term]
                        score += (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * doc_length / 100))
                
                # Boost for graduate program context
                for term in query_terms:
                    if term in content_lower:
                        if any(keyword in content_lower for keyword in ["หลักสูตร", "program", "บัณฑิต"]):
                            score += 2.0
                        elif any(keyword in content_lower for keyword in ["โท", "เอก", "master", "phd"]):
                            score += 1.5
                        else:
                            score += 1.0
                
                scored_docs.append((doc, score))
            
            return scored_docs
            
        except Exception as e:
            print(f"❌ Error calculating BM25 scores: {e}")
            return [(doc, 0.0) for doc in documents]
    
    def _calculate_hybrid_score(self, doc: Document, bm25_score: float, vector_score: float, query: str) -> float:
        """คำนวณคะแนนรวมจาก BM25 และ Vector similarity (Improved v2 - Better context ranking)"""
        try:
            # ✅ ปรับ weights ให้ vector search มีน้ำหนักมากขึ้น
            bm25_weight = 0.3      # ลดลงจาก 0.4
            vector_weight = 0.5    # เพิ่มขึ้นจาก 0.4
            bonus_weight = 0.2
            
            if bm25_score > 15:
                normalized_bm25 = min(bm25_score / 50.0, 1.0)
            else:
                normalized_bm25 = min(bm25_score / 8.0, 1.0)
            normalized_vector = vector_score
            
            bonus_score = 0.0
            query_lower = query.lower()
            content_lower = doc.page_content.lower()
            
            # ✅ STRONG Penalty สำหรับ generic/irrelevant content (เพิ่มความเข้มงวด!)
            generic_patterns = [
                ("ติดต่อเรา", 0.7),           # Penalty สูงมาก!
                ("เกี่ยวกับเรา", 0.7),         # Penalty สูงมาก!
                ("ประวัติความเป็นมา", 0.6),
                ("โครงสร้างองค์กร", 0.6),
                ("วิสัยทัศน์", 0.5),
                ("พันธกิจ", 0.5),
                ("ผู้บริหาร", 0.5),
                ("บุคลากร", 0.5)
            ]
            
            penalty = 0.0
            for pattern, penalty_value in generic_patterns:
                if pattern in content_lower:
                    # ถ้าเป็น content สั้นๆ ให้ penalty สูงกว่า
                    if len(content_lower) < 300:
                        penalty += penalty_value
                    else:
                        penalty += penalty_value * 0.5  # ลด penalty ถ้า content ยาว
            
            # ✅ STRONG Bonus สำหรับ specific program information (เพิ่มโบนัส!)
            specific_indicators = [
                ("แผนการศึกษา", 0.3),      # เพิ่มจาก 0.15
                ("รหัสสาขาวิชา", 0.3),     # เพิ่มจาก 0.15
                ("จำนวนรับ", 0.25),         # เพิ่มจาก 0.15
                ("ระบบการศึกษา", 0.25),    # เพิ่มจาก 0.15
                ("แผน ก", 0.2),
                ("แผน ข", 0.2),
                ("แบบ 1.", 0.2),
                ("แบบ 2.", 0.2),
                ("ภาคต้น", 0.15),
                ("ภาคปลาย", 0.15),
                ("โครงการพิเศษ", 0.2),
                ("นานาชาติ", 0.2)
            ]
            
            specificity_bonus = 0.0
            for indicator, bonus_value in specific_indicators:
                if indicator in content_lower:
                    specificity_bonus += bonus_value
            
            specificity_bonus = min(specificity_bonus, 0.8)  # เพิ่ม cap จาก 0.6 เป็น 0.8
            
            # Query word matching bonus (ลดโบนัสลงเล็กน้อย)
            query_words = [word for word in query_lower.split() if len(word) >= 2]
            for word in query_words:
                if word in content_lower:
                    if any(indicator in content_lower for indicator in ["หลักสูตร", "program", "บัณฑิต"]):
                        bonus_score += 0.3  # ลดจาก 0.5
                    elif any(indicator in content_lower for indicator in ["โท", "เอก", "master", "phd"]):
                        bonus_score += 0.2  # ลดจาก 0.3
                    else:
                        bonus_score += 0.1  # ลดจาก 0.2
            
            # Program-specific bonuses
            program_bonuses = {
                "หลักสูตร": 0.2 if "หลักสูตร" in content_lower else 0,  # ลดจาก 0.3
                "บัณฑิต": 0.15 if "บัณฑิต" in content_lower else 0,     # ลดจาก 0.2
                "โท": 0.15 if "โท" in content_lower else 0,
                "เอก": 0.15 if "เอก" in content_lower else 0,
                "master": 0.15 if "master" in content_lower else 0,
                "phd": 0.15 if "phd" in content_lower else 0,
            }
            
            for term, bonus in program_bonuses.items():
                if term in query_lower:
                    bonus_score += bonus
            
            # Add specificity bonus (ส่วนสำคัญที่สุด!)
            bonus_score += specificity_bonus
            
            # ✅ เพิ่ม Debug logging เพื่อดูว่า penalty/bonus ทำงานถูกต้องหรือไม่
            # if penalty > 0 or specificity_bonus > 0:
            #     program_name = doc.metadata.get('program_name', 'Unknown')[:30]
            #     print(f"   📊 {program_name}... | Penalty: -{penalty:.2f} | Bonus: +{specificity_bonus:.2f}")
            
            normalized_bonus = min(bonus_score, 1.0)
            
            # Calculate combined score
            combined_score = (
                bm25_weight * normalized_bm25 +
                vector_weight * normalized_vector +
                bonus_weight * normalized_bonus
            )
            
            # ✅ Apply penalty AGGRESSIVELY (คูณด้วย 1.5 เพื่อให้มีผลมากขึ้น!)
            combined_score = max(0.0, combined_score - (penalty * 1.5))
            
            return combined_score
            
        except Exception as e:
            print(f"❌ Error calculating hybrid score: {e}")
            return max(bm25_score / 8.0, vector_score)
    
    def _fallback_keyword_search(self, query: str, filter_category: Optional[str] = None) -> List[Document]:
        """Fallback keyword search"""
        print("🔄 Using fallback keyword search for graduate programs...")
        all_documents = []
        
        try:
            keywords = self._extract_search_keywords(query)
            query_filter = {"metadata.category": filter_category} if filter_category else {}
            results = self._collection.find(query_filter, limit=50)
            
            for result in results:
                content = result.get("content", "").lower()
                original_content = result.get("content", "")
                
                matched = False
                for keyword in keywords:
                    if keyword.lower() in content:
                        matched = True
                        break
                
                if matched:
                    doc = Document(
                        page_content=original_content,
                        metadata=result.get("metadata", {})
                    )
                    if not any(d.page_content == doc.page_content for d in all_documents):
                        all_documents.append(doc)
            
            print(f"📝 Fallback search: พบ {len(all_documents)} documents")
            return all_documents[:10]
            
        except Exception as e:
            print(f"❌ Error in fallback search: {e}")
            return []
    
    def _extract_search_keywords(self, query: str) -> List[str]:
        """แยกคำสำคัญจากคำค้นหา"""
        import re
        keywords = []
        
        if PYTHAINLP_AVAILABLE:
            try:
                tokens = word_tokenize(query, engine='newmm')
                stop_words = thai_stopwords()
                custom_stops = {"ขอ", "ข้อมูล", "หา", "ค้นหา", "บอก", "แสดง", "ใคร", "คือ"}
                all_stop_words = stop_words.union(custom_stops)
                
                filtered_tokens = [
                    token.strip() for token in tokens 
                    if token.strip() not in all_stop_words 
                    and len(token.strip()) >= 2
                    and not token.isspace()
                ]
                
                keywords.extend(filtered_tokens)
                print(f"🔤 PyThaiNLP tokenized: {tokens}")
                print(f"🔤 Filtered keywords: {filtered_tokens}")
                
            except Exception as e:
                print(f"⚠️ PyThaiNLP error: {e}")
        
        if not PYTHAINLP_AVAILABLE or not keywords:
            stop_words = ["ขอ", "ข้อมูล", "หา", "ค้นหา", "บอก", "แสดง", "ใคร", "คือ", "ของ", "ใน", "ที่", "และ", "หรือ"]
            
            query_clean = query
            for stop_word in stop_words:
                query_clean = query_clean.replace(stop_word, " ")
            query_clean = re.sub(r'\s+', ' ', query_clean).strip()
            
            if query_clean and len(query_clean) >= 2:
                keywords.append(query_clean)
            
            words_by_space = query.split()
            for word in words_by_space:
                clean_word = word.strip()
                if clean_word not in stop_words and len(clean_word) >= 2:
                    keywords.append(clean_word)
            
            thai_pattern = r'[ก-๙]{3,20}'
            potential_terms = re.findall(thai_pattern, query)
            for term in potential_terms:
                if term not in stop_words and len(term) >= 3 and term not in keywords:
                    keywords.append(term)
        
        if query.strip() not in keywords:
            keywords.append(query.strip())
        
        unique_keywords = []
        for keyword in keywords:
            if keyword not in unique_keywords and len(keyword) >= 2:
                unique_keywords.append(keyword)
        
        print(f"🔤 Debug: Query '{query}' -> Final Keywords: {unique_keywords}")
        return unique_keywords

retriever = GraduateRetriever(collection, embedding)

# ✅ สร้าง Prompt - สำหรับหลักสูตรบัณฑิตศึกษา
PROMPT = PromptTemplate.from_template("""
บริบทต่อไปนี้คือข้อมูลเกี่ยวกับหลักสูตรบัณฑิตศึกษา (ปริญญาโท และ ปริญญาเอก) 
ในคณะวิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น

คุณคือผู้ช่วยที่ให้ข้อมูลเกี่ยวกับหลักสูตรบัณฑิตศึกษา โดยเฉพาะ:
- หลักสูตรปริญญาโท (Master's Program)
- หลักสูตรปริญญาเอก (Ph.D. Program)

สำคัญ: ให้ตรวจสอบข้อมูลในบริบทอย่างละเอียด หากมีข้อมูลที่ตรงกับคำถาม ให้นำมาตอบทันที 
อย่าตอบว่าไม่พบข้อมูลถ้าจริงๆ แล้วมีข้อมูลอยู่

หากคำถามเกี่ยวกับรายการหลักสูตร ให้แสดงผลแบบรายการที่ชัดเจน:
- ใช้หัวข้อชัดเจน เช่น "หลักสูตรบัณฑิตศึกษา คณะวิทยาลัยการคอมพิวเตอร์:"
- แยกแต่ละหลักสูตรเป็นบรรทัดใหม่
- ใช้เครื่องหมาย • หรือ - นำหน้าแต่ละหลักสูตร
- ระบุประเภท (ปริญญาโท/เอก) ชื่อหลักสูตร และข้อมูลสำคัญ

หากคำถามเกี่ยวกับข้อมูลเฉพาะของหลักสูตรใดหลักสูตรหนึ่ง ให้แสดงข้อมูลที่ครบถ้วน:
- ชื่อหลักสูตร (ทั้งไทยและอังกฤษ)
- ระดับการศึกษา (ปริญญาโท/เอก)
- คุณสมบัติผู้สมัคร
- โครงสร้างหลักสูตร
- ค่าใช้จ่าย
- ระยะเวลาการศึกษา
- ข้อมูลการติดต่อ

หากไม่มีข้อมูลที่ตรงกับคำถามเลยในบริบท ให้ตอบว่า "ขอโทษ ฉันไม่พบข้อมูลหลักสูตรบัณฑิตศึกษาที่คุณต้องการในระบบ"

---------------------
{context}
---------------------
คำถาม: {question}
คำตอบ (จัดรูปแบบให้อ่านง่าย):
""")

# ✅ โหลด Chat Model
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

if not openrouter_api_key:
    print("⚠️ Warning: OPENROUTER_API_KEY not found in .env file")
    llm = None
else:
    try:
        llm = ChatOpenAI(
            model="openai/gpt-4o-mini",
            temperature=0,
            openai_api_key=openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/your-repo",
                "X-Title": "Graduate Programs RAG Chatbot"
            }
        )
        print("✅ OpenRouter LLM initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing LLM: {e}")
        llm = None

# ✅ สร้าง Manual QA Function
def manual_qa_chain(question: str) -> str:
    """
    Manual QA chain สำหรับหลักสูตรบัณฑิตศึกษา
    """
    try:
        print(f"🔍 กำลังค้นหาข้อมูลหลักสูตรบัณฑิตศึกษาสำหรับคำถาม: {question}")
        print("🌐 ใช้ AstraDB Cloud Vector Database (astrapy) - Graduate Collection")
        print(f"📚 Collection: graduate_embedding")
        
        # ขั้นตอน 1: ดึงข้อมูลจาก retriever
        retrieved_docs = retriever.get_relevant_documents(question)
        
        if not retrieved_docs:
            return "ขอโทษ ฉันไม่พบข้อมูลหลักสูตรบัณฑิตศึกษาในระบบ"
        
        print(f"📚 พบข้อมูลหลักสูตร {len(retrieved_docs)} รายการจาก AstraDB")
        
        # ขั้นตอน 2: แสดงผลลัพธ์ทั้งหมดก่อน
        print("\n" + "="*60)
        print("📋 ผลการค้นหาหลักสูตรบัณฑิตศึกษาทั้งหมดจาก AstraDB (Graduate Collection):")
        print("="*60)
        
        for i, doc in enumerate(retrieved_docs, 1):
            combined_score = doc.metadata.get('combined_score', 0.0)
            bm25_score = doc.metadata.get('bm25_score', 0.0)
            vector_score = doc.metadata.get('vector_score', 0.0)
            program_name = doc.metadata.get('program_name', 'Unknown')
            category = doc.metadata.get('category', 'Unknown')
            
            category_display = "🎓 ปริญญาโท" if category == "master" else "🔬 ปริญญาเอก" if category == "phd" else "📚 บัณฑิตศึกษา"
            
            print(f"\n🔸 ผลลัพธ์ที่ {i}:")
            print(f"   {category_display}: {program_name}")
            if combined_score > 0:
                print(f"   📊 คะแนนความเกี่ยวข้อง: {combined_score:.4f} (BM25: {bm25_score:.4f}, Vector: {vector_score:.4f})")
            print("-" * 40)
            print(doc.page_content.strip()[:300] + "...")
            print("-" * 40)
        
        print("\n" + "="*60)
        
        # ขั้นตอน 3: ใช้ข้อมูลทั้งหมดที่ค้นหาได้
        selected_docs = retrieved_docs
        
        print(f"🎯 ใช้ข้อมูลหลักสูตรทั้งหมด {len(selected_docs)} รายการ สำหรับการตอบคำถาม")
        print("="*60)
        
        # จัดเตรียม context สำหรับ LLM
        context_parts = []
        print("\n📝 CONTEXT สำหรับ LLM:")
        print("-" * 40)
        
        for i, doc in enumerate(selected_docs, 1):
            program_name = doc.metadata.get('program_name', 'Unknown')
            category = doc.metadata.get('category', 'Unknown')
            print(f"📄 Context {i} ({category}: {program_name[:30]}...): {doc.page_content.strip()[:100]}...")
            context_parts.append(f"ข้อมูลหลักสูตรที่ {i} ({category}):\n{doc.page_content}\n")
        
        context = "\n".join(context_parts)
        print("\n" + "="*60)
        
        # ขั้นตอน 3: สร้าง prompt
        formatted_prompt = PROMPT.format(
            context=context,
            question=question
        )
        
        print("💭 กำลังประมวลผลคำตอบ...")
        
        # ขั้นตอน 4: เรียกใช้ LLM
        if llm is None:
            return "⚠️ ไม่สามารถตอบคำถามได้ เนื่องจากไม่มี API key สำหรับ LLM"
        
        try:
            response = llm.invoke(formatted_prompt)
            return response.content.strip()
        except Exception as llm_error:
            print(f"❌ LLM Error: {llm_error}")
            return f"พบข้อมูลหลักสูตรที่เกี่ยวข้อง {len(retrieved_docs)} รายการ แต่ไม่สามารถประมวลผลคำตอบได้"
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        return "ขอโทษ เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่อีกครั้ง"

# ✅ เริ่มถาม
if __name__ == "__main__":
    # Fix encoding for Windows terminal
    if sys.platform == "win32":
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    
    print("🎓 ระบบถาม-ตอบ หลักสูตรบัณฑิตศึกษา คณะคอมพิวเตอร์ มข.")
    print("🌐 ใช้ AstraDB Cloud Vector Database - Graduate Collection")
    print(f"📚 Collection: graduate_embedding")
    print("📌 รองรับ: ปริญญาโท (Master) และ ปริญญาเอก (Ph.D.)")
    print("พิมพ์ 'exit' เพื่อออก\n")

    while True:
        question = input("❓ ถามเกี่ยวกับหลักสูตรบัณฑิตศึกษา: ")
        if question.lower() == "exit":
            break

        result = manual_qa_chain(question)
        print("🤖 คำตอบ:", result)
        print("-" * 50)

