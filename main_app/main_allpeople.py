# main_allpeople.py - AstraDB Version สำหรับ allpeople_data.py โดยเฉพาะ (Enhanced with PyThaiNLP)

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
from typing import List, Any 
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

# Get single collection for allpeople data
try:
    collection = database.get_collection("allpeople_embedding")
    print(f"✅ Connected to collection: allpeople_embedding")
except Exception as e:
    print(f"❌ Error accessing collection: {e}")
    exit(1)

# ✅ สร้าง Custom Retriever สำหรับ AstraDB (Single Collection)
class AllPeopleRetriever(BaseRetriever):
    collection: Any = None
    embedding: Any = None
    bm25_retriever: Any = None
    documents_cache: Any = None
    thai_bm25: Any = None
    thai_bm25_docs: Any = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, collection, embedding):
        super().__init__()
        self.collection = collection
        self.embedding = embedding
        self.bm25_retriever = None  # Will be initialized when first used
        self.documents_cache = None  # Cache documents for BM25
        self.thai_bm25 = None
        self.thai_bm25_docs = None
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        safe_print(f"🔍 Debug: กำลังค้นหาอาจารย์ด้วย query: '{query}'")
        
        # ถ้าต้องการข้อมูลทั้งหมด ให้ใช้วิธีพิเศษ
        if any(word in query.lower() for word in ["ทั้งหมด", "ทุกคน", "all", "15", "20", "30", "รายชื่อ"]):
            safe_print("🎯 ตรวจพบคำขอข้อมูลทั้งหมด - ใช้การค้นหาแบบครอบคลุม")
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

        # Strategy 1: Text search first for exact name matches
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
                    thai_score = doc.metadata.get("thai_advanced_score", 0.0) or doc.metadata.get("thai_exact_score", 0.0)
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
                    
                    search_type_display = "🇹🇭 Advanced" if search_type == "thai_advanced" else "🎯 Exact"
                    faculty_name = doc.metadata.get('firstname_th', 'Unknown') + " " + doc.metadata.get('lastname_th', '')
                    print(f"➕ Thai {search_type_display} #{i+1}: {faculty_name} (Thai: {thai_score:.2f}, Combined: {doc.metadata['combined_score']:.4f})")
        
        # Add text search results
        for i, doc in enumerate(text_results):
            if doc.page_content not in seen_content:
                bm25_score = doc.metadata.get("bm25_score", 0.0)
                doc.metadata["combined_score"] = self._calculate_hybrid_score(doc, bm25_score, 0.0, query)
                candidate_docs.append(doc)
                seen_content.add(doc.page_content)
                faculty_name = doc.metadata.get('firstname_th', 'Unknown') + " " + doc.metadata.get('lastname_th', '')
                print(f"➕ Text Search #{i+1}: {faculty_name} (BM25: {bm25_score:.4f})")
        
        # Add vector search results
        for i, doc in enumerate(vector_results):
            if doc.page_content not in seen_content:
                vector_score = doc.metadata.get("vector_score", 0.0)
                doc.metadata["combined_score"] = self._calculate_hybrid_score(doc, 0.0, vector_score, query)
                candidate_docs.append(doc)
                seen_content.add(doc.page_content)
                faculty_name = doc.metadata.get('firstname_th', 'Unknown') + " " + doc.metadata.get('lastname_th', '')
                print(f"➕ Vector Search #{i+1}: {faculty_name} (Vector: {vector_score:.4f})")
            else:
                # Update existing document with vector score
                for existing_doc in candidate_docs:
                    if existing_doc.page_content == doc.page_content:
                        vector_score = doc.metadata.get("vector_score", 0.0)
                        bm25_score = existing_doc.metadata.get("bm25_score", 0.0)
                        existing_doc.metadata["vector_score"] = vector_score
                        existing_doc.metadata["combined_score"] = self._calculate_hybrid_score(existing_doc, bm25_score, vector_score, query)
                        faculty_name = existing_doc.metadata.get('firstname_th', 'Unknown') + " " + existing_doc.metadata.get('lastname_th', '')
                        print(f"🔄 อัปเดตคะแนน: {faculty_name} (BM25: {bm25_score:.4f}, Vector: {vector_score:.4f})")
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
            faculty_name = doc.metadata.get('firstname_th', 'Unknown') + " " + doc.metadata.get('lastname_th', '')
            
            # Show search type icon
            type_icon = "🇹🇭" if "thai" in search_type else "📝" if "bm25" in search_type else "🧠" if "vector" in search_type else "❓"
            
            print(f"#{i}: {type_icon} {faculty_name}")
            print(f"    🎯 Combined: {combined_score:.4f} | 📝 BM25/Thai: {bm25_score:.4f} | 🧠 Vector: {vector_score:.4f}")
            print(f"    🔍 Search Type: {search_type}")
            print("-" * 40)
        
        print(f"📊 สรุป: Text={len(text_results)}, Vector={len(vector_results)}, รวม={len(all_documents)} (unique)")
        
        return all_documents[:20]  # Return top 20 results
    
    def _get_comprehensive_search(self) -> List[Document]:
        """ค้นหาข้อมูลแบบครอบคลุมทั้งหมดจาก collection"""
        print("🚀 เริ่มการค้นหาแบบครอบคลุมจาก allpeople_embedding collection...")
        
        all_documents = []
        
        try:
            print("🔍 ค้นหาจาก allpeople_embedding collection...")
            results = self.collection.find({}, limit=65)  # Get up to 50 records
            
            for result in results:
                doc = Document(
                    page_content=result.get("content", ""),
                    metadata=result.get("metadata", {})
                )
                all_documents.append(doc)
            
            print(f"📊 จาก allpeople_embedding: {len(all_documents)} รายการ")
            
        except Exception as e:
            print(f"❌ Error in comprehensive search: {e}")
        
        print(f"🎯 พบข้อมูลครอบคลุมรวม: {len(all_documents)} รายการ")
        return all_documents
    
    def _vector_search(self, query: str) -> List[Document]:
        """ค้นหาแบบ vector search จาก collection พร้อมคะแนนความใกล้เคียง"""
        all_documents = []
        
        try:
            print(f"🧠 Vector Search: กำลังสร้าง embedding สำหรับ query: '{query}'")
            
            # Generate query embedding
            query_vector = self.embedding.embed_query(query)
            print(f"📊 Vector Search: สร้าง embedding แล้ว (dimension: {len(query_vector)})")
            
            # Perform vector search with similarity scores
            results = self.collection.find(
                {},
                sort={"$vector": query_vector},
                limit=15,  # Top 15 semantic matches
                include_similarity=True  # Include cosine similarity scores
            )
            
            for result in results:
                # Get similarity score (cosine similarity from AstraDB)
                similarity_score = result.get("$similarity", 0.0)
                
                # Create metadata with score
                metadata = result.get("metadata", {}).copy()
                metadata["vector_score"] = similarity_score
                metadata["search_type"] = "vector"
                
                doc = Document(
                    page_content=result.get("content", ""),
                    metadata=metadata
                )
                all_documents.append(doc)
            
            print(f"🧠 Vector search: พบ {len(all_documents)} documents จาก semantic similarity")
            
            # Show top matches with scores
            if all_documents:
                print("   🏆 Top Vector Matches:")
                for i, doc in enumerate(all_documents[:3]):
                    score = doc.metadata.get("vector_score", 0.0)
                    content_preview = doc.page_content[:50]
                    print(f"   #{i+1}: {content_preview}... (score: {score:.4f})")
            
            return all_documents
            
        except Exception as e:
            print(f"❌ Error in vector search: {e}")
            return []

    def _ensure_bm25_initialized(self):
        """Initialize BM25 retriever with Thai tokenization support"""
        if self.bm25_retriever is None:
            print("🔧 Initializing Enhanced BM25 retriever with Thai support...")
            try:
                # Get all documents from collection for BM25
                results = self.collection.find({}, limit=200)  # Increase limit for better BM25 corpus
                documents = []
                
                for result in results:
                    doc = Document(
                        page_content=result.get("content", ""),
                        metadata=result.get("metadata", {})
                    )
                    documents.append(doc)
                
                self.documents_cache = documents
                print(f"📚 Loaded {len(documents)} documents for BM25")
                
                # Create Enhanced BM25 with Thai tokenization
                if documents and PYTHAINLP_AVAILABLE:
                    self._create_thai_bm25(documents)
                    print("✅ Enhanced Thai BM25 initialized successfully")
                elif documents:
                    # Fallback to standard BM25
                    self.bm25_retriever = BM25Retriever.from_documents(documents)
                    self.bm25_retriever.k = 15  # Return top 15 results
                    print("✅ Standard BM25 initialized successfully (PyThaiNLP not available)")
                else:
                    print("⚠️ No documents found for BM25 initialization")
                    
            except Exception as e:
                print(f"❌ Error initializing BM25: {e}")
                self.bm25_retriever = None
    
    def _create_thai_bm25(self, documents: List[Document]):
        """สร้าง BM25 ที่ใช้ Thai tokenization"""
        try:
            # Tokenize all documents with PyThaiNLP
            tokenized_docs = []
            
            for doc in documents:
                content = doc.page_content
                
                # Normalize and tokenize
                normalized = normalize(content)
                tokens = word_tokenize(normalized, engine='newmm')
                
                # Filter meaningful tokens
                stopwords = thai_stopwords()
                filtered_tokens = []
                
                for token in tokens:
                    token_clean = token.strip()
                    if (len(token_clean) > 1 and 
                        token_clean not in stopwords and
                        not token_clean.isspace()):
                        filtered_tokens.append(token_clean.lower())
                
                tokenized_docs.append(filtered_tokens)
            
            # Create BM25 with tokenized documents
            self.thai_bm25 = BM25Okapi(tokenized_docs)
            self.thai_bm25_docs = documents  # Keep reference to original docs
            
            print(f"🇹🇭 Thai BM25 created with {len(tokenized_docs)} tokenized documents")
            
        except Exception as e:
            print(f"❌ Error creating Thai BM25: {e}")
            # Fallback to standard BM25
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            self.bm25_retriever.k = 15

    def _preprocess_query_for_bm25(self, query: str) -> str:
        """ประมวลผลคำถามก่อนส่งให้ BM25 เพื่อแก้ปัญหาการไม่เว้นวรรค"""
        import re
        
        # Remove common prefixes but keep the core name
        processed = query.replace("ขอข้อมูล", "").replace("อาจารย์", "").strip()
        
        # Don't try to add spaces - just return the clean name
        # BM25 should handle Thai text better without forced spacing
        
        print(f"🔤 Query preprocessing: '{query}' → '{processed}'")
        return processed if processed else query

    def _text_search(self, query: str) -> List[Document]:
        """Enhanced BM25 text search with Thai tokenization support"""
        try:
            # Ensure BM25 is initialized
            self._ensure_bm25_initialized()
            
            # Check if Thai BM25 is available
            if hasattr(self, 'thai_bm25') and self.thai_bm25 is not None and PYTHAINLP_AVAILABLE:
                return self.thai_bm25_search(query)
            elif self.bm25_retriever is not None:
                return self._standard_bm25_search(query)
            else:
                print("⚠️ BM25 not available, falling back to keyword search")
                return self._fallback_keyword_search(query)
                
        except Exception as e:
            print(f"❌ Error in text search: {e}")
            return self._fallback_keyword_search(query)
    
    def _thai_bm25_search(self, query: str) -> List[Document]:
        """ค้นหาด้วย Thai BM25 ที่ใช้ PyThaiNLP tokenization"""
        try:
            print(f"🇹🇭 Thai BM25 Search: กำลังค้นหาด้วย query: '{query}'")
            
            # Tokenize query with PyThaiNLP
            normalized_query = normalize(query)
            query_tokens = word_tokenize(normalized_query, engine='newmm')
            
            # Filter meaningful tokens
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
            
            # Get BM25 scores
            bm25_scores = self.thai_bm25.get_scores(filtered_tokens)
            
            # Create results with scores
            results = []
            for i, score in enumerate(bm25_scores):
                if score > 0:  # Only include documents with positive scores
                    doc = self.thai_bm25_docs[i]
                    
                    # Add BM25 score to metadata
                    doc.metadata = doc.metadata.copy()  # Avoid modifying original
                    doc.metadata["bm25_score"] = score
                    doc.metadata["search_type"] = "thai_bm25"
                    doc.metadata["query_tokens"] = filtered_tokens
                    
                    results.append(doc)
            
            # Sort by BM25 score (descending)
            results.sort(key=lambda doc: doc.metadata.get("bm25_score", 0), reverse=True)
            
            print(f"📝 Thai BM25 search: พบ {len(results)} documents")
            
            # Show top matches
            if results:
                print("   🏆 Top Thai BM25 Matches:")
                for i, doc in enumerate(results[:3]):
                    score = doc.metadata.get("bm25_score", 0.0)
                    faculty_name = doc.metadata.get('firstname_th', 'Unknown') + " " + doc.metadata.get('lastname_th', '')
                    print(f"   #{i+1}: {faculty_name} (Thai BM25: {score:.4f})")
            
            return results[:15]  # Return top 15
            
        except Exception as e:
            print(f"❌ Error in Thai BM25 search: {e}")
            return self._standard_bm25_search(query)
    
    def _standard_bm25_search(self, query: str) -> List[Document]:
        """Standard BM25 search (fallback when PyThaiNLP not available)"""
        try:
            print(f"🔍 Standard BM25 Search: กำลังค้นหาด้วย query: '{query}'")
            
            # Try multiple query variations for better matching
            query_variations = [
                query,  # Original query
                self._preprocess_query_for_bm25(query),  # Preprocessed
                query.replace("อาจารย์", "").replace("ขอข้อมูล", "").strip(),  # Clean version
            ]
            
            # Remove duplicates while preserving order
            unique_variations = []
            for q in query_variations:
                if q and q not in unique_variations:
                    unique_variations.append(q)
            
            all_bm25_results = []
            seen_content = set()
            
            for i, q_variant in enumerate(unique_variations):
                print(f"🔍 BM25 Variant #{i+1}: '{q_variant}'")
                try:
                    variant_results = self.bm25_retriever.get_relevant_documents(q_variant)
                    
                    # Calculate BM25 scores for this variant
                    scored_results = self._calculate_bm25_scores(variant_results, q_variant)
                    
                    # Add unique results with scores
                    for doc, bm25_score in scored_results:
                        if doc.page_content not in seen_content:
                            # Add enhanced BM25 score to metadata
                            doc.metadata = doc.metadata.copy()
                            doc.metadata["bm25_score"] = bm25_score
                            doc.metadata["search_type"] = "standard_bm25"
                            doc.metadata["query_variant"] = q_variant
                            
                            all_bm25_results.append(doc)
                            seen_content.add(doc.page_content)
                    
                    print(f"   Found {len(variant_results)} docs (unique so far: {len(all_bm25_results)})")
                    
                except Exception as e:
                    print(f"   Error with variant '{q_variant}': {e}")
            
            # Sort by BM25 score (descending)
            all_bm25_results.sort(key=lambda doc: doc.metadata.get("bm25_score", 0), reverse=True)
            
            print(f"📝 Standard BM25 search: พบ {len(all_bm25_results)} documents รวม")
            
            # Show top matches with scores
            if all_bm25_results:
                print("   🏆 Top BM25 Matches:")
                for i, doc in enumerate(all_bm25_results[:3]):
                    score = doc.metadata.get("bm25_score", 0.0)
                    faculty_name = doc.metadata.get('firstname_th', 'Unknown') + " " + doc.metadata.get('lastname_th', '')
                    print(f"   #{i+1}: {faculty_name} (BM25: {score:.4f})")
            
            return all_bm25_results[:15]  # Limit to top 15
            
        except Exception as e:
            print(f"❌ Error in BM25 search: {e}")
            return self._fallback_keyword_search(query)
    
    def _calculate_bm25_scores(self, documents: List[Document], query: str) -> List[tuple]:
        """คำนวณคะแนน BM25 สำหรับ documents (ปรับปรุงสำหรับข้อมูลอาจารย์)"""
        try:
            # Simple BM25-like scoring based on term frequency and document length
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
                
                # Calculate term frequency
                term_freq = Counter(content_terms)
                doc_length = len(content_terms)
                
                # Simple BM25-like score
                score = 0.0
                for term in query_terms:
                    if term in term_freq:
                        tf = term_freq[term]
                        # Simplified BM25 formula (without IDF calculation)
                        score += (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * doc_length / 50))
                
                # Boost score for exact matches in names and positions
                # Check for name matches (higher boost for faculty data)
                for term in query_terms:
                    if term in content_lower:
                        # Higher boost for name matches in faculty data
                        if any(keyword in content_lower for keyword in ["ชื่อ", "name", "อาจารย์"]):
                            score += 3.0  # High boost for name context
                        elif any(keyword in content_lower for keyword in ["ตำแหน่ง", "position", "หัวหน้า"]):
                            score += 2.0  # Medium boost for position context
                        else:
                            score += 1.0  # Base boost for other matches
                
                scored_docs.append((doc, score))
            
            return scored_docs
            
        except Exception as e:
            print(f"❌ Error calculating BM25 scores: {e}")
            return [(doc, 0.0) for doc in documents]
    
    def _calculate_hybrid_score(self, doc: Document, bm25_score: float, vector_score: float, query: str) -> float:
        """คำนวณคะแนนรวมจาก BM25 และ Vector similarity (สำหรับข้อมูลอาจารย์)"""
        try:
            # Weights for different components
            bm25_weight = 0.4      # 40% for text matching
            vector_weight = 0.4    # 40% for semantic similarity
            bonus_weight = 0.2     # 20% for exact matches and metadata
            
            # Base scores (normalized)
            # Handle both regular BM25 and Thai scores (which can be higher)
            if bm25_score > 15:  # Likely Thai score
                normalized_bm25 = min(bm25_score / 50.0, 1.0)  # Normalize Thai score to 0-1 (max 50)
            else:
                normalized_bm25 = min(bm25_score / 10.0, 1.0)  # Normalize regular BM25 to 0-1 (max 10)
            normalized_vector = vector_score  # Vector score is already 0-1
            
            # Calculate bonus score for exact matches (faculty-specific)
            bonus_score = 0.0
            query_lower = query.lower()
            content_lower = doc.page_content.lower()
            
            # Name match bonus (high priority for faculty search)
            query_words = [word for word in query_lower.split() if len(word) >= 2]
            for word in query_words:
                if word in content_lower:
                    # Higher bonus for exact name matches
                    if any(name_indicator in content_lower for name_indicator in ["ชื่อ", "name", "อาจารย์"]):
                        bonus_score += 0.6  # High bonus for name context
                    elif any(pos_indicator in content_lower for pos_indicator in ["ตำแหน่ง", "position", "หัวหน้า"]):
                        bonus_score += 0.4  # Medium bonus for position context
                    else:
                        bonus_score += 0.2  # Base bonus for other matches
            
            # Faculty-specific bonus terms
            faculty_bonuses = {
                "อาจารย์": 0.3 if "อาจารย์" in content_lower else 0,
                "ผู้ช่วย": 0.2 if "ผู้ช่วย" in content_lower else 0,
                "รอง": 0.2 if "รอง" in content_lower else 0,
                "ศาสตราจารย์": 0.3 if "ศาสตราจารย์" in content_lower else 0,
                "หัวหน้า": 0.2 if "หัวหน้า" in content_lower else 0,
            }
            
            for term, bonus in faculty_bonuses.items():
                if term in query_lower:
                    bonus_score += bonus
            
            # Normalize bonus score
            normalized_bonus = min(bonus_score, 1.0)
            
            # Calculate combined score
            combined_score = (
                bm25_weight * normalized_bm25 +
                vector_weight * normalized_vector +
                bonus_weight * normalized_bonus
            )
            
            return combined_score
            
        except Exception as e:
            print(f"❌ Error calculating hybrid score: {e}")
            return max(bm25_score / 10.0, vector_score)  # Fallback to max of normalized scores
    
    def _advanced_thai_search(self, query: str) -> List[Document]:
        """ค้นหาขั้นสูงด้วย PyThaiNLP - รองรับทั้ง exact และ semantic matching"""
        if not PYTHAINLP_AVAILABLE:
            print("⚠️ PyThaiNLP not available, falling back to basic search")
            return self._exact_thai_keyword_search(query)
        
        try:
            print(f"🇹🇭 Advanced Thai Search with PyThaiNLP: '{query}'")
            
            # 1. Normalize and tokenize query
            normalized_query = normalize(query)
            query_tokens = word_tokenize(normalized_query, engine='newmm')
            
            # 2. POS tagging to get meaningful words
            pos_tags = pos_tag(query_tokens, engine='perceptron')
            stopwords = thai_stopwords()
            
            # 3. Extract meaningful tokens (nouns, verbs, adjectives)
            meaningful_tokens = []
            for word, pos in pos_tags:
                if (pos in ['NOUN', 'VERB', 'ADJ', 'PROPN'] and 
                    word not in stopwords and 
                    len(word.strip()) > 1):
                    meaningful_tokens.append(word)
            
            # If no meaningful tokens, use original tokens
            if not meaningful_tokens:
                meaningful_tokens = [token for token in query_tokens if len(token.strip()) > 1]
            
            print(f"   🔤 Tokenized: {query_tokens}")
            print(f"   🎯 Meaningful: {meaningful_tokens}")
            
            # 4. Search in documents
            all_results = list(self.collection.find({}, limit=100))
            matched_docs = []
            
            for result in all_results:
                content = result.get("content", "")
                metadata = result.get("metadata", {})
                
                # Calculate advanced matching score
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
                    faculty_name = metadata.get('firstname_th', 'Unknown') + " " + metadata.get('lastname_th', '')
                    print(f"  ✅ Match: {faculty_name} (score: {match_score:.2f})")
            
            # Sort by match score
            matched_docs.sort(key=lambda x: x[1], reverse=True)
            
            result_docs = [doc for doc, score in matched_docs[:10]]
            print(f"🎯 Advanced Thai Search: Found {len(result_docs)} matches")
            
            return result_docs
            
        except Exception as e:
            print(f"❌ Error in advanced Thai search: {e}")
            # Fallback to basic search
            return self._exact_thai_keyword_search(query)
    
    def _calculate_thai_match_score(self, meaningful_tokens: List[str], all_tokens: List[str], 
                                   content: str, metadata: dict) -> float:
        """คำนวณคะแนนการ match แบบขั้นสูงสำหรับภาษาไทย"""
        score = 0.0
        
        content_lower = content.lower()
        firstname_th = metadata.get('firstname_th', '').lower()
        lastname_th = metadata.get('lastname_th', '').lower()
        firstname_en = metadata.get('firstname_en', '').lower()
        lastname_en = metadata.get('lastname_en', '').lower()
        
        # 1. Exact token matching in names (highest priority)
        for token in meaningful_tokens:
            token_lower = token.lower()
            if token_lower in firstname_th or token_lower in lastname_th:
                score += 10.0  # Very high score for name match
            elif token_lower in firstname_en or token_lower in lastname_en:
                score += 8.0   # High score for English name match
                
        # 2. Exact token matching in content
        for token in meaningful_tokens:
            if token.lower() in content_lower:
                score += 3.0  # Medium score for content match
        
        # 3. Partial matching (for compound names)
        for token in all_tokens:
            if len(token) > 2:  # Only check longer tokens
                token_lower = token.lower()
                if token_lower in firstname_th or token_lower in lastname_th:
                    score += 5.0
                elif token_lower in content_lower:
                    score += 1.0
        
        # 4. Faculty-specific semantic bonus
        faculty_patterns = {
            'อาจารย์': ['อาจารย์', 'ผู้ช่วย', 'รอง', 'ศาสตราจารย์'],
            'ผู้ช่วย': ['ผู้ช่วย', 'assistant'],
            'รอง': ['รอง', 'associate'],
            'ศาสตราจารย์': ['ศาสตราจารย์', 'professor'],
            'หัวหน้า': ['หัวหน้า', 'head', 'chair']
        }
        
        for token in meaningful_tokens:
            if token in faculty_patterns:
                related_words = faculty_patterns[token]
                for word in related_words:
                    if word in content_lower:
                        score += 2.0
        
        return score
    
    def _exact_thai_keyword_search(self, query: str) -> List[Document]:
        """ค้นหาแบบ exact match สำหรับคำสำคัญภาษาไทย"""
        try:
            # Thai keyword mappings for exact matching
            thai_keyword_mappings = {
                # Faculty patterns
                "อาจารย์": ["อาจารย์", "ผู้ช่วย", "รอง", "ศาสตราจารย์"],
                "ผู้ช่วย": ["ผู้ช่วย", "assistant"],
                "รอง": ["รอง", "associate"],
                "ศาสตราจารย์": ["ศาสตราจารย์", "professor"],
                "หัวหน้า": ["หัวหน้า", "head"],
            }
            
            # Get all documents from collection
            all_results = list(self.collection.find({}, limit=100))
            matched_docs = []
            
            query_lower = query.lower().strip()
            
            # Check if query matches any Thai keyword patterns
            target_keywords = thai_keyword_mappings.get(query_lower, [query_lower])
            
            print(f"🔍 Thai Exact Search: '{query}' → looking for keywords: {target_keywords}")
            
            for result in all_results:
                content = result.get("content", "").lower()
                metadata = result.get("metadata", {})
                
                # Calculate match score
                match_score = 0.0
                matches = []
                
                # Check matches in different fields
                for keyword in target_keywords:
                    if keyword in content:
                        match_score += 3.0  # High score for content match
                        matches.append(f"content:{keyword}")
                
                # If we have matches, create document with score
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
                    faculty_name = metadata.get('firstname_th', 'Unknown') + " " + metadata.get('lastname_th', '')
                    print(f"  ✅ Match: {faculty_name} (score: {match_score:.2f}, matches: {matches})")
            
            # Sort by match score (descending)
            matched_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Return top matches
            result_docs = [doc for doc, score in matched_docs[:10]]
            print(f"🎯 Thai Exact Search: Found {len(result_docs)} matches")
            
            return result_docs
            
        except Exception as e:
            print(f"❌ Error in Thai exact search: {e}")
            return []

    def _fallback_keyword_search(self, query: str) -> List[Document]:
        """Fallback keyword search if BM25 fails - with better Thai name matching"""
        print("🔄 Using fallback keyword search...")
        all_documents = []
        
        try:
            keywords = self._extract_search_keywords(query)
            results = self.collection.find({}, limit=100)
            
            for result in results:
                content = result.get("content", "").lower()
                original_content = result.get("content", "")
                
                matched = False
                for keyword in keywords:
                    keyword_lower = keyword.lower()
                    
                    # Method 1: Direct substring match
                    if keyword_lower in content:
                        matched = True
                        break
                    
                    # Method 2: Character-by-character fuzzy match for Thai names
                    elif len(keyword) >= 3:
                        # Remove spaces from both content and keyword for comparison
                        content_no_space = content.replace(" ", "")
                        keyword_no_space = keyword_lower.replace(" ", "")
                        
                        if keyword_no_space in content_no_space:
                            matched = True
                            print(f"🎯 Fuzzy match: '{keyword}' found in content (no spaces)")
                            break
                        
                        # Method 3: Try partial matches for compound names
                        if len(keyword) > 5:
                            # Split keyword into parts and check if all parts exist
                            keyword_parts = [p for p in keyword_lower.split() if len(p) >= 2]
                            if len(keyword_parts) >= 2:
                                if all(part in content for part in keyword_parts):
                                    matched = True
                                    print(f"🎯 Partial match: All parts of '{keyword}' found")
                                    break
                
                if matched:
                    doc = Document(
                        page_content=original_content,
                        metadata=result.get("metadata", {})
                    )
                    if not any(d.page_content == doc.page_content for d in all_documents):
                        all_documents.append(doc)
            
            print(f"📝 Fallback search: พบ {len(all_documents)} documents")
            return all_documents[:15]
            
        except Exception as e:
            print(f"❌ Error in fallback search: {e}")
            return []
    
    def _extract_search_keywords(self, query: str) -> List[str]:
        """แยกคำสำคัญจากคำค้นหา - ใช้ PyThaiNLP ถ้ามี"""
        import re
        
        keywords = []
        
        if PYTHAINLP_AVAILABLE:
            # ใช้ PyThaiNLP ตัดคำ
            try:
                # ตัดคำด้วย PyThaiNLP
                tokens = word_tokenize(query, engine='newmm')
                
                # ใช้ stopwords จาก PyThaiNLP
                stop_words = thai_stopwords()
                custom_stops = {"ขอ", "ข้อมูล", "อาจารย์", "หา", "ค้นหา", "บอก", "แสดง", "ใคร", "คือ"}
                all_stop_words = stop_words.union(custom_stops)
                
                # กรองคำที่ไม่ใช่ stopwords
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
                print(f"⚠️ PyThaiNLP error: {e}, falling back to basic method")
                PYTHAINLP_AVAILABLE = False
        
        if not PYTHAINLP_AVAILABLE:
            # Fallback: ใช้วิธีเดิม
            stop_words = ["ขอ", "ข้อมูล", "อาจารย์", "หา", "ค้นหา", "บอก", "แสดง", "ใคร", "คือ", "ของ", "ใน", "ที่", "และ", "หรือ"]
            
            # Clean version - remove stop words first
            query_clean = query
            for stop_word in stop_words:
                query_clean = query_clean.replace(stop_word, " ")
            query_clean = re.sub(r'\s+', ' ', query_clean).strip()
            
            if query_clean and len(query_clean) >= 2:
                keywords.append(query_clean)
            
            # Split by spaces
            words_by_space = query.split()
            for word in words_by_space:
                clean_word = word.strip()
                if clean_word not in stop_words and len(clean_word) >= 2:
                    keywords.append(clean_word)
            
            # Extract Thai name patterns
            thai_name_pattern = r'[ก-๙]{3,15}'
            potential_names = re.findall(thai_name_pattern, query)
            for name in potential_names:
                if name not in stop_words and len(name) >= 3 and name not in keywords:
                    keywords.append(name)
        
        # Add the original query as fallback
        if query.strip() not in keywords:
            keywords.append(query.strip())
        
        # Remove duplicates while preserving order
        unique_keywords = []
        for keyword in keywords:
            if keyword not in unique_keywords and len(keyword) >= 2:
                unique_keywords.append(keyword)
        
        print(f"🔤 Debug: Query '{query}' -> Final Keywords: {unique_keywords}")
        return unique_keywords

retriever = AllPeopleRetriever(collection, embedding)

# ✅ สร้าง Prompt - ปรับปรุงเพื่อให้เหมาะกับข้อมูลอาจารย์ (กระชับ มีประโยคนำ จัดรูปแบบเป็นระเบียบ)
PROMPT = PromptTemplate.from_template("""
บริบทต่อไปนี้คือข้อมูลเกี่ยวกับอาจารย์ในคณะวิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น

**คำแนะนำ:**
- ตอบเฉพาะสิ่งที่ผู้ใช้ถาม ไม่ยัดข้อมูลที่ไม่เกี่ยวข้อง
- มีประโยคนำสั้นๆ ได้ (เช่น "อีเมลอาจารย์... คือ ...") ไม่ห้วนจนเหลือแค่ค่าอย่างเดียว
- **จัดรูปแบบให้อ่านง่าย:** ถ้ามีหลายรายการหรือหลายฟิลด์ ให้แยกบรรทัด ใช้ bullet (- หรือ •) นำหน้า แต่ละข้อความบรรทัดเดียว ไม่รวมเป็นย่อหน้าเดียวยาวๆ
- หากถามรายชื่อหลายคน แสดงเป็นรายการ bullet สั้นๆ
- หากไม่มีข้อมูลที่ตรงกับคำถาม ให้ตอบว่า "ขอโทษ ฉันไม่พบข้อมูลในระบบ"

---------------------
{context}
---------------------
คำถาม: {question}
คำตอบ (กระชับ มีประโยคนำเล็กน้อย จัดรูปแบบเป็นบรรทัด/bullet ให้อ่านง่าย):
""")

# ✅ โหลด Chat Model (KKU IntelSphere first, then OpenRouter fallback)
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("CHATBOT_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
if api_key and base_url:
    try:
        llm = ChatOpenAI(model=model_name, temperature=0, openai_api_key=api_key, openai_api_base=base_url)
        print("✅ LLM initialized (KKU IntelSphere)")
    except Exception as e:
        print(f"❌ Error initializing LLM: {e}")
        llm = None
else:
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_api_key:
        try:
            llm = ChatOpenAI(
                model="openai/gpt-4o-mini",
                temperature=0,
                openai_api_key=openrouter_api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                default_headers={"HTTP-Referer": "https://github.com/your-repo", "X-Title": "Faculty RAG Chatbot"}
            )
            print("✅ OpenRouter LLM initialized")
        except Exception as e:
            print(f"❌ Error initializing LLM: {e}")
            llm = None
    else:
        llm = None
        print("⚠️ Warning: No LLM API key (set OPENAI_API_KEY+OPENAI_BASE_URL for KKU, or OPENROUTER_API_KEY), LLM responses will not work")

# ✅ สร้าง Manual QA Function
def manual_qa_chain(question: str) -> str:
    """
    Manual QA chain ที่ควบคุมการทำงานได้ทุกขั้นตอน - AllPeople Version with astrapy
    """
    try:
        print(f"🔍 กำลังค้นหาข้อมูลสำหรับคำถาม: {question}")
        print("🌐 ใช้ AstraDB Cloud Vector Database (astrapy) - AllPeople Collection")
        print(f"📚 Collection: allpeople_embedding")
        
        # ขั้นตอน 1: ดึงข้อมูลจาก retriever
        retrieved_docs = retriever.get_relevant_documents(question)
        
        if not retrieved_docs:
            return "ขอโทษ ฉันไม่พบข้อมูลในระบบ"
        
        print(f"📚 พบข้อมูล {len(retrieved_docs)} รายการจาก AstraDB")
        
        # ขั้นตอน 2: แสดงผลลัพธ์ทั้งหมดก่อน
        print("\n" + "="*60)
        print("📋 ผลการค้นหาทั้งหมดจาก AstraDB (AllPeople Collection):")
        print("="*60)
        
        for i, doc in enumerate(retrieved_docs, 1):
            combined_score = doc.metadata.get('combined_score', 0.0)
            bm25_score = doc.metadata.get('bm25_score', 0.0)
            vector_score = doc.metadata.get('vector_score', 0.0)
            
            print(f"\n🔸 ผลลัพธ์ที่ {i}:")
            if combined_score > 0:
                print(f"   📊 คะแนนความเกี่ยวข้อง: {combined_score:.4f} (BM25: {bm25_score:.4f}, Vector: {vector_score:.4f})")
            print("-" * 40)
            print(doc.page_content.strip())
            print("-" * 40)
        
        print("\n" + "="*60)
        
        # ขั้นตอน 3: ใช้ข้อมูลทั้งหมดที่ค้นหาได้
        selected_docs = retrieved_docs  # ใช้ทั้งหมด
        
        print(f"🎯 ใช้ข้อมูลทั้งหมด {len(selected_docs)} รายการ สำหรับการตอบคำถาม")
        print(f"💡 เหตุผล: ใช้ข้อมูลทั้งหมดเพื่อให้ครอบคลุมและแม่นยำที่สุด")
        print("="*60)
        
        # จัดเตรียม context สำหรับ LLM
        context_parts = []
        print("\n📝 CONTEXT สำหรับ LLM:")
        print("-" * 40)
        
        for i, doc in enumerate(selected_docs, 1):
            print(f"📄 Context {i}: {doc.page_content.strip()[:100]}...")
            context_parts.append(f"ข้อมูลที่ {i}:\n{doc.page_content}\n")
        
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
            response = llm.invoke(formatted_prompt)  # Use invoke instead of predict
            return response.content.strip()
        except Exception as llm_error:
            print(f"❌ LLM Error: {llm_error}")
            # Return search results directly if LLM fails
            return f"พบข้อมูลที่เกี่ยวข้อง {len(retrieved_docs)} รายการ แต่ไม่สามารถประมวลผลคำตอบได้"
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        return "ขอโทษ เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่อีกครั้ง"

# ✅ เริ่มถาม
if __name__ == "__main__":
    # Fix encoding for Windows terminal (only when running as main script)
    if sys.platform == "win32":
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    
    print("🎓 ระบบถาม-ตอบ ข้อมูลคณะคอมพิวเตอร์ มข. (AllPeople Version with astrapy)")
    print("🌐 ใช้ AstraDB Cloud Vector Database - AllPeople Collection")
    print(f"📚 Collection: allpeople_embedding")
    print("พิมพ์ 'exit' เพื่อออก\n")

    while True:
        question = input("❓ ถามมาเลย: ")
        if question.lower() == "exit":
            break

        # ใช้ manual QA chain แทน
        result = manual_qa_chain(question)
        print("🤖 คำตอบ:", result)
        print("-" * 50)
