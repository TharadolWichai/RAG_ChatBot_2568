# main_contact.py - AstraDB Version สำหรับข้อมูลติดต่อ (Enhanced with PyThaiNLP)
import os
import sys
import uuid
from typing import List, Any

from astrapy import DataAPIClient
from dotenv import load_dotenv
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

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
COLLECTION_NAME = "services_embedding"  # รวมกับ links และ students

if not ASTRA_TOKEN or not ASTRA_ENDPOINT:
    print("❌ Missing AstraDB credentials in .env")
    exit(1)

client = DataAPIClient(token=ASTRA_TOKEN)
database = client.get_database_by_api_endpoint(ASTRA_ENDPOINT)

try:
    collection = database.get_collection(COLLECTION_NAME)
    print(f"✅ Connected to collection: {COLLECTION_NAME} (category: contact)")
except Exception as e:
    print(f"❌ Error accessing collection: {e}")
    exit(1)

# -------------------------------
# Custom Retriever
# -------------------------------
class ContactInfoRetriever(BaseRetriever):
    collection: Any = None
    embedding: Any = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, collection, embedding):
        super().__init__()
        self.collection = collection
        self.embedding = embedding
        self._bm25_retriever = None
        self._documents_cache = None
        self._thai_bm25 = None
        self._thai_bm25_docs = None

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        safe_print(f"🔍 Debug: กำลังค้นหาข้อมูลติดต่อด้วย query: '{query}'")
        
        # ถ้าต้องการข้อมูลทั้งหมด
        if any(word in query.lower() for word in ["ทั้งหมด", "ทุกอัน", "all", "ข้อมูลติดต่อ", "contact"]):
            safe_print("🎯 ตรวจพบคำขอข้อมูลติดต่อทั้งหมด - ใช้การค้นหาแบบครอบคลุม")
            return self._get_comprehensive_search()
        
        # Try multiple search strategies
        print("🔍 กำลังค้นหาด้วย AstraDB hybrid search...")
        
        # Strategy 0: Thai Exact Search (for Thai queries)
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
                    print(f"➕ Thai {search_type_display} #{i+1}: Contact info (Thai: {thai_score:.2f}, Combined: {doc.metadata['combined_score']:.4f})")
        
        # Add text search results
        for i, doc in enumerate(text_results):
            if doc.page_content not in seen_content:
                bm25_score = doc.metadata.get("bm25_score", 0.0)
                doc.metadata["combined_score"] = self._calculate_hybrid_score(doc, bm25_score, 0.0, query)
                candidate_docs.append(doc)
                seen_content.add(doc.page_content)
                print(f"➕ Text Search #{i+1}: Contact info (BM25: {bm25_score:.4f})")
        
        # Add vector search results
        for i, doc in enumerate(vector_results):
            if doc.page_content not in seen_content:
                vector_score = doc.metadata.get("vector_score", 0.0)
                doc.metadata["combined_score"] = self._calculate_hybrid_score(doc, 0.0, vector_score, query)
                candidate_docs.append(doc)
                seen_content.add(doc.page_content)
                print(f"➕ Vector Search #{i+1}: Contact info (Vector: {vector_score:.4f})")
            else:
                # Update existing document with vector score
                for existing_doc in candidate_docs:
                    if existing_doc.page_content == doc.page_content:
                        vector_score = doc.metadata.get("vector_score", 0.0)
                        bm25_score = existing_doc.metadata.get("bm25_score", 0.0)
                        existing_doc.metadata["vector_score"] = vector_score
                        existing_doc.metadata["combined_score"] = self._calculate_hybrid_score(existing_doc, bm25_score, vector_score, query)
                        print(f"🔄 อัปเดตคะแนน: Contact info (BM25: {bm25_score:.4f}, Vector: {vector_score:.4f})")
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
            
            # Show search type icon
            type_icon = "🇹🇭" if "thai" in search_type else "📝" if "bm25" in search_type else "🧠" if "vector" in search_type else "❓"
            
            print(f"#{i}: {type_icon} Contact Information")
            print(f"    🎯 Combined: {combined_score:.4f} | 📝 BM25/Thai: {bm25_score:.4f} | 🧠 Vector: {vector_score:.4f}")
            print(f"    🔍 Search Type: {search_type}")
            print("-" * 40)
        
        print(f"📊 สรุป: Text={len(text_results)}, Vector={len(vector_results)}, รวม={len(all_documents)} (unique)")
        
        return all_documents[:10]  # Return top 10 results

    def _get_comprehensive_search(self) -> List[Document]:
        print("🚀 Comprehensive search in contact info collection...")
        docs = []
        try:
            results = self.collection.find({"metadata.category": "contact"}, limit=50)
            for r in results:
                docs.append(Document(page_content=r.get("content", ""), metadata=r.get("metadata", {})))
            print(f"📊 Found {len(docs)} contact documents")
        except Exception as e:
            print(f"❌ Error: {e}")
        return docs

    def _vector_search(self, query: str) -> List[Document]:
        """ค้นหาแบบ vector search จาก collection พร้อมคะแนนความใกล้เคียง"""
        all_documents = []
        
        try:
            print(f"🧠 Vector Search: กำลังสร้าง embedding สำหรับ query: '{query}'")
            
            # Generate query embedding
            query_vector = self.embedding.embed_query(query)
            print(f"📊 Vector Search: สร้าง embedding แล้ว (dimension: {len(query_vector)})")
            
            # Perform vector search with similarity scores (filter เฉพาะ contact)
            results = self.collection.find(
                {"metadata.category": "contact"},
                sort={"$vector": query_vector},
                limit=10,  # Top 10 semantic matches
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
                    print(f"   #{i+1}: Contact info (score: {score:.4f})")
            
            return all_documents
            
        except Exception as e:
            print(f"❌ Error in vector search: {e}")
            return []

    def _ensure_bm25_initialized(self):
        """Initialize BM25 retriever with Thai tokenization support"""
        if self._bm25_retriever is None:
            print("🔧 Initializing Enhanced BM25 retriever with Thai support...")
            try:
                # Get all documents from collection for BM25 (filter เฉพาะ contact)
                results = self.collection.find({"metadata.category": "contact"}, limit=100)
                documents = []
                
                for result in results:
                    doc = Document(
                        page_content=result.get("content", ""),
                        metadata=result.get("metadata", {})
                    )
                    documents.append(doc)
                
                self._documents_cache = documents
                print(f"📚 Loaded {len(documents)} documents for BM25")
                
                # Create Enhanced BM25 with Thai tokenization
                if documents and PYTHAINLP_AVAILABLE:
                    self._create_thai_bm25(documents)
                    print("✅ Enhanced Thai BM25 initialized successfully")
                elif documents:
                    # Fallback to standard BM25
                    self._bm25_retriever = BM25Retriever.from_documents(documents)
                    self._bm25_retriever.k = 10
                    print("✅ Standard BM25 initialized successfully (PyThaiNLP not available)")
                else:
                    print("⚠️ No documents found for BM25 initialization")
                    
            except Exception as e:
                print(f"❌ Error initializing BM25: {e}")
                self._bm25_retriever = None
    
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
            self._thai_bm25 = BM25Okapi(tokenized_docs)
            self._thai_bm25_docs = documents  # Keep reference to original docs
            
            print(f"🇹🇭 Thai BM25 created with {len(tokenized_docs)} tokenized documents")
            
        except Exception as e:
            print(f"❌ Error creating Thai BM25: {e}")
            # Fallback to standard BM25
            self._bm25_retriever = BM25Retriever.from_documents(documents)
            self._bm25_retriever.k = 10

    def _preprocess_query_for_bm25(self, query: str) -> str:
        return query.replace("ขอข้อมูล", "").replace("ติดต่อ", "").strip()

    def _text_search(self, query: str) -> List[Document]:
        """Enhanced BM25 text search with Thai tokenization support"""
        try:
            # Ensure BM25 is initialized
            self._ensure_bm25_initialized()
            
            # Check if Thai BM25 is available
            if hasattr(self, '_thai_bm25') and PYTHAINLP_AVAILABLE:
                return self._thai_bm25_search(query)
            elif self._bm25_retriever is not None:
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
            bm25_scores = self._thai_bm25.get_scores(filtered_tokens)
            
            # Create results with scores
            results = []
            for i, score in enumerate(bm25_scores):
                if score > 0:  # Only include documents with positive scores
                    doc = self._thai_bm25_docs[i]
                    
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
                    print(f"   #{i+1}: Contact info (Thai BM25: {score:.4f})")
            
            return results[:10]  # Return top 10
            
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
                query.replace("ขอ", "").replace("ข้อมูล", "").replace("ติดต่อ", "").strip(),  # Clean version
            ]
            
            # Add contact-specific variations
            if "ติดต่อ" in query:
                query_variations.extend(["โทรศัพท์", "อีเมล", "ที่อยู่"])
            if "เบอร์" in query or "โทร" in query:
                query_variations.extend(["โทรศัพท์", "043-"])
            
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
                    variant_results = self._bm25_retriever.get_relevant_documents(q_variant)
                    
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
                    print(f"   #{i+1}: Contact info (BM25: {score:.4f})")
            
            return all_bm25_results[:10]  # Limit to top 10
            
        except Exception as e:
            print(f"❌ Error in standard BM25 search: {e}")
            return self._fallback_keyword_search(query)

    def _calculate_bm25_scores(self, documents: List[Document], query: str) -> List[tuple]:
        """คำนวณคะแนน BM25 สำหรับ documents"""
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
                
                # Boost score for exact matches in metadata
                contact_type = doc.metadata.get('type', '').lower()
                
                for term in query_terms:
                    if term in content_lower:
                        score += 1.0  # Boost for content match
                    if "contact" in contact_type or "ติดต่อ" in contact_type:
                        score += 0.5  # Boost for contact-related content
                
                scored_docs.append((doc, score))
            
            return scored_docs
            
        except Exception as e:
            print(f"❌ Error calculating BM25 scores: {e}")
            return [(doc, 0.0) for doc in documents]
    
    def _calculate_hybrid_score(self, doc: Document, bm25_score: float, vector_score: float, query: str) -> float:
        """คำนวณคะแนนรวมจาก BM25 และ Vector similarity"""
        try:
            # Weights for different components
            bm25_weight = 0.4      # 40% for text matching
            vector_weight = 0.4    # 40% for semantic similarity
            bonus_weight = 0.2     # 20% for exact matches and metadata
            
            # Base scores (normalized)
            if bm25_score > 15:  # Likely Thai score
                normalized_bm25 = min(bm25_score / 50.0, 1.0)  # Normalize Thai score to 0-1 (max 50)
            else:
                normalized_bm25 = min(bm25_score / 10.0, 1.0)  # Normalize regular BM25 to 0-1 (max 10)
            normalized_vector = vector_score  # Vector score is already 0-1
            
            # Calculate bonus score for exact matches
            bonus_score = 0.0
            query_lower = query.lower()
            content_lower = doc.page_content.lower()
            doc_type = doc.metadata.get('type', '').lower()
            
            # Contact-specific bonuses
            contact_bonuses = {
                "ติดต่อ": 0.5 if "ติดต่อ" in content_lower else 0,
                "โทรศัพท์": 0.5 if ("โทรศัพท์" in content_lower or "043-" in content_lower) else 0,
                "อีเมล": 0.5 if "@" in content_lower else 0,
                "ที่อยู่": 0.3 if "ที่อยู่" in content_lower else 0,
                "hotline": 0.3 if ("hotline" in content_lower or "089-" in content_lower) else 0,
            }
            
            for contact_term, bonus in contact_bonuses.items():
                if contact_term in query_lower:
                    bonus_score += bonus
            
            # Type-specific bonus
            if "contact" in doc_type or "ติดต่อ" in doc_type:
                bonus_score += 0.3
            
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
            
            # 4. Search in documents (filter เฉพาะ contact)
            all_results = list(self.collection.find({"metadata.category": "contact"}, limit=100))
            matched_docs = []
            
            for result in all_results:
                content = result.get("content", "")
                metadata = result.get("metadata", {})
                doc_type = metadata.get("type", "")
                
                # Calculate advanced matching score
                match_score = self._calculate_thai_match_score(
                    meaningful_tokens, query_tokens, content, doc_type
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
                    print(f"  ✅ Match: Contact info (score: {match_score:.2f})")
            
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
                                   content: str, doc_type: str) -> float:
        """คำนวณคะแนนการ match แบบขั้นสูงสำหรับภาษาไทย"""
        score = 0.0
        
        content_lower = content.lower()
        doc_type_lower = doc_type.lower()
        
        # 1. Exact token matching in content (highest priority)
        for token in meaningful_tokens:
            if token.lower() in content_lower:
                score += 5.0  # High score for content match
        
        # 2. Partial matching (for compound words)
        for token in all_tokens:
            if len(token) > 2:  # Only check longer tokens
                if token.lower() in content_lower:
                    score += 2.0
        
        # 3. Contact-specific semantic bonus
        contact_patterns = {
            'ติดต่อ': ['ติดต่อ', 'contact'],
            'โทรศัพท์': ['โทรศัพท์', 'โทร', 'phone', '043-'],
            'อีเมล': ['อีเมล', 'email', '@'],
            'ที่อยู่': ['ที่อยู่', 'address', 'อาคาร'],
            'hotline': ['hotline', 'hot line', '089-']
        }
        
        for token in meaningful_tokens:
            if token in contact_patterns:
                related_words = contact_patterns[token]
                for word in related_words:
                    if word in content_lower:
                        score += 3.0
        
        # 4. Document type bonus
        if "contact" in doc_type_lower or "ติดต่อ" in doc_type_lower:
            score += 2.0
        
        return score
    
    def _exact_thai_keyword_search(self, query: str) -> List[Document]:
        """ค้นหาแบบ exact match สำหรับคำสำคัญภาษาไทย"""
        try:
            # Thai keyword mappings for exact matching
            thai_keyword_mappings = {
                # Contact patterns
                "ติดต่อ": ["ติดต่อ", "โทรศัพท์", "อีเมล", "ที่อยู่"],
                "โทรศัพท์": ["โทรศัพท์", "โทร", "043-", "phone"],
                "เบอร์": ["โทรศัพท์", "โทร", "043-", "089-"],
                "อีเมล": ["อีเมล", "email", "@"],
                "ที่อยู่": ["ที่อยู่", "address", "อาคาร"],
                "hotline": ["hotline", "hot line", "089-"],
            }
            
            # Get all documents from collection (filter เฉพาะ contact)
            all_results = list(self.collection.find({"metadata.category": "contact"}, limit=100))
            matched_docs = []
            
            query_lower = query.lower().strip()
            
            # Check if query matches any Thai keyword patterns
            target_keywords = thai_keyword_mappings.get(query_lower, [query_lower])
            
            print(f"🔍 Thai Exact Search: '{query}' → looking for keywords: {target_keywords}")
            
            for result in all_results:
                content = result.get("content", "").lower()
                metadata = result.get("metadata", {})
                doc_type = metadata.get("type", "").lower()
                
                # Calculate match score
                match_score = 0.0
                matches = []
                
                # Check matches in different fields
                for keyword in target_keywords:
                    if keyword in content:
                        match_score += 3.0  # High score for content match
                        matches.append(f"content:{keyword}")
                    if keyword in doc_type:
                        match_score += 2.0  # Medium score for type match
                        matches.append(f"type:{keyword}")
                
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
                    print(f"  ✅ Match: Contact info (score: {match_score:.2f}, matches: {matches})")
            
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
        """Fallback keyword search if BM25 fails"""
        print("🔄 Using fallback keyword search...")
        all_documents = []
        
        try:
            keywords = self._extract_search_keywords(query)
            results = self.collection.find({"metadata.category": "contact"}, limit=50)
            
            for result in results:
                content = result.get("content", "").lower()
                original_content = result.get("content", "")
                doc_type = result.get("metadata", {}).get("type", "").lower()
                
                matched = False
                for keyword in keywords:
                    keyword_lower = keyword.lower()
                    
                    # Check in content or document type
                    if keyword_lower in content or keyword_lower in doc_type:
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
        model="openai/gpt-4o-mini",
        temperature=0,
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1"
    )

# -------------------------------
# Manual QA Chain
# -------------------------------
def manual_qa_chain(question: str) -> str:
    """
    Manual QA chain สำหรับตอบคำถามเกี่ยวกับข้อมูลติดต่อ - Contact Version with astrapy
    """
    try:
        print(f"🔍 กำลังค้นหาข้อมูลติดต่อสำหรับคำถาม: {question}")
        print("🌐 ใช้ AstraDB Cloud Vector Database (astrapy) - Contact Collection")
        print(f"📚 Collection: {COLLECTION_NAME}")
        
        # ขั้นตอน 1: ดึงข้อมูลจาก retriever
        retrieved_docs = retriever.get_relevant_documents(question, run_manager=None)
        
        if not retrieved_docs:
            return "ขอโทษ ฉันไม่พบข้อมูลในระบบ"
        
        print(f"📚 พบข้อมูลติดต่อ {len(retrieved_docs)} รายการจาก AstraDB")
        
        # ขั้นตอน 2: แสดงผลลัพธ์ทั้งหมดก่อน
        print("\n" + "="*60)
        print("📋 ผลการค้นหาข้อมูลติดต่อทั้งหมดจาก AstraDB (Contact Collection):")
        print("="*60)
        
        for i, doc in enumerate(retrieved_docs, 1):
            combined_score = doc.metadata.get('combined_score', 0.0)
            bm25_score = doc.metadata.get('bm25_score', 0.0)
            vector_score = doc.metadata.get('vector_score', 0.0)
            doc_type = doc.metadata.get('type', 'Unknown')
            
            print(f"\n🔸 ข้อมูลติดต่อที่ {i}:")
            print(f"   ประเภท: {doc_type}")
            print(f"   เนื้อหา: {doc.page_content[:100]}{'...' if len(doc.page_content) > 100 else ''}")
            if combined_score > 0:
                print(f"   📊 คะแนนความเกี่ยวข้อง: {combined_score:.4f} (BM25: {bm25_score:.4f}, Vector: {vector_score:.4f})")
            print("-" * 40)
        
        print("\n" + "="*60)
        
        # ขั้นตอน 3: ใช้ข้อมูลทั้งหมดที่ค้นหาได้
        selected_docs = retrieved_docs  # ใช้ทั้งหมด
        
        print(f"🎯 ใช้ข้อมูลติดต่อทั้งหมด {len(selected_docs)} รายการ สำหรับการตอบคำถาม")
        print("="*60)
        
        # จัดเตรียม context สำหรับ LLM
        context_parts = []
        print("\n📝 CONTEXT สำหรับ LLM:")
        print("-" * 40)
        
        for i, doc in enumerate(selected_docs, 1):
            doc_type = doc.metadata.get('type', 'Unknown')
            print(f"📄 Context {i}: {doc_type} - {doc.page_content[:50]}{'...' if len(doc.page_content) > 50 else ''}")
            context_parts.append(f"ข้อมูลติดต่อที่ {i}:\n{doc.page_content}\n")
        
        context = "\n".join(context_parts)
        print("\n" + "="*60)
        
        # ขั้นตอน 4: สร้าง prompt
        formatted_prompt = PROMPT.format(
            context=context,
            question=question
        )
        
        print("💭 กำลังประมวลผลคำตอบ...")
        
        # ขั้นตอน 5: เรียกใช้ LLM
        if llm is None:
            return "⚠️ ไม่สามารถตอบคำถามได้ เนื่องจากไม่มี API key สำหรับ LLM"
        
        try:
            response = llm.invoke(formatted_prompt)
            return response.content.strip()
        except Exception as llm_error:
            print(f"❌ LLM Error: {llm_error}")
            # Return search results directly if LLM fails
            result_text = f"พบข้อมูลติดต่อที่เกี่ยวข้อง {len(retrieved_docs)} รายการ:\n\n"
            for i, doc in enumerate(retrieved_docs, 1):
                doc_type = doc.metadata.get('type', 'Unknown')
                result_text += f"{i}. {doc_type}\n   {doc.page_content[:200]}{'...' if len(doc.page_content) > 200 else ''}\n\n"
            return result_text
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        return "ขอโทษ เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่อีกครั้ง"

# -------------------------------
# Interactive Loop
# -------------------------------
if __name__ == "__main__":
    # Fix encoding for Windows terminal (only when running as main script)
    if sys.platform == "win32":
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    
    print("📞 ระบบถาม-ตอบข้อมูลติดต่อคณะวิทยาการคอมพิวเตอร์")
    print("พิมพ์ 'exit' เพื่อออก\n")
    while True:
        question = input("❓ ถามข้อมูลติดต่อ: ")
        if question.lower() == "exit":
            break
        answer = manual_qa_chain(question)
        print("🤖 คำตอบ:", answer)
        print("-"*50)
