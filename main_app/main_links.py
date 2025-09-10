# main_links.py - AstraDB Version สำหรับ links_data.py โดยเฉพาะ (Enhanced with PyThaiNLP)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
import os
from dotenv import load_dotenv
from typing import List 
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

# Get single collection for links data
try:
    collection = database.get_collection("links_embedding")
    print(f"✅ Connected to collection: links_embedding")
except Exception as e:
    print(f"❌ Error accessing collection: {e}")
    exit(1)

# ✅ สร้าง Custom Retriever สำหรับ AstraDB (Links Collection)
class LinksRetriever(BaseRetriever):
    def __init__(self, collection, embedding):
        super().__init__()
        self._collection = collection
        self._embedding = embedding
        self._bm25_retriever = None  # Will be initialized when first used
        self._documents_cache = None  # Cache documents for BM25
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        print(f"🔍 Debug: กำลังค้นหาลิงก์ด้วย query: '{query}'")
        
        # ถ้าต้องการข้อมูลลิงก์ทั้งหมด
        if any(word in query.lower() for word in ["ทั้งหมด", "ทุกอัน", "all", "รายการ", "ลิงก์ทั้งหมด"]):
            print("🎯 ตรวจพบคำขอลิงก์ทั้งหมด - ใช้การค้นหาแบบครอบคลุม")
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
                    
                    # Give Thai matches very high combined score
                    if thai_score > 10:  # Very high score
                        doc.metadata["combined_score"] = 0.9 + min(thai_score * 0.01, 0.1)  # 0.9-1.0
                    else:
                        doc.metadata["combined_score"] = 0.7 + (thai_score * 0.05)  # 0.7-0.95
                    
                    doc.metadata["bm25_score"] = 0.0
                    doc.metadata["vector_score"] = 0.0
                    candidate_docs.append(doc)
                    seen_content.add(doc.page_content)
                    
                    search_type_display = "🇹🇭 Advanced" if search_type == "thai_advanced" else "🎯 Exact"
                    print(f"➕ Thai {search_type_display} #{i+1}: {doc.metadata.get('link_text', 'Unknown')} (Thai: {thai_score:.2f}, Combined: {doc.metadata['combined_score']:.4f})")
        
        # Add text search results
        for i, doc in enumerate(text_results):
            if doc.page_content not in seen_content:
                bm25_score = doc.metadata.get("bm25_score", 0.0)
                doc.metadata["combined_score"] = self._calculate_hybrid_score(doc, bm25_score, 0.0, query)
                candidate_docs.append(doc)
                seen_content.add(doc.page_content)
                print(f"➕ Text Search #{i+1}: {doc.metadata.get('link_text', 'Unknown')} (BM25: {bm25_score:.4f})")
        
        # Add vector search results
        for i, doc in enumerate(vector_results):
            if doc.page_content not in seen_content:
                vector_score = doc.metadata.get("vector_score", 0.0)
                doc.metadata["combined_score"] = self._calculate_hybrid_score(doc, 0.0, vector_score, query)
                candidate_docs.append(doc)
                seen_content.add(doc.page_content)
                print(f"➕ Vector Search #{i+1}: {doc.metadata.get('link_text', 'Unknown')} (Vector: {vector_score:.4f})")
            else:
                # Update existing document with vector score
                for existing_doc in candidate_docs:
                    if existing_doc.page_content == doc.page_content:
                        vector_score = doc.metadata.get("vector_score", 0.0)
                        bm25_score = existing_doc.metadata.get("bm25_score", 0.0)
                        existing_doc.metadata["vector_score"] = vector_score
                        existing_doc.metadata["combined_score"] = self._calculate_hybrid_score(existing_doc, bm25_score, vector_score, query)
                        print(f"🔄 อัปเดตคะแนน: {existing_doc.metadata.get('link_text', 'Unknown')} (BM25: {bm25_score:.4f}, Vector: {vector_score:.4f})")
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
            link_text = doc.metadata.get('link_text', 'Unknown')
            
            print(f"#{i}: {link_text}")
            print(f"    🎯 Combined: {combined_score:.4f} | 📝 BM25: {bm25_score:.4f} | 🧠 Vector: {vector_score:.4f}")
            print("-" * 40)
        
        print(f"📊 สรุป: Text={len(text_results)}, Vector={len(vector_results)}, รวม={len(all_documents)} (unique)")
        
        return all_documents[:10]  # Return top 10 results
    
    def _get_comprehensive_search(self) -> List[Document]:
        """ค้นหาลิงก์แบบครอบคลุมทั้งหมดจาก collection"""
        print("🚀 เริ่มการค้นหาลิงก์แบบครอบคลุมจาก links_embedding collection...")
        
        all_documents = []
        
        try:
            print("🔍 ค้นหาจาก links_embedding collection...")
            results = self._collection.find({}, limit=50)  # Get up to 50 links
            
            for result in results:
                doc = Document(
                    page_content=result.get("content", ""),
                    metadata=result.get("metadata", {})
                )
                all_documents.append(doc)
            
            print(f"📊 จาก links_embedding: {len(all_documents)} รายการ")
            
        except Exception as e:
            print(f"❌ Error in comprehensive search: {e}")
        
        print(f"🎯 พบลิงก์ครอบคลุมรวม: {len(all_documents)} รายการ")
        return all_documents
    
    def _vector_search(self, query: str) -> List[Document]:
        """ค้นหาแบบ vector search จาก collection พร้อมคะแนนความใกล้เคียง"""
        all_documents = []
        
        try:
            print(f"🧠 Vector Search: กำลังสร้าง embedding สำหรับ query: '{query}'")
            
            # Generate query embedding
            query_vector = self._embedding.embed_query(query)
            print(f"📊 Vector Search: สร้าง embedding แล้ว (dimension: {len(query_vector)})")
            
            # Perform vector search with similarity scores
            results = self._collection.find(
                {},
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
                    link_text = doc.metadata.get('link_text', 'Unknown')
                    print(f"   #{i+1}: {link_text} (score: {score:.4f})")
            
            return all_documents
            
        except Exception as e:
            print(f"❌ Error in vector search: {e}")
            return []

    def _ensure_bm25_initialized(self):
        """Initialize BM25 retriever with Thai tokenization support"""
        if self._bm25_retriever is None:
            print("🔧 Initializing Enhanced BM25 retriever with Thai support...")
            try:
                # Get all documents from collection for BM25
                results = self._collection.find({}, limit=100)
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
        """ประมวลผลคำถามก่อนส่งให้ BM25"""
        import re
        
        # Remove common prefixes but keep the core keywords
        processed = query.replace("ขอ", "").replace("ลิงก์", "").replace("ลิงค์", "").replace("ระบบ", "").strip()
        
        # แยกคำสำคัญสำหรับการค้นหาที่ดีขึ้น
        if "จองห้องประชุม" in query:
            processed = "จองห้อง ห้องประชุม"
        elif "จองห้องแล็บ" in query:
            processed = "จองห้อง ห้องแล็บ ปฏิบัติการ"
        elif "จองห้อง" in query:
            processed = "จองห้อง"
        
        print(f"🔤 Query preprocessing: '{query}' → '{processed}'")
        return processed if processed else query

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
                    link_text = doc.metadata.get('link_text', 'Unknown')
                    print(f"   #{i+1}: {link_text} (Thai BM25: {score:.4f})")
            
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
                query.replace("ขอ", "").replace("ลิงก์", "").replace("ลิงค์", "").strip(),  # Clean version
            ]
            
            # Add word-by-word variations for compound queries
            if "จองห้องประชุม" in query:
                query_variations.extend(["จองห้อง", "ห้องประชุม", "จอง ห้องประชุม"])
            if "จองห้องแล็บ" in query or "จองแล็บ" in query:
                query_variations.extend(["จองห้อง", "ห้องแล็บ", "ปฏิบัติการ", "จอง ปฏิบัติการ"])
            
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
                    link_text = doc.metadata.get('link_text', 'Unknown')
                    print(f"   #{i+1}: {link_text} (BM25: {score:.4f})")
            
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
                link_text = doc.metadata.get('link_text', '').lower()
                keywords = doc.metadata.get('keywords', [])
                
                for term in query_terms:
                    if term in link_text:
                        score += 2.0  # Boost for title match
                    for keyword in keywords:
                        if term in keyword.lower():
                            score += 1.0  # Boost for keyword match
                
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
            normalized_bm25 = min(bm25_score / 10.0, 1.0)  # Normalize BM25 to 0-1
            normalized_vector = vector_score  # Vector score is already 0-1
            
            # Calculate bonus score for exact matches
            bonus_score = 0.0
            query_lower = query.lower()
            link_text = doc.metadata.get('link_text', '').lower()
            keywords = doc.metadata.get('keywords', [])
            
            # Exact title match bonus
            if any(word in link_text for word in query_lower.split() if len(word) >= 2):
                bonus_score += 0.5
            
            # Keyword match bonus
            for keyword in keywords:
                if any(word in keyword.lower() for word in query_lower.split() if len(word) >= 2):
                    bonus_score += 0.3
                    break
            
            # Service type bonus (for specific service queries) - Enhanced
            service_bonuses = {
                "จอง": 0.5 if "จอง" in link_text else 0,  # Increased bonus for booking
                "ห้องประชุม": 0.5 if "ห้องประชุม" in link_text else 0,  # Increased bonus for meeting room
                "ห้องแล็บ": 0.3 if ("ห้องแล็บ" in link_text or "ปฏิบัติการ" in link_text) else 0,
                "อัปโหลด": 0.3 if ("อัปโหลด" in link_text or "upload" in link_text.lower()) else 0,
                "reservation": 0.4 if "reservation" in link_text.lower() else 0,  # Added reservation bonus
            }
            
            for service_term, bonus in service_bonuses.items():
                if service_term in query_lower:
                    bonus_score += bonus
            
            # Special exact match bonus for common queries
            exact_match_bonuses = {
                "จองห้องประชุม": 1.0 if ("จอง" in query_lower and "ห้องประชุม" in query_lower and "จอง" in link_text and "ห้องประชุม" in link_text) else 0,
                "จองห้องแล็บ": 1.0 if ("จอง" in query_lower and ("ห้องแล็บ" in query_lower or "แล็บ" in query_lower) and "จอง" in link_text and "ปฏิบัติการ" in link_text) else 0,
            }
            
            for exact_term, bonus in exact_match_bonuses.items():
                if bonus > 0:
                    bonus_score += bonus
                    print(f"🎯 Exact match bonus for '{exact_term}': +{bonus}")
            
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
            all_results = list(self._collection.find({}, limit=100))
            matched_docs = []
            
            for result in all_results:
                content = result.get("content", "")
                metadata = result.get("metadata", {})
                link_text = metadata.get("link_text", "")
                keywords = metadata.get("keywords", [])
                
                # Calculate advanced matching score
                match_score = self._calculate_thai_match_score(
                    meaningful_tokens, query_tokens, content, link_text, keywords
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
                    print(f"  ✅ Match: {link_text} (score: {match_score:.2f})")
            
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
                                   content: str, link_text: str, keywords: List[str]) -> float:
        """คำนวณคะแนนการ match แบบขั้นสูงสำหรับภาษาไทย"""
        score = 0.0
        
        content_lower = content.lower()
        link_text_lower = link_text.lower()
        keywords_lower = [k.lower() for k in keywords]
        
        # 1. Exact token matching in title (highest priority)
        for token in meaningful_tokens:
            if token.lower() in link_text_lower:
                score += 5.0  # High score for title match
                
        # 2. Exact token matching in content
        for token in meaningful_tokens:
            if token.lower() in content_lower:
                score += 3.0  # Medium score for content match
                
        # 3. Keyword matching
        for token in meaningful_tokens:
            for keyword in keywords_lower:
                if token.lower() in keyword or keyword in token.lower():
                    score += 2.0  # Medium score for keyword match
                    break
        
        # 4. Partial matching (for compound words)
        for token in all_tokens:
            if len(token) > 2:  # Only check longer tokens
                if token.lower() in link_text_lower:
                    score += 1.5
                elif token.lower() in content_lower:
                    score += 1.0
        
        # 5. Semantic bonus for service-related queries
        service_patterns = {
            'จอง': ['จอง', 'booking', 'reservation'],
            'ห้อง': ['ห้อง', 'room'],
            'ประชุม': ['ประชุม', 'meeting'],
            'ปฏิบัติการ': ['ปฏิบัติการ', 'lab', 'laboratory'],
            'แล็บ': ['แล็บ', 'lab', 'laboratory']
        }
        
        for token in meaningful_tokens:
            if token in service_patterns:
                related_words = service_patterns[token]
                for word in related_words:
                    if word in link_text_lower or word in content_lower:
                        score += 1.0
        
        return score
    
    def _exact_thai_keyword_search(self, query: str) -> List[Document]:
        """ค้นหาแบบ exact match สำหรับคำสำคัญภาษาไทย"""
        try:
            # Thai keyword mappings for exact matching
            thai_keyword_mappings = {
                # จองห้อง patterns
                "จองห้อง": ["จองห้อง", "ห้องประชุม", "reservation", "meeting"],
                "จองห้องประชุม": ["จองห้อง", "ห้องประชุม", "reservation", "meeting"],
                "ห้องประชุม": ["จองห้อง", "ห้องประชุม", "reservation", "meeting"],
                
                # ห้องแล็บ patterns  
                "ห้องแล็บ": ["ห้องแล็บ", "ปฏิบัติการ", "laboratory", "lab"],
                "แล็บ": ["ห้องแล็บ", "ปฏิบัติการ", "laboratory", "lab"],
                "ห้องปฏิบัติการ": ["ห้องแล็บ", "ปฏิบัติการ", "laboratory", "lab"],
                "จองแล็บ": ["ห้องแล็บ", "ปฏิบัติการ", "laboratory", "lab"],
                "จองห้องแล็บ": ["ห้องแล็บ", "ปฏิบัติการ", "laboratory", "lab"],
            }
            
            # Get all documents from collection
            all_results = list(self._collection.find({}, limit=100))
            matched_docs = []
            
            query_lower = query.lower().strip()
            
            # Check if query matches any Thai keyword patterns
            target_keywords = thai_keyword_mappings.get(query_lower, [query_lower])
            
            print(f"🔍 Thai Exact Search: '{query}' → looking for keywords: {target_keywords}")
            
            for result in all_results:
                content = result.get("content", "").lower()
                metadata = result.get("metadata", {})
                link_text = metadata.get("link_text", "").lower()
                keywords = [k.lower() for k in metadata.get("keywords", [])]
                
                # Calculate match score
                match_score = 0.0
                matches = []
                
                # Check matches in different fields
                for keyword in target_keywords:
                    if keyword in link_text:
                        match_score += 3.0  # High score for title match
                        matches.append(f"title:{keyword}")
                    elif keyword in content:
                        match_score += 2.0  # Medium score for content match
                        matches.append(f"content:{keyword}")
                    elif any(keyword in kw for kw in keywords):
                        match_score += 1.5  # Medium score for keyword match
                        matches.append(f"keywords:{keyword}")
                
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
                    print(f"  ✅ Match: {metadata.get('link_text', 'Unknown')} (score: {match_score:.2f}, matches: {matches})")
            
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
            results = self._collection.find({}, limit=50)
            
            for result in results:
                content = result.get("content", "").lower()
                original_content = result.get("content", "")
                link_text = result.get("metadata", {}).get("link_text", "").lower()
                
                matched = False
                for keyword in keywords:
                    keyword_lower = keyword.lower()
                    
                    # Check in content or link text
                    if keyword_lower in content or keyword_lower in link_text:
                        matched = True
                        break
                    
                    # Check in keywords metadata
                    metadata_keywords = result.get("metadata", {}).get("keywords", [])
                    if any(keyword_lower in mk.lower() for mk in metadata_keywords):
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
        
        # Remove common words
        stop_words = ["ขอ", "ลิงก์", "ลิงค์", "หา", "ค้นหา", "บอก", "แสดง", "ใคร", "คือ", "ของ", "ใน", "ที่", "และ", "หรือ"]
        
        keywords = []
        
        # Method 1: Clean version - remove stop words first
        query_clean = query
        for stop_word in stop_words:
            query_clean = query_clean.replace(stop_word, " ")
        query_clean = re.sub(r'\s+', ' ', query_clean).strip()
        
        if query_clean and len(query_clean) >= 2:
            keywords.append(query_clean)
        
        # Method 2: Split by spaces
        words_by_space = query.split()
        for word in words_by_space:
            clean_word = word.strip()
            if clean_word not in stop_words and len(clean_word) >= 2:
                keywords.append(clean_word)
        
        # Method 3: Add specific service keywords
        service_keywords = {
            "จอง": ["จอง", "booking", "reservation"],
            "ห้องประชุม": ["ห้องประชุม", "meeting", "room"],
            "ห้องแล็บ": ["ห้องแล็บ", "lab", "laboratory"],
            "อัปโหลด": ["อัปโหลด", "upload"],
            "เฟสบุ๊ก": ["เฟสบุ๊ก", "facebook"]
        }
        
        for service, related_words in service_keywords.items():
            if any(word in query.lower() for word in related_words):
                keywords.extend(related_words)
        
        # Remove duplicates while preserving order
        unique_keywords = []
        for keyword in keywords:
            if keyword not in unique_keywords and len(keyword) >= 2:
                unique_keywords.append(keyword)
        
        print(f"🔤 Debug: Query '{query}' -> Keywords: {unique_keywords}")
        return unique_keywords

retriever = LinksRetriever(collection, embedding)

# ✅ สร้าง Prompt - ปรับปรุงเพื่อให้เหมาะกับข้อมูลลิงก์
PROMPT = PromptTemplate.from_template("""
บริบทต่อไปนี้คือข้อมูลเกี่ยวกับลิงก์และบริการต่างๆ ของคณะวิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น
คุณคือผู้ช่วยที่ให้ข้อมูลเกี่ยวกับลิงก์และบริการของคณะ

สำคัญ: ให้ตรวจสอบข้อมูลในบริบทอย่างละเอียด หากมีข้อมูลที่ตรงกับคำถาม ให้นำมาตอบทันที

หากคำถามเกี่ยวกับลิงก์เฉพาะ ให้ตอบในรูปแบบ:
- ชื่อบริการ: [ชื่อบริการ]
- ลิงก์: [URL]
- คำอธิบาย: [อธิบายสั้นๆ ว่าใช้ทำอะไร]

หากคำถามเกี่ยวกับรายการลิงก์ ให้แสดงผลแบบรายการที่ชัดเจน:
- ใช้หัวข้อชัดเจน เช่น "รายการลิงก์บริการ:"
- แยกแต่ละลิงก์เป็นบรรทัดใหม่
- ใช้เครื่องหมาย • หรือ - นำหน้าแต่ละลิงก์
- แสดงทั้งชื่อบริการและ URL

หากไม่มีข้อมูลที่ตรงกับคำถามเลยในบริบท ให้ตอบว่า "ขอโทษ ฉันไม่พบลิงก์ที่คุณต้องการในระบบ"

---------------------
{context}
---------------------
คำถาม: {question}
คำตอบ (จัดรูปแบบให้อ่านง่าย):
""")

# ✅ โหลด Chat Model - Fixed for OpenRouter
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

if not openrouter_api_key:
    print("⚠️ Warning: OPENROUTER_API_KEY not found in .env file")
    print("LLM responses will not work without API key")
    llm = None
else:
    try:
        llm = ChatOpenAI(
            model="openai/gpt-4o-2024-11-20",  # Free model on OpenRouter
            temperature=0,
            openai_api_key=openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/your-repo",
                "X-Title": "Links RAG Chatbot"
            }
        )
        print("✅ OpenRouter LLM initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing LLM: {e}")
        llm = None

# ✅ สร้าง Manual QA Function
def manual_qa_chain(question: str) -> str:
    """
    Manual QA chain สำหรับตอบคำถามเกี่ยวกับลิงก์ - Links Version with astrapy
    """
    try:
        print(f"🔍 กำลังค้นหาลิงก์สำหรับคำถาม: {question}")
        print("🌐 ใช้ AstraDB Cloud Vector Database (astrapy) - Links Collection")
        print(f"📚 Collection: links_embedding")
        
        # ขั้นตอน 1: ดึงข้อมูลจาก retriever
        retrieved_docs = retriever.get_relevant_documents(question)
        
        if not retrieved_docs:
            return "ขอโทษ ฉันไม่พบลิงก์ที่คุณต้องการในระบบ"
        
        print(f"📚 พบข้อมูลลิงก์ {len(retrieved_docs)} รายการจาก AstraDB")
        
        # ขั้นตอน 2: แสดงผลลัพธ์ทั้งหมดก่อน
        print("\n" + "="*60)
        print("📋 ผลการค้นหาลิงก์ทั้งหมดจาก AstraDB (Links Collection):")
        print("="*60)
        
        for i, doc in enumerate(retrieved_docs, 1):
            combined_score = doc.metadata.get('combined_score', 0.0)
            bm25_score = doc.metadata.get('bm25_score', 0.0)
            vector_score = doc.metadata.get('vector_score', 0.0)
            
            print(f"\n🔸 ลิงก์ที่ {i}:")
            print(f"   ชื่อ: {doc.metadata.get('link_text', 'Unknown')}")
            print(f"   URL: {doc.metadata.get('url', 'Unknown')}")
            if combined_score > 0:
                print(f"   📊 คะแนนความเกี่ยวข้อง: {combined_score:.4f} (BM25: {bm25_score:.4f}, Vector: {vector_score:.4f})")
            print("-" * 40)
        
        print("\n" + "="*60)
        
        # ขั้นตอน 3: ใช้ข้อมูลทั้งหมดที่ค้นหาได้
        selected_docs = retrieved_docs  # ใช้ทั้งหมด
        
        print(f"🎯 ใช้ข้อมูลลิงก์ทั้งหมด {len(selected_docs)} รายการ สำหรับการตอบคำถาม")
        print("="*60)
        
        # จัดเตรียม context สำหรับ LLM
        context_parts = []
        print("\n📝 CONTEXT สำหรับ LLM:")
        print("-" * 40)
        
        for i, doc in enumerate(selected_docs, 1):
            link_text = doc.metadata.get('link_text', 'Unknown')
            print(f"📄 Context {i}: {link_text}")
            context_parts.append(f"ลิงก์ที่ {i}:\n{doc.page_content}\n")
        
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
            result_text = f"พบลิงก์ที่เกี่ยวข้อง {len(retrieved_docs)} รายการ:\n\n"
            for i, doc in enumerate(retrieved_docs, 1):
                link_text = doc.metadata.get('link_text', 'Unknown')
                url = doc.metadata.get('url', 'Unknown')
                result_text += f"{i}. {link_text}\n   ลิงก์: {url}\n\n"
            return result_text
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        return "ขอโทษ เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่อีกครั้ง"

# ✅ เริ่มถาม
if __name__ == "__main__":
    print("🔗 ระบบถาม-ตอบ ลิงก์บริการคณะคอมพิวเตอร์ มข. (Links Version with astrapy)")
    print("🌐 ใช้ AstraDB Cloud Vector Database - Links Collection")
    print(f"📚 Collection: links_embedding")
    print("พิมพ์ 'exit' เพื่อออก\n")
    print("ตัวอย่างคำถาม:")
    print("- ขอลิงก์จองห้องประชุม")
    print("- ลิงก์อัปโหลดไฟล์")
    print("- แสดงลิงก์ทั้งหมด")
    print("-" * 50)

    while True:
        question = input("❓ ถามมาเลย: ")
        if question.lower() == "exit":
            break

        # ใช้ manual QA chain แทน
        result = manual_qa_chain(question)
        print("🤖 คำตอบ:", result)
        print("-" * 50)
