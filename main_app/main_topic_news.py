# main_topic_news.py - AstraDB Version สำหรับระบบค้นหาหัวข้อข่าวสาร
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
import os
from dotenv import load_dotenv
from typing import List, Any 
from astrapy import DataAPIClient

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

# Get single collection for topic news data
try:
    collection = database.get_collection("topicnews_embedding")
    print(f"✅ Connected to collection: topicnews_embedding")
except Exception as e:
    print(f"❌ Error accessing collection: {e}")
    print("Please make sure the 'topicnews_embedding' collection exists in AstraDB")
    print("You can create it by running: python data_ingestion/topic_news_data.py")
    exit(1)

# ✅ สร้าง Custom Retriever สำหรับ AstraDB (Topic News Collection)
class TopicNewsRetriever(BaseRetriever):
    collection: Any = None
    embedding: Any = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, collection, embedding):
        super().__init__()
        self.collection = collection
        self.embedding = embedding
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        print(f"🔍 Debug: กำลังค้นหาหัวข้อข่าวด้วย query: '{query}'")
        
        # ถ้าต้องการข้อมูลทั้งหมด ให้ใช้วิธีพิเศษ
        if any(word in query.lower() for word in ["ทั้งหมด", "ทุกหัวข้อ", "รายการทั้งหมด", "หัวข้อทั้งหมด", "ข่าวล่าสุด"]):
            print("🎯 ตรวจพบคำขอข้อมูลหัวข้อข่าวทั้งหมด - ใช้การค้นหาแบบครอบคลุม")
            return self._get_comprehensive_search()
        
        # Try multiple search strategies
        print("🔍 กำลังค้นหาด้วย AstraDB hybrid search...")
        
        # Strategy 1: Text search first for keyword matches
        print("🔍 เริ่ม Text Search...")
        text_results = self._text_search(query)
        
        # Strategy 2: Vector search
        print("🧠 เริ่ม Vector Search...")
        vector_results = self._vector_search(query)
        
        # Combine results with smart prioritization
        all_documents = []
        seen_content = set()
        
        print("🔄 รวมผลลัพธ์...")
        
        # Extract keywords for relevance scoring
        query_keywords = self._extract_search_keywords(query)
        
        # Smart merge: prioritize exact matches first
        prioritized_docs = []
        regular_docs = []
        
        # Check text search results first (exact matches)
        for i, doc in enumerate(text_results):
            if doc.page_content not in seen_content:
                content_lower = doc.page_content.lower()
                has_exact_match = any(keyword.lower() in content_lower 
                                    for keyword in query_keywords 
                                    if len(keyword) >= 3)
                
                if has_exact_match:
                    prioritized_docs.append(doc)
                    print(f"🎯 เพิ่มจาก Text Search #{i+1} (ความสำคัญสูง): {doc.metadata.get('title', doc.page_content[:50])}...")
                else:
                    regular_docs.append(doc)
                    print(f"➕ เพิ่มจาก Text Search #{i+1}: {doc.metadata.get('title', doc.page_content[:50])}...")
                seen_content.add(doc.page_content)
        
        # Check vector search results
        for i, doc in enumerate(vector_results):
            if doc.page_content not in seen_content:
                content_lower = doc.page_content.lower()
                has_semantic_match = any(keyword.lower() in content_lower 
                                       for keyword in query_keywords 
                                       if len(keyword) >= 3)
                
                if has_semantic_match:
                    prioritized_docs.append(doc)
                    print(f"🎯 เพิ่มจาก Vector Search #{i+1} (ความสำคัญสูง): {doc.metadata.get('title', doc.page_content[:50])}...")
                else:
                    regular_docs.append(doc)
                    print(f"➕ เพิ่มจาก Vector Search #{i+1}: {doc.metadata.get('title', doc.page_content[:50])}...")
                seen_content.add(doc.page_content)
            else:
                print(f"⚠️ ข้าม Vector Search #{i+1}: ซ้ำกับ Text Search")
        
        # Combine: prioritized first, then regular
        all_documents = prioritized_docs + regular_docs
        
        print(f"📊 สรุป: Text={len(text_results)}, Vector={len(vector_results)}, รวม={len(all_documents)} (unique)")
        
        return all_documents[:15]  # Return top 15 results
    
    def _get_comprehensive_search(self) -> List[Document]:
        """ค้นหาหัวข้อข่าวแบบครอบคลุมทั้งหมดจาก collection"""
        print("🚀 เริ่มการค้นหาหัวข้อข่าวแบบครอบคลุมจาก topicnews_embedding collection...")
        
        all_documents = []
        
        try:
            print("🔍 ค้นหาจาก topicnews_embedding collection...")
            results = self.collection.find({}, limit=20)  # Get up to 20 topic news
            
            for result in results:
                doc = Document(
                    page_content=result.get("content", ""),
                    metadata=result.get("metadata", {})
                )
                all_documents.append(doc)
            
            print(f"📊 จาก topicnews_embedding: {len(all_documents)} รายการ")
            
        except Exception as e:
            print(f"❌ Error in comprehensive search: {e}")
        
        print(f"🎯 พบหัวข้อข่าวครอบคลุมรวม: {len(all_documents)} รายการ")
        return all_documents
    
    def _vector_search(self, query: str) -> List[Document]:
        """ค้นหาแบบ vector search จาก collection"""
        all_documents = []
        
        try:
            print(f"🧠 Vector Search: กำลังสร้าง embedding สำหรับ query: '{query}'")
            
            # Generate query embedding
            query_vector = self.embedding.embed_query(query)
            print(f"📊 Vector Search: สร้าง embedding แล้ว (dimension: {len(query_vector)})")
            
            # Perform vector search
            results = self.collection.find(
                {},
                sort={"$vector": query_vector},
                limit=10  # Top 10 semantic matches
            )
            
            for result in results:
                doc = Document(
                    page_content=result.get("content", ""),
                    metadata=result.get("metadata", {})
                )
                all_documents.append(doc)
            
            print(f"🧠 Vector search: พบ {len(all_documents)} documents จาก semantic similarity")
            
            # Show top match
            if all_documents:
                title = all_documents[0].metadata.get('title', 'Unknown')
                print(f"   Top match: {title}")
            
            return all_documents
            
        except Exception as e:
            print(f"❌ Error in vector search: {e}")
            return []

    def _text_search(self, query: str) -> List[Document]:
        """ค้นหาแบบ text search จาก collection สำหรับคำที่แม่นยำ"""
        all_documents = []
        matched_keywords = []
        
        try:
            # Extract keywords from query
            keywords = self._extract_search_keywords(query)
            print(f"🔍 Text Search: กำลังใช้ keywords: {keywords}")
            
            # Get documents from collection
            results = self.collection.find({}, limit=50)  # Get more for text matching
            collection_docs = []
            
            for result in results:
                content = result.get("content", "").lower()
                
                # Check if any keyword matches
                matched = False
                for keyword in keywords:
                    keyword_lower = keyword.lower()
                    
                    # Method 1: Exact match in content
                    if keyword_lower in content:
                        matched = True
                        matched_keyword = keyword
                    
                    # Method 2: Match in title
                    title = result.get("metadata", {}).get("title", "").lower()
                    if keyword_lower in title:
                        matched = True
                        matched_keyword = keyword
                    
                    # Method 3: Match in categories
                    categories = result.get("metadata", {}).get("categories", [])
                    for category in categories:
                        if keyword_lower in category.lower():
                            matched = True
                            matched_keyword = keyword
                            break
                    
                    # Method 4: Match in author name
                    author = result.get("metadata", {}).get("author_name", "").lower()
                    if keyword_lower in author:
                        matched = True
                        matched_keyword = keyword
                    
                    if matched:
                        doc = Document(
                            page_content=result.get("content", ""),
                            metadata=result.get("metadata", {})
                        )
                        # Avoid duplicates
                        if not any(d.page_content == doc.page_content for d in collection_docs):
                            collection_docs.append(doc)
                            if matched_keyword not in matched_keywords:
                                matched_keywords.append(matched_keyword)
                            title = doc.metadata.get('title', 'Unknown')
                            print(f"✅ Text Match: '{matched_keyword}' พบใน: {title}")
                        break  # Found a match, no need to check other keywords
            
            all_documents.extend(collection_docs)
            print(f"📝 Text search: ใช้ keywords {matched_keywords} พบ {len(all_documents)} documents")
            return all_documents[:10]  # Return top 10
            
        except Exception as e:
            print(f"❌ Error in text search: {e}")
            return []
    
    def _extract_search_keywords(self, query: str) -> List[str]:
        """แยกคำสำคัญจากคำค้นหา"""
        import re
        
        # Remove common words
        stop_words = ["ขอ", "ข้อมูล", "ข่าว", "หัวข้อ", "หา", "ค้นหา", "บอก", "แสดง", "เกี่ยวกับ", "คือ", "ของ", "ใน", "ที่", "และ", "หรือ", "เรื่อง", "ล่าสุด"]
        
        keywords = []
        
        # Method 1: Split by spaces (for queries with spaces)
        words_by_space = query.split()
        for word in words_by_space:
            clean_word = word.strip()
            if clean_word not in stop_words and len(clean_word) > 1:
                keywords.append(clean_word)
        
        # Method 2: Extract potential keywords (Thai/English pattern)
        thai_pattern = r'[ก-๙]{2,15}'
        english_pattern = r'[A-Za-z]{3,15}'
        
        thai_words = re.findall(thai_pattern, query)
        english_words = re.findall(english_pattern, query)
        
        for word in thai_words + english_words:
            if word not in stop_words and len(word) >= 2 and word not in keywords:
                keywords.append(word)
        
        # Method 3: Extract individual words by removing stop words
        query_clean = query
        for stop_word in stop_words:
            query_clean = query_clean.replace(stop_word, " ")
        
        # Split cleaned query and add non-empty parts
        clean_parts = [part.strip() for part in query_clean.split() if part.strip()]
        for part in clean_parts:
            if len(part) >= 2 and part not in keywords:
                keywords.append(part)
        
        # Method 4: Add the original query
        if query.strip() not in keywords:
            keywords.append(query.strip())
        
        # Remove duplicates while preserving order
        unique_keywords = []
        for keyword in keywords:
            if keyword not in unique_keywords:
                unique_keywords.append(keyword)
        
        print(f"🔤 Debug: Query '{query}' -> Keywords: {unique_keywords}")
        return unique_keywords

retriever = TopicNewsRetriever(collection, embedding)

# ✅ สร้าง Prompt - ปรับปรุงเพื่อให้เหมาะกับข้อมูลหัวข้อข่าวสาร
PROMPT = PromptTemplate.from_template("""
บริบทต่อไปนี้คือข้อมูลหัวข้อข่าวสารจากวิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น
คุณคือผู้ช่วยที่ให้ข้อมูลหัวข้อข่าวสารและกิจกรรมล่าสุดของวิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น 

สำคัญ: ให้ตรวจสอบข้อมูลในบริบทอย่างละเอียด หากมีข้อมูลที่ตรงกับคำถาม ให้นำมาตอบทันที อย่าตอบว่าไม่พบข้อมูลถ้าจริงๆ แล้วมีข้อมูลอยู่

หากคำถามเกี่ยวกับรายการหัวข้อข่าว ให้แสดงผลแบบรายการที่ชัดเจน ดังนี้:
- ใช้หัวข้อชัดเจน เช่น "หัวข้อข่าวล่าสุดจากวิทยาลัยการคอมพิวเตอร์:"
- แยกแต่ละหัวข้อเป็นบรรทัดใหม่
- ใช้เครื่องหมาย • หรือ - นำหน้าแต่ละหัวข้อ
- แสดงชื่อหัวข้อ วันที่เผยแพร่ และคำอธิบายสั้น (ถ้ามี)
- เรียงตามวันที่เผยแพร่จากใหม่ไปเก่า

หากคำถามเกี่ยวกับหัวข้อข่าวเฉพาะเรื่อง ให้แสดงข้อมูลที่ครบถ้วน:
- ชื่อหัวข้อข่าว
- คำอธิบายสั้น
- เนื้อหาเพิ่มเติม (ถ้ามี)
- วันที่เผยแพร่
- ผู้เขียน (ถ้ามี)
- หมวดหมู่ข่าว
- Slug สำหรับอ้างอิง

หากคำถามเกี่ยวกับช่วงเวลา ให้เน้นข้อมูลวันที่และเรียงตามลำดับเวลา

หากไม่มีข้อมูลที่ตรงกับคำถามเลยในบริบท ให้ตอบว่า "ขอโทษ ฉันไม่พบหัวข้อข่าวสารที่เกี่ยวข้องในระบบ"

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
            model="openai/gpt-4o-mini",  # Free model on OpenRouter
            temperature=0,
            openai_api_key=openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/your-repo",
                "X-Title": "Topic News RAG Chatbot"
            }
        )
        print("✅ OpenRouter LLM initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing LLM: {e}")
        llm = None

# ✅ สร้าง Manual QA Function
def manual_qa_chain(question: str) -> str:
    """
    Manual QA chain ที่ควบคุมการทำงานได้ทุกขั้นตอน - Topic News Version with astrapy
    """
    try:
        print(f"🔍 กำลังค้นหาหัวข้อข่าวสำหรับคำถาม: {question}")
        print("🌐 ใช้ AstraDB Cloud Vector Database (astrapy) - Topic News Collection")
        print(f"📚 Collection: topicnews_embedding")
        
        # ขั้นตอน 1: ดึงข้อมูลจาก retriever
        retrieved_docs = retriever.get_relevant_documents(question)
        
        if not retrieved_docs:
            return "ขอโทษ ฉันไม่พบหัวข้อข่าวสารที่เกี่ยวข้องในระบบ"
        
        print(f"📚 พบหัวข้อข่าว {len(retrieved_docs)} รายการจาก AstraDB")
        
        # ขั้นตอน 2: แสดงผลลัพธ์ทั้งหมดก่อน
        print("\n" + "="*60)
        print("📋 ผลการค้นหาหัวข้อข่าวทั้งหมดจาก AstraDB (Topic News Collection):")
        print("="*60)
        
        # Group by title/slug to avoid showing duplicate news
        unique_news = {}
        for doc in retrieved_docs:
            title = doc.metadata.get('title', 'ไม่มีชื่อหัวข้อ')
            slug = doc.metadata.get('slug', 'unknown')
            key = f"{title}_{slug}"
            if key not in unique_news:
                unique_news[key] = []
            unique_news[key].append(doc)
        
        print(f"📰 พบหัวข้อข่าวที่แตกต่างกัน: {len(unique_news)} เรื่อง")
        
        for i, (key, docs) in enumerate(unique_news.items(), 1):
            title = docs[0].metadata.get('title', 'ไม่มีชื่อหัวข้อ')
            slug = docs[0].metadata.get('slug', 'unknown')
            published_at = docs[0].metadata.get('published_at', 'ไม่ระบุ')
            has_full_content = docs[0].metadata.get('has_full_content', False)
            
            print(f"\n🔸 หัวข้อที่ {i}: {title}")
            print(f"   Slug: {slug}")
            print(f"   วันที่เผยแพร่: {published_at}")
            print(f"   จำนวน chunks: {len(docs)}")
            print(f"   มีเนื้อหาเต็ม: {'✅' if has_full_content else '❌'}")
            print("-" * 40)
            # แสดง chunk แรกที่มีเนื้อหาครบถ้วน - แสดงทั้งหมดไม่ตัด
            best_chunk = max(docs, key=lambda x: len(x.page_content))
            print(best_chunk.page_content.strip())
            print("-" * 40)
        
        print("\n" + "="*60)
        
        # ขั้นตอน 3: ใช้ข้อมูลทั้งหมดที่ค้นหาได้
        selected_docs = retrieved_docs  # ใช้ทั้งหมด
        
        print(f"🎯 ใช้หัวข้อข่าวทั้งหมด {len(selected_docs)} รายการ สำหรับการตอบคำถาม")
        print(f"💡 เหตุผล: ใช้ข้อมูลทั้งหมดเพื่อให้ครอบคลุมและแม่นยำที่สุด")
        print("="*60)
        
        # จัดเตรียม context สำหรับ LLM
        context_parts = []
        print("\n📝 CONTEXT สำหรับ LLM:")
        print("-" * 40)
        
        for i, doc in enumerate(selected_docs, 1):
            title = doc.metadata.get('title', f'หัวข้อที่ {i}')
            print(f"📄 Context {i}: {title}")
            context_parts.append(f"หัวข้อข่าวที่ {i}:\n{doc.page_content}\n")
        
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
            return f"พบหัวข้อข่าวที่เกี่ยวข้อง {len(retrieved_docs)} รายการ แต่ไม่สามารถประมวลผลคำตอบได้"
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        return "ขอโทษ เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่อีกครั้ง"

# ✅ เริ่มถาม
if __name__ == "__main__":
    print("📰 ระบบถาม-ตอบ หัวข้อข่าวสารวิทยาลัยการคอมพิวเตอร์ มข. (Topic News Version with astrapy)")
    print("🌐 ใช้ AstraDB Cloud Vector Database - Topic News Collection")
    print(f"📚 Collection: topicnews_embedding")
    print("📋 ข้อมูลจากหน้า 1 และ 2 ของ API list")
    print("พิมพ์ 'exit' เพื่อออก\n")

    while True:
        question = input("❓ ถามเกี่ยวกับหัวข้อข่าวสาร: ")
        if question.lower() == "exit":
            break

        # ใช้ manual QA chain แทน
        result = manual_qa_chain(question)
        print("🤖 คำตอบ:", result)
        print("-" * 50)
