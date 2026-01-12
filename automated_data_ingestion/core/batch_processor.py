"""
Batch processor สำหรับดึง list จาก API และสร้าง jobs สำหรับแต่ละรายการ
"""
import json
import re
from typing import List, Dict, Any, Optional
import requests
from langchain.schema import Document
from langchain_openai import ChatOpenAI
import os


class BatchProcessor:
    """Process batch jobs - extract URLs/slugs from list API and create jobs for each item"""
    
    def __init__(self, use_openai: bool = True):
        self.use_openai = use_openai
        self.llm = None
        if use_openai:
            self._init_openai()
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.llm = ChatOpenAI(
                    model_name="gpt-4o-mini",
                    temperature=0,
                    openai_api_key=api_key
                )
                print("✅ OpenAI LLM initialized for batch processing")
            else:
                print("⚠️ OPENAI_API_KEY not found, using rule-based extraction")
                self.use_openai = False
        except Exception as e:
            print(f"⚠️ Failed to initialize OpenAI: {e}")
            self.use_openai = False
    
    def fetch_list_api(self, api_url: str) -> Any:
        """Fetch data from list API"""
        print(f"📥 Fetching list from API: {api_url}")
        try:
            response = requests.get(api_url, verify=False, timeout=30)
            response.encoding = 'utf-8'
            if response.status_code != 200:
                raise Exception(f"API Error: {response.status_code}")
            return response.json()
        except Exception as e:
            raise Exception(f"Failed to fetch list API: {e}")
    
    def extract_urls_slugs_with_llm(self, api_data: Any) -> List[Dict[str, str]]:
        """Extract URLs/slugs from API data using LLM"""
        if not self.llm:
            return self.extract_urls_slugs_rule_based(api_data)
        
        try:
            # Convert API data to JSON string
            json_str = json.dumps(api_data, ensure_ascii=False, indent=2)
            
            system_prompt = """คุณเป็นผู้ช่วยในการ extract URLs หรือ slugs จาก JSON API response

ภารกิจ:
1. ค้นหา URLs, slugs, หรือ identifiers ของรายการต่างๆ ใน JSON
2. ส่งคืนผลลัพธ์เป็น JSON array ของ objects ที่มี "slug" หรือ "url" field
3. ถ้าเป็น slug ให้ส่งใน field "slug", ถ้าเป็น URL ให้ส่งใน field "url"

รูปแบบ:
[
  {"slug": "mlislab"},
  {"slug": "aiii"},
  {"url": "https://computing.kku.ac.th/hardware-human"}
]

สำคัญ: Extract ทุกรายการที่มี ไม่ใช่แค่ตัวอย่าง"""
            
            # เพิ่มขนาด JSON ที่ส่งให้ LLM จาก 20K เป็น 100K เพื่อให้ extract ได้ครบถ้วน
            # และเพิ่มคำแนะนำให้ extract ทุกรายการ
            json_preview = json_str[:100000] if len(json_str) > 100000 else json_str
            
            user_prompt = f"""JSON API Response:
{json_preview}

กรุณา extract URLs หรือ slugs ของรายการทั้งหมดที่มีอยู่ใน JSON นี้

สำคัญ: 
- Extract ทุกรายการที่มี ไม่ใช่แค่ตัวอย่างแรกๆ
- ตรวจสอบว่ามีรายการทั้งหมดกี่รายการ และ extract ให้ครบทุกรายการ
- ถ้า JSON มี pagination หรือ structure ที่ซับซ้อน ให้ extract ทุกรายการในทุก level

ส่งคืนเป็น JSON array เท่านั้น (ไม่มี markdown formatting)"""
            
            from langchain.schema import HumanMessage, SystemMessage
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON from response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                extracted = json.loads(json_str)
                print(f"   ✅ LLM extracted {len(extracted)} items")
                return extracted
            
            # Fallback to rule-based
            print("   ⚠️ LLM extraction failed, using rule-based")
            return self.extract_urls_slugs_rule_based(api_data)
            
        except Exception as e:
            print(f"   ⚠️ LLM extraction error: {e}, using rule-based")
            return self.extract_urls_slugs_rule_based(api_data)
    
    def extract_urls_slugs_rule_based(self, api_data: Any) -> List[Dict[str, str]]:
        """Extract URLs/slugs using rule-based method"""
        results = []
        
        def find_slugs_urls(obj, path=""):
            """Recursively find slugs and URLs"""
            if isinstance(obj, dict):
                # Check for slug field (prioritize this)
                if "slug" in obj:
                    slug = obj["slug"]
                    if slug and isinstance(slug, str) and slug.strip():
                        slug_value = slug.strip()
                        # ตรวจสอบว่าเป็น slug ที่ถูกต้อง (ไม่ใช่ empty หรือ special values)
                        if slug_value and slug_value not in ["", "null", "undefined"]:
                            results.append({"slug": slug_value})
                
                # Check for url field
                if "url" in obj:
                    url = obj["url"]
                    if url and isinstance(url, str) and url.strip():
                        results.append({"url": url.strip()})
                
                # Check for href field
                if "href" in obj:
                    href = obj["href"]
                    if href and isinstance(href, str) and href.strip():
                        results.append({"url": href.strip()})
                
                # Recursively search
                for key, value in obj.items():
                    find_slugs_urls(value, f"{path}.{key}")
            
            elif isinstance(obj, list):
                for item in obj:
                    find_slugs_urls(item, path)
        
        find_slugs_urls(api_data)
        
        # Remove duplicates
        seen = set()
        unique_results = []
        for item in results:
            key = tuple(sorted(item.items()))
            if key not in seen:
                seen.add(key)
                unique_results.append(item)
        
        print(f"   ✅ Rule-based extracted {len(unique_results)} items")
        if len(unique_results) > 0:
            print(f"   📋 Sample slugs: {[item.get('slug', item.get('url', ''))[:50] for item in unique_results[:5]]}")
        return unique_results
    
    def generate_detail_urls(self, items: List[Dict[str, str]], 
                           url_pattern: Optional[str] = None,
                           api_pattern: Optional[str] = None) -> List[Dict[str, str]]:
        """Generate detail URLs from slugs using patterns"""
        detail_configs = []
        
        for item in items:
            slug = item.get("slug")
            url = item.get("url")
            
            detail_url = None
            detail_api = None
            
            if slug:
                # Generate URL from pattern
                if url_pattern:
                    detail_url = url_pattern.replace("{slug}", slug)
                
                # Generate API URL from pattern
                if api_pattern:
                    detail_api = api_pattern.replace("{slug}", slug)
            
            elif url:
                # Use existing URL
                detail_url = url
            
            if detail_url or detail_api:
                detail_configs.append({
                    "url": detail_url,
                    "api_url": detail_api,
                    "slug": slug
                })
        
        return detail_configs
    
    def process_batch_list(self, list_api_url: str,
                          detail_url_pattern: Optional[str] = None,
                          detail_api_pattern: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Process batch list API and return list of detail configurations
        
        Args:
            list_api_url: API URL ที่มี list ของรายการ
            detail_url_pattern: Pattern สำหรับสร้าง URL (เช่น "https://computing.kku.ac.th/{slug}")
            detail_api_pattern: Pattern สำหรับสร้าง API URL (เช่น "https://api.computing.kku.ac.th/api/v1/page/getPageMappingBySlug/{slug}")
        
        Returns:
            List of dicts with "url" and/or "api_url" keys
        """
        # 1. Fetch list API
        api_data = self.fetch_list_api(list_api_url)
        
        # 2. Extract URLs/slugs
        items = self.extract_urls_slugs_with_llm(api_data)
        
        if not items:
            print("   ⚠️ No items extracted from list API")
            return []
        
        # 3. Generate detail URLs
        detail_configs = self.generate_detail_urls(items, detail_url_pattern, detail_api_pattern)
        
        print(f"   ✅ Generated {len(detail_configs)} detail configurations")
        return detail_configs

