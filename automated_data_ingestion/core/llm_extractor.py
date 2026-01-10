"""
LLM-based content extractor ที่ใช้ LLM ในการ extract ข้อมูลตาม prompt ที่ระบุ
"""
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from bs4 import BeautifulSoup
import json
import re


class LLMExtractor:
    """
    ใช้ LLM เพื่อ extract ข้อมูลจาก HTML หรือ JSON ตาม prompt ที่ระบุ
    
    วิธีทำงาน:
    1. รับ HTML/JSON content และ extraction prompt
    2. ใช้ LLM เพื่อ parse และ extract ข้อมูลที่ต้องการ
    3. สร้าง Document objects สำหรับเก็บใน database
    """
    
    def __init__(self, use_openai: bool = True, model_name: str = "gpt-4o-mini"):
        """
        Args:
            use_openai: ใช้ OpenAI API (ถ้า False จะใช้ rule-based extraction)
            model_name: ชื่อ OpenAI model
        """
        self.use_openai = use_openai
        self.model_name = model_name
        self.llm = None
        
        if use_openai:
            self._init_openai()
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        try:
            from langchain_openai import ChatOpenAI
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.llm = ChatOpenAI(
                    model_name=self.model_name,
                    temperature=0,
                    openai_api_key=api_key
                )
                print("✅ OpenAI LLM initialized")
            else:
                print("⚠️ OPENAI_API_KEY not found, falling back to rule-based extraction")
                self.use_openai = False
        except ImportError:
            print("⚠️ langchain-openai not installed, falling back to rule-based extraction")
            self.use_openai = False
    
    def extract_with_llm(self, content: str, prompt: str, content_type: str = "html") -> List[Document]:
        """
        Extract ข้อมูลด้วย LLM
        
        Args:
            content: HTML หรือ JSON content
            content_type: "html" หรือ "json"
            prompt: Prompt ที่ระบุว่าต้องการดึงข้อมูลส่วนไหน
            
        Returns:
            List of Document objects
        """
        if not self.use_openai or not self.llm:
            # Fallback to rule-based extraction
            return self.extract_rule_based(content, prompt, content_type)
        
        try:
            # สร้าง system prompt
            system_prompt = f"""คุณเป็นผู้ช่วยในการ extract ข้อมูลจาก {content_type.upper()} ตามที่ผู้ใช้ระบุ
- อ่านและเข้าใจเนื้อหา
- Extract ข้อมูลตาม prompt ที่ให้มา
- ส่งคืนผลลัพธ์เป็น JSON array ที่มี structure ตามนี้:
  [
    {{
      "content": "เนื้อหาที่ extract ได้",
      "metadata": {{
        "title": "หัวข้อ (ถ้ามี)",
        "url": "URL (ถ้ามี)",
        "type": "ประเภทข้อมูล",
        ...
      }}
    }}
  ]

- ถ้ามีข้อมูลหลายชิ้น ให้แยกเป็น document ต่างหาก
- เก็บ metadata ที่สำคัญไว้ใน metadata field
"""
            
            user_prompt = f"""เนื้อหา ({content_type}):
{content[:5000]}  # จำกัดขนาดเพื่อไม่ให้ token เกิน

Prompt สำหรับ extraction:
{prompt}

กรุณา extract ข้อมูลตาม prompt และส่งคืนเป็น JSON array เท่านั้น (ไม่ต้องมี markdown formatting)"""
            
            # Call LLM
            from langchain.schema import HumanMessage, SystemMessage
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            
            # Parse response
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON from response (try array first, then dict)
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    extracted_data = json.loads(json_str)
                    
                    documents = []
                    for item in extracted_data:
                        # Handle both dict and string formats
                        if isinstance(item, dict):
                            content = item.get("content", "")
                            # If content is empty, try to convert dict to string
                            if not content:
                                content = json.dumps(item, ensure_ascii=False, indent=2)
                            elif isinstance(content, dict):
                                content = json.dumps(content, ensure_ascii=False, indent=2)
                            
                            doc = Document(
                                page_content=str(content),
                                metadata=item.get("metadata", {})
                            )
                            documents.append(doc)
                        elif isinstance(item, str):
                            documents.append(Document(
                                page_content=item,
                                metadata={"extraction_method": "llm"}
                            ))
                    
                    if documents:
                        return documents
                except json.JSONDecodeError:
                    pass
            
            # Try to parse as single JSON object (dict)
            json_dict_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_dict_match:
                json_str = json_dict_match.group(0)
                try:
                    extracted_data = json.loads(json_str)
                    # Convert dict to string content
                    content = json.dumps(extracted_data, ensure_ascii=False, indent=2)
                    return [Document(
                        page_content=content,
                        metadata={"extraction_method": "llm", "prompt": prompt[:100]}
                    )]
                except json.JSONDecodeError:
                    pass
            
            # Fallback: สร้าง document เดียวจาก response text
            return [Document(
                page_content=response_text[:10000],  # Limit size
                metadata={"extraction_method": "llm", "prompt": prompt[:100]}
            )]
                
        except Exception as e:
            print(f"⚠️ LLM extraction failed: {e}")
            print("   Falling back to rule-based extraction...")
            return self.extract_rule_based(content, prompt, content_type)
    
    def extract_rule_based(self, content: str, prompt: str, content_type: str = "html") -> List[Document]:
        """
        Rule-based extraction (fallback เมื่อไม่มี LLM)
        
        Args:
            content: HTML หรือ JSON content
            prompt: Prompt สำหรับการ extract
            content_type: "html" หรือ "json"
            
        Returns:
            List of Document objects
        """
        documents = []
        prompt_lower = prompt.lower()
        
        # Check if prompt is asking for links extraction
        is_links_extraction = any(keyword in prompt_lower for keyword in [
            'ลิงค์', 'ลิงก์', 'link', 'url', 'ชื่อลิงค์', 'urlของ', 'แยกกันของแต่ละลิงค์'
        ])
        
        if content_type == "html":
            soup = BeautifulSoup(content, 'html.parser')
            
            # If prompt asks for links, extract links separately
            if is_links_extraction:
                print("   🔗 Detected links extraction mode from prompt")
                
                # Find main content area
                main_content = (
                    soup.find('main') or
                    soup.find('article') or
                    soup.find('div', class_=re.compile('content|main|body')) or
                    soup.find('body')
                )
                
                if not main_content:
                    main_content = soup
                
                # Extract all links
                links_found = []
                seen_urls = set()
                
                # Method 1: Find <li> -> <a> structure (common in navigation)
                all_li_elements = main_content.find_all('li')
                for li in all_li_elements:
                    link = li.find('a', href=True)
                    if link:
                        href = link.get('href', '').strip()
                        
                        # Get text from <span> inside <a> or from <a> directly
                        span = link.find('span')
                        if span:
                            link_text = span.get_text(strip=True)
                        else:
                            link_text = link.get_text(strip=True)
                        
                        if link_text and href and href not in seen_urls:
                            # Convert relative URL to absolute
                            if href.startswith('/'):
                                href = f"https://computing.kku.ac.th{href}"
                            elif not href.startswith('http'):
                                href = f"https://computing.kku.ac.th/{href}"
                            
                            # Skip invalid links
                            if href.startswith('#') or href.startswith('javascript:') or len(link_text) < 2:
                                continue
                            
                            # Skip navigation links
                            skip_keywords = ['เข้าสู่ระบบ', 'login', 'ค้นหา', 'search', 'menu', 'home', 'logo']
                            if any(kw in link_text.lower() for kw in skip_keywords):
                                continue
                            
                            seen_urls.add(href)
                            links_found.append({
                                "text": link_text,
                                "url": href
                            })
                
                # Method 2: Find all <a> tags (fallback)
                if not links_found:
                    for link in main_content.find_all('a', href=True):
                        href = link.get('href', '').strip()
                        link_text = link.get_text(strip=True)
                        
                        if link_text and href and href not in seen_urls:
                            # Convert relative URL
                            if href.startswith('/'):
                                href = f"https://computing.kku.ac.th{href}"
                            elif not href.startswith('http'):
                                href = f"https://computing.kku.ac.th/{href}"
                            
                            if href.startswith('#') or href.startswith('javascript:'):
                                continue
                            
                            seen_urls.add(href)
                            links_found.append({
                                "text": link_text,
                                "url": href
                            })
                
                # Create one document per link
                if links_found:
                    print(f"   📊 Found {len(links_found)} links")
                    for i, link_data in enumerate(links_found):
                        doc = Document(
                            page_content=f"ชื่อลิงค์: {link_data['text']}\nURL: {link_data['url']}",
                            metadata={
                                "extraction_method": "rule_based",
                                "content_type": "html",
                                "link_text": link_data['text'],
                                "url": link_data['url'],
                                "type": "link",
                                "link_index": i
                            }
                        )
                        documents.append(doc)
                else:
                    print("   ⚠️ No links found in HTML")
            else:
                # Regular content extraction (not links)
                main_content = (
                    soup.find('main') or
                    soup.find('article') or
                    soup.find('div', class_=re.compile('content|main|body')) or
                    soup.find('body')
                )
                
                if main_content:
                    # Extract text
                    text = main_content.get_text(separator='\n', strip=True)
                    
                    # Extract links
                    links = []
                    for link in main_content.find_all('a', href=True):
                        link_text = link.get_text(strip=True)
                        link_href = link.get('href', '')
                        if link_text and link_href:
                            links.append(f"{link_text}: {link_href}")
                    
                    # Extract lists
                    list_items = []
                    for ul in main_content.find_all(['ul', 'ol']):
                        for li in ul.find_all('li', recursive=False):
                            list_items.append(li.get_text(strip=True))
                    
                    # Combine content
                    content_parts = [text]
                    if links:
                        content_parts.append("\n\nLinks:\n" + "\n".join(links))
                    if list_items:
                        content_parts.append("\n\nList:\n" + "\n".join(list_items))
                    
                    doc = Document(
                        page_content="\n".join(content_parts),
                        metadata={
                            "extraction_method": "rule_based",
                            "content_type": "html",
                            "prompt_summary": prompt[:100]
                        }
                    )
                    documents.append(doc)
                else:
                    # ถ้าไม่เจอ main content ให้ใช้ body ทั้งหมด
                    body = soup.find('body')
                    if body:
                        documents.append(Document(
                            page_content=body.get_text(separator='\n', strip=True),
                            metadata={"extraction_method": "rule_based", "content_type": "html"}
                        ))
        
        elif content_type == "json":
            try:
                # Parse JSON if it's a string
                if isinstance(content, str):
                    data = json.loads(content)
                else:
                    data = content
                
                # Debug: print structure
                print(f"   🔍 Parsing JSON structure...")
                if isinstance(data, dict):
                    print(f"      Root keys: {list(data.keys())[:5]}...")  # Show first 5 keys
                    if "data" in data:
                        if isinstance(data["data"], dict) and "items" in data["data"]:
                            items_count = len(data["data"]["items"]) if isinstance(data["data"]["items"], list) else 0
                            print(f"      Found data.data.items with {items_count} items")
                elif isinstance(data, list):
                    print(f"      Root is array with {len(data)} items")
                
                # Check if prompt is asking for links extraction
                if is_links_extraction:
                    print("   🔗 Detected links extraction mode from prompt")
                    
                    # Try to extract links from JSON structure
                    links_found = []
                    
                    # Common patterns for links in JSON:
                    # 1. Array of objects with "text"/"title" and "url"/"href"/"link"
                    # 2. Nested structure with links
                    
                    def extract_links_from_object(obj, path=""):
                        """Recursively extract links from JSON object"""
                        links = []
                        
                        if isinstance(obj, dict):
                            # Check if this object has link-like fields
                            link_text = None
                            link_url = None
                            
                            # Try various field names for link text
                            for key in ["text", "title", "name", "label", "linkText", "link_text"]:
                                if key in obj and obj[key]:
                                    link_text = str(obj[key]).strip()
                                    break
                            
                            # Try various field names for link URL
                            for key in ["url", "href", "link", "urlPath", "path", "slug"]:
                                if key in obj and obj[key]:
                                    link_url = str(obj[key]).strip()
                                    # Convert relative URLs
                                    if link_url and not link_url.startswith('http') and link_url.startswith('/'):
                                        link_url = f"https://computing.kku.ac.th{link_url}"
                                    break
                            
                            # If we found both text and URL, add as link
                            if link_text and link_url:
                                # Check for duplicates by URL
                                if link_url not in [l["url"] for l in links_found]:
                                    links_found.append({
                                        "text": link_text,
                                        "url": link_url
                                    })
                            
                            # Also check nested structures
                            if "items" in obj or "links" in obj or "components" in obj:
                                for key in ["items", "links", "components", "pageComponent_Mapping"]:
                                    if key in obj and isinstance(obj[key], list):
                                        for item in obj[key]:
                                            extract_links_from_object(item, f"{path}.{key}")
                            
                            # Recursively check other nested objects
                            for key, value in obj.items():
                                if isinstance(value, (dict, list)) and key not in ["userLocalized"]:
                                    extract_links_from_object(value, f"{path}.{key}")
                        
                        elif isinstance(obj, list):
                            for i, item in enumerate(obj):
                                extract_links_from_object(item, f"{path}[{i}]")
                    
                    # Extract links from JSON structure
                    extract_links_from_object(data)
                    
                    # Create one document per link
                    if links_found:
                        print(f"   📊 Found {len(links_found)} links in JSON")
                        for i, link_data in enumerate(links_found):
                            doc = Document(
                                page_content=f"ชื่อลิงค์: {link_data['text']}\nURL: {link_data['url']}",
                                metadata={
                                    "extraction_method": "rule_based",
                                    "content_type": "json",
                                    "link_text": link_data['text'],
                                    "url": link_data['url'],
                                    "type": "link",
                                    "link_index": i
                                }
                            )
                            documents.append(doc)
                        
                        if documents:
                            return documents  # Return early if we found links
                    else:
                        print("   ⚠️ No links found in JSON structure, falling back to regular extraction")
                
                # Helper function to find array items in nested structure
                def find_array_items(obj, path=""):
                    """Recursively find arrays in nested JSON structure"""
                    items = []
                    
                    if isinstance(obj, list):
                        # Found an array - return its items
                        return obj, path
                    elif isinstance(obj, dict):
                        # Check common patterns for data arrays
                        # Pattern 1: data.items or data.data.items (common API structure)
                        if "items" in obj:
                            items_arr = obj["items"]
                            if isinstance(items_arr, list):
                                return items_arr, f"{path}.items" if path else "items"
                        
                        # Pattern 2: data array directly (most common pattern: data.data.items)
                        if "data" in obj:
                            data_val = obj["data"]
                            if isinstance(data_val, list):
                                return data_val, f"{path}.data" if path else "data"
                            elif isinstance(data_val, dict):
                                # Check for data.data.items pattern first
                                if "items" in data_val:
                                    if isinstance(data_val["items"], list):
                                        return data_val["items"], f"{path}.data.items" if path else "data.items"
                                # Also check if data itself contains another data layer
                                if "data" in data_val:
                                    nested_data = data_val["data"]
                                    if isinstance(nested_data, dict) and "items" in nested_data:
                                        if isinstance(nested_data["items"], list):
                                            return nested_data["items"], f"{path}.data.data.items" if path else "data.data.items"
                                    elif isinstance(nested_data, list):
                                        return nested_data, f"{path}.data.data" if path else "data.data"
                        
                        # Pattern 3: results array
                        if "results" in obj and isinstance(obj["results"], list):
                            return obj["results"], f"{path}.results" if path else "results"
                        
                        # Pattern 4: Recursively search nested objects
                        for key, value in obj.items():
                            if isinstance(value, (list, dict)):
                                result = find_array_items(value, f"{path}.{key}" if path else key)
                                if result:
                                    return result
                    
                    return None
                
                # For links extraction, try to extract links first
                if is_links_extraction and documents:
                    # Already found links, return early
                    return documents
                
                # Try to find array items in nested structure
                array_items = None
                array_path = None
                
                # First, try direct access pattern from allpeople_data.py: data["data"]["items"]
                if isinstance(data, dict):
                    if "data" in data and isinstance(data["data"], dict):
                        if "items" in data["data"] and isinstance(data["data"]["items"], list):
                            array_items = data["data"]["items"]
                            array_path = "data.data.items"
                            print(f"   ✅ Direct access: Found data.data.items with {len(array_items)} items")
                        elif "pageComponent_Mapping" in data["data"] and isinstance(data["data"]["pageComponent_Mapping"], list):
                            # For page mapping API
                            array_items = data["data"]["pageComponent_Mapping"]
                            array_path = "data.pageComponent_Mapping"
                            print(f"   ✅ Found pageComponent_Mapping with {len(array_items)} items")
                
                # If not found, try recursive search
                if not array_items:
                    if isinstance(data, list):
                        # Direct array
                        array_items = data
                        array_path = "root"
                    elif isinstance(data, dict):
                        # Try to find nested array using recursive function
                        result = find_array_items(data)
                        if result:
                            array_items, array_path = result
                            if array_items:
                                print(f"   ✅ Recursive search: Found array at {array_path} with {len(array_items)} items")
                
                # If found array items, create one document per item
                if array_items and isinstance(array_items, list):
                    print(f"   📊 Processing array with {len(array_items)} items at path: {array_path}")
                    items_processed = 0
                    
                    # Special handling for links extraction
                    if is_links_extraction:
                        print("   🔗 Processing as links extraction...")
                        links_found = []
                        
                        for i, item in enumerate(array_items):
                            if isinstance(item, dict):
                                # Try to extract link information from item
                                link_text = None
                                link_url = None
                                
                                # Try various field names for link text
                                for key in ["text", "title", "name", "label", "linkText", "link_text", "componentTitle", "componentName"]:
                                    if key in item and item[key]:
                                        link_text = str(item[key]).strip()
                                        break
                                
                                # Try various field names for link URL
                                for key in ["url", "href", "link", "urlPath", "path", "slug", "componentUrl", "redirectUrl"]:
                                    if key in item and item[key]:
                                        link_url = str(item[key]).strip()
                                        # Convert relative URLs
                                        if link_url and not link_url.startswith('http'):
                                            if link_url.startswith('/'):
                                                link_url = f"https://computing.kku.ac.th{link_url}"
                                            else:
                                                link_url = f"https://computing.kku.ac.th/{link_url}"
                                        break
                                
                                # Also check nested structures for links
                                if not link_text or not link_url:
                                    # Check nested data structure
                                    if "data" in item and isinstance(item["data"], dict):
                                        if not link_text:
                                            for key in ["text", "title", "name"]:
                                                if key in item["data"]:
                                                    link_text = str(item["data"][key]).strip()
                                                    break
                                        if not link_url:
                                            for key in ["url", "href", "path", "slug"]:
                                                if key in item["data"]:
                                                    link_url = str(item["data"][key]).strip()
                                                    if link_url and not link_url.startswith('http'):
                                                        if link_url.startswith('/'):
                                                            link_url = f"https://computing.kku.ac.th{link_url}"
                                                        else:
                                                            link_url = f"https://computing.kku.ac.th/{link_url}"
                                                    break
                                
                                # If we found link information, add it
                                if link_text and link_url:
                                    if link_url not in [l["url"] for l in links_found]:
                                        links_found.append({
                                            "text": link_text,
                                            "url": link_url
                                        })
                        
                        # Create one document per link
                        if links_found:
                            print(f"   ✅ Found {len(links_found)} links from array items")
                            for i, link_data in enumerate(links_found):
                                doc = Document(
                                    page_content=f"ชื่อลิงค์: {link_data['text']}\nURL: {link_data['url']}",
                                    metadata={
                                        "extraction_method": "rule_based",
                                        "content_type": "json",
                                        "link_text": link_data['text'],
                                        "url": link_data['url'],
                                        "type": "link",
                                        "link_index": i,
                                        "array_path": array_path
                                    }
                                )
                                documents.append(doc)
                            
                            if documents:
                                print(f"   ✅ Successfully created {len(documents)} link documents")
                                return documents
                    
                    # Regular array processing (not links extraction)
                    for i, item in enumerate(array_items):
                        if (i + 1) % 50 == 0 or i == 0:  # Print every 50 items or first item
                            print(f"      Processing item {i+1}/{len(array_items)}...")
                        if isinstance(item, dict):
                            # Extract relevant fields based on prompt
                            text_parts = []
                            
                            # Extract common fields that match the prompt
                            # (ชื่อ, นามสกุล, ตำแหน่ง, อีเมล, ความเชี่ยวชาญ, ผลงานวิจัย)
                            field_mappings = {
                                "fullname_th": ["fullname_th", "fullnameTh", "fullname", "name_th"],
                                "fullname_en": ["fullname_en", "fullnameEn", "fullnameEnglish"],
                                "firstname_th": ["firstname_th", "firstnameTh", "firstname"],
                                "lastname_th": ["lastname_th", "lastnameTh", "lastname"],
                                "firstname_en": ["firstname_en", "firstnameEn"],
                                "lastname_en": ["lastname_en", "lastnameEn"],
                                "position": ["academicPosition", "position", "ตำแหน่ง"],
                                "email": ["email", "emailAddress"],
                                "specialize": ["specializeDescription", "specialize", "ความเชี่ยวชาญ"],
                                "research": ["researchDescription", "research", "research_work", "ผลงานวิจัย"]
                            }
                            
                            # Handle userLocalized if it exists (common in allpeople API)
                            localized_data = {}
                            if "userLocalized" in item:
                                user_localized = item["userLocalized"]
                                if isinstance(user_localized, str):
                                    try:
                                        user_localized = json.loads(user_localized)
                                    except:
                                        pass
                                
                                if isinstance(user_localized, list):
                                    # Extract Thai (languageId=1) and English (languageId=2)
                                    for loc in user_localized:
                                        lang_id = loc.get("languageId")
                                        if lang_id == 1:  # Thai
                                            localized_data["firstname_th"] = loc.get("firstname", "")
                                            localized_data["lastname_th"] = loc.get("lastname", "")
                                            localized_data["specializeDescription"] = loc.get("specializeDescription", "")
                                            localized_data["researchDescription"] = loc.get("researchDescription", "")
                                        elif lang_id == 2:  # English
                                            localized_data["firstname_en"] = loc.get("firstname", "")
                                            localized_data["lastname_en"] = loc.get("lastname", "")
                            
                            # Build content from item
                            extracted_data = {}
                            
                            # Add localized data first
                            extracted_data.update(localized_data)
                            
                            # Extract fields from mappings
                            for field_key, possible_keys in field_mappings.items():
                                for key in possible_keys:
                                    if key in item and item[key]:
                                        extracted_data[field_key] = item[key]
                                        break
                            
                            # Also add important direct fields from item
                            important_fields = ["slug", "academicPosition", "email", "telephone", "id"]
                            for key in important_fields:
                                if key in item and item[key] and key not in extracted_data:
                                    extracted_data[key] = item[key]
                            
                            # Build text content in a structured way
                            # Format similar to original allpeople_data.py
                            if extracted_data.get("firstname_th") or extracted_data.get("firstname_en"):
                                # Build formatted content
                                name_parts = []
                                if extracted_data.get("firstname_th") and extracted_data.get("lastname_th"):
                                    name_parts.append(f"ชื่อ: {extracted_data['firstname_th']} {extracted_data.get('lastname_th', '')}")
                                if extracted_data.get("firstname_en") and extracted_data.get("lastname_en"):
                                    name_parts.append(f"ชื่อภาษาอังกฤษ: {extracted_data['firstname_en']} {extracted_data.get('lastname_en', '')}")
                                
                                if name_parts:
                                    text_parts.extend(name_parts)
                                
                                if extracted_data.get("academicPosition"):
                                    text_parts.append(f"ตำแหน่ง: {extracted_data['academicPosition']}")
                                if extracted_data.get("email"):
                                    text_parts.append(f"อีเมล: {extracted_data['email']}")
                                if extracted_data.get("telephone"):
                                    text_parts.append(f"เบอร์โทร: {extracted_data['telephone']}")
                                if extracted_data.get("specializeDescription"):
                                    # Clean HTML tags
                                    spec = str(extracted_data["specializeDescription"]).replace("<[^>]+>", "").replace("\n", " ")
                                    text_parts.append(f"ความเชี่ยวชาญ: {spec}")
                                if extracted_data.get("researchDescription"):
                                    # Clean HTML tags and truncate if too long
                                    research = str(extracted_data["researchDescription"]).replace("<[^>]+>", "").replace("\n", " ")
                                    if len(research) > 1000:
                                        research = research[:1000] + "..."
                                    text_parts.append(f"ผลงานวิจัย: {research}")
                            
                            # If no structured format found, include all fields
                            if not text_parts:
                                for key, value in extracted_data.items():
                                    if value:
                                        if isinstance(value, (dict, list)):
                                            text_parts.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
                                        else:
                                            text_parts.append(f"{key}: {value}")
                            
                            # Build metadata with important fields for incremental indexing
                            doc_metadata = {
                                "extraction_method": "rule_based",
                                "content_type": "json",
                                "item_index": i,
                                "array_path": array_path,
                                "prompt_summary": prompt[:100],
                                "type": "allpeople",  # Default type for allpeople data
                            }
                            
                            # Add extracted fields to metadata (for hash_keys)
                            if extracted_data.get("firstname_th") and extracted_data.get("lastname_th"):
                                doc_metadata["name"] = f"{extracted_data['firstname_th']} {extracted_data.get('lastname_th', '')}"
                            if extracted_data.get("academicPosition"):
                                doc_metadata["position"] = extracted_data["academicPosition"]
                            if extracted_data.get("slug"):
                                doc_metadata["slug"] = extracted_data["slug"]
                            if extracted_data.get("email"):
                                doc_metadata["email"] = extracted_data["email"]
                            
                            # Add other simple fields from item
                            for k, v in item.items():
                                if isinstance(v, (str, int, float, bool)) and k not in ["userLocalized"] and k not in doc_metadata:
                                    doc_metadata[k] = v
                            
                            doc = Document(
                                page_content="\n".join(text_parts),
                                metadata=doc_metadata
                            )
                            documents.append(doc)
                            items_processed += 1
                        else:
                            # Simple item (string or number)
                            doc = Document(
                                page_content=str(item),
                                metadata={
                                    "extraction_method": "rule_based",
                                    "content_type": "json",
                                    "item_index": i,
                                    "array_path": array_path
                                }
                            )
                            documents.append(doc)
                            items_processed += 1
                    
                    if documents:
                        print(f"   ✅ Successfully created {len(documents)} documents from {items_processed} array items")
                        return documents  # Return early if we found and processed array
                
                # If we didn't find array items, handle as single object
                if not documents:
                    if isinstance(data, dict):
                        # Single dict - convert to text
                        text_parts = []
                        for key, value in data.items():
                            if value:
                                if isinstance(value, (dict, list)):
                                    text_parts.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
                                else:
                                    text_parts.append(f"{key}: {value}")
                        
                        doc = Document(
                            page_content="\n".join(text_parts),
                            metadata={
                                "extraction_method": "rule_based",
                                "content_type": "json",
                                "prompt_summary": prompt[:100]
                            }
                        )
                        documents.append(doc)
                else:
                    # Primitive type
                    doc = Document(
                        page_content=str(data),
                        metadata={
                            "extraction_method": "rule_based",
                            "content_type": "json"
                        }
                    )
                    documents.append(doc)
                
            except Exception as e:
                print(f"⚠️ Error parsing JSON: {e}")
                documents.append(Document(
                    page_content=str(content)[:10000],  # Limit size
                    metadata={"extraction_method": "rule_based", "error": str(e)}
                ))
        
        return documents if documents else [Document(
            page_content=content[:1000],
            metadata={"extraction_method": "rule_based", "note": "minimal_extraction"}
        )]
    
    def extract_from_json_object(self, json_object: dict, prompt: str) -> List[Document]:
        """
        Extract directly from JSON object (bypass LLM for API responses with known structure)
        This is more efficient for structured API data like data.data.items
        
        Args:
            json_object: Raw JSON object from API
            prompt: Prompt สำหรับการ extract (for reference)
            
        Returns:
            List of Document objects
        """
        # Direct extraction from JSON object using rule-based method
        # Pass the object directly (not as string) so extract_rule_based can work with it
        print("   🔧 Using direct JSON object extraction (bypassing LLM)")
        import json
        content_str = json.dumps(json_object, ensure_ascii=False, indent=2)
        # Call extract_rule_based which will parse it back, but this ensures we use rule-based logic
        result = self.extract_rule_based(content_str, prompt, "json")
        print(f"   📊 Direct extraction result: {len(result)} documents")
        return result
    
    def extract(self, content: str, prompt: str, content_type: str = "html") -> List[Document]:
        """
        Main extraction method (auto-select LLM or rule-based)
        
        Args:
            content: HTML หรือ JSON content
            prompt: Prompt สำหรับการ extract
            content_type: "html" หรือ "json"
            
        Returns:
            List of Document objects
        """
        if self.use_openai and self.llm:
            return self.extract_with_llm(content, prompt, content_type)
        else:
            return self.extract_rule_based(content, prompt, content_type)

