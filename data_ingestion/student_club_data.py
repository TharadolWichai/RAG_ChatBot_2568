#!/usr/bin/env python3
"""
Student Club Data Ingestion from Web Scraping
ดึงข้อมูลสโมสรนักศึกษาจาก Web Scraping และบันทึกลง AstraDB
"""

import os
import re
import time
import uuid
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from astrapy import DataAPIClient
from incremental_utils import enable_incremental_mode
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

load_dotenv()

# -------------------------------
# Selenium setup
# -------------------------------
def setup_selenium_driver():
    """ตั้งค่า Selenium WebDriver"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--remote-debugging-port=9222")
    
    # ลองใช้ webdriver-manager เพื่อดาวน์โหลด ChromeDriver อัตโนมัติ
    try:
        from selenium.webdriver.chrome.service import Service as ChromeService
        from webdriver_manager.chrome import ChromeDriverManager
        service = ChromeService(ChromeDriverManager().install())
        print("✅ ใช้ webdriver-manager สำหรับ ChromeDriver")
    except ImportError:
        # Fallback: ใช้ chromedriver จากโฟลเดอร์ drivers
        driver_path = os.path.join(os.path.dirname(__file__), "..", "drivers", "chromedriver.exe")
        if os.path.exists(driver_path):
            service = Service(driver_path)
            print(f"⚠️  ใช้ ChromeDriver จากโฟลเดอร์ drivers (อาจเวอร์ชันไม่ตรง)")
        else:
            raise FileNotFoundError(
                f"ChromeDriver not found at {driver_path}. "
                "Please install webdriver-manager: pip install webdriver-manager"
            )
    
    return webdriver.Chrome(service=service, options=chrome_options)

def scrape_student_club_data():
    """ดึงข้อมูลสโมสรนักศึกษาจาก Web Scraping"""
    url = "https://computing.kku.ac.th/student-club"
    
    print(f"🌐 กำลังเข้าถึงเว็บไซต์: {url}")
    
    driver = setup_selenium_driver()
    try:
        driver.get(url)
        time.sleep(5)  # รอให้โหลด JavaScript
        
        soup = BeautifulSoup(driver.page_source, "html.parser")
        print("✅ โหลดหน้าเว็บสำเร็จ!")
        
        return soup
        
    except Exception as e:
        print(f"❌ Error accessing website: {e}")
        return None
    finally:
        driver.quit()

def clean_html_content(html_content: str) -> str:
    """ทำความสะอาด HTML content ให้เป็นข้อความธรรมดา"""
    try:
        # ใช้ BeautifulSoup ทำความสะอาด HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # ลบ script และ style tags
        for script in soup(["script", "style"]):
            script.decompose()
        
        # แยกข้อความออกมา
        text = soup.get_text()
        
        # ทำความสะอาดข้อความ
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    except:
        # ถ้าไม่สามารถ parse HTML ได้ ให้ลบ HTML tags ด้วย regex
        text = re.sub(r'<[^>]+>', '', html_content)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

def extract_student_club_info(soup):
    """ดึงข้อมูลสโมสรนักศึกษาจากหน้าเว็บ"""
    club_info = {
        "background": [],
        "individual_members": [],
        "basic_info": {}
    }
    
    print("🔍 กำลังแยกข้อมูลจากหน้าเว็บ...")
    
    # 1. ดึงข้อมูลพื้นฐาน
    basic_info = extract_basic_info(soup)
    if basic_info:
        club_info["basic_info"] = basic_info
    
    # 2. ดึงข้อมูลความเป็นมา
    background_info = extract_background_info(soup)
    if background_info:
        club_info["background"] = background_info
    
    # 3. ดึงข้อมูลคณะกรรมการแต่ละคน
    committee_members = extract_committee_members(soup)
    if committee_members:
        club_info["individual_members"] = committee_members
    
    return club_info

def extract_basic_info(soup):
    """ดึงข้อมูลพื้นฐานของสโมสร"""
    basic_info = {}
    
    # หาชื่อสโมสร
    title_elements = soup.find_all(['h1', 'h2', 'h3'], string=re.compile(r'สโมสร.*นักศึกษา'))
    if title_elements:
        basic_info["title"] = title_elements[0].get_text(strip=True)
    
    # หาข้อมูลเพิ่มเติม
    text_content = soup.get_text()
    if "สโมสรนักศึกษาวิทยาลัยการคอมพิวเตอร์" in text_content:
        basic_info["college"] = "วิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น"
    
    return basic_info

def extract_background_info(soup):
    """ดึงข้อมูลความเป็นมาของสโมสร - ปรับปรุงให้สะอาดขึ้น"""
    background_sections = []
    
    # ค้นหาส่วนที่มีความเป็นมา
    background_keywords = ["ความเป็นมา", "ประวัติ", "จัดตั้ง", "ก่อตั้ง", "เริ่มต้น"]
    
    print("🔍 กำลังค้นหาข้อมูลความเป็นมา...")
    
    # วิธีที่ 1: หาจาก heading ที่มีคำว่าความเป็นมา
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    for heading in headings:
        heading_text = heading.get_text(strip=True)
        if any(keyword in heading_text for keyword in background_keywords):
            print(f"✅ พบหัวข้อความเป็นมา: {heading_text}")
            
            # หาเนื้อหาที่ตามมาหลัง heading
            next_elements = []
            current = heading.find_next_sibling()
            
            while current and current.name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                if current.name in ['p', 'div'] and current.get_text(strip=True):
                    text = current.get_text(strip=True)
                    # กรองข้อความที่ไม่ต้องการ
                    if not is_unwanted_text(text):
                        next_elements.append(text)
                current = current.find_next_sibling()
            
            if next_elements:
                background_content = "\n".join(next_elements)
                # ทำความสะอาดเนื้อหา
                cleaned_content = clean_background_content(background_content)
                if cleaned_content and len(cleaned_content) > 50:
                    background_sections.append({
                        "section": heading_text,
                        "content": cleaned_content
                    })
    
    # วิธีที่ 2: หาจากเนื้อหาที่มีคำสำคัญเฉพาะ
    if not background_sections:
        print("🔍 ค้นหาจากเนื้อหาทั่วไป...")
        
        # หาจาก paragraph ที่มีเนื้อหาเกี่ยวกับความเป็นมา
        all_paragraphs = soup.find_all(['p', 'div'])
        for para in all_paragraphs:
            para_text = para.get_text(strip=True)
            
            # ตรวจสอบว่ามีเนื้อหาเกี่ยวกับความเป็นมาหรือไม่
            if (any(keyword in para_text for keyword in background_keywords) and 
                len(para_text) > 100 and 
                "สโมสร" in para_text and
                not is_unwanted_text(para_text)):
                
                cleaned_content = clean_background_content(para_text)
                if cleaned_content and len(cleaned_content) > 50:
                    background_sections.append({
                        "section": "ความเป็นมาสโมสรนักศึกษา",
                        "content": cleaned_content
                    })
    
    # วิธีที่ 3: หาจากข้อความที่มีรูปแบบเฉพาะ (วันที่ก่อตั้ง, วัตถุประสงค์)
    if not background_sections:
        print("🔍 ค้นหาจากรูปแบบเฉพาะ...")
        
        text_content = soup.get_text()
        
        # ค้นหาข้อความที่มีวันที่ก่อตั้ง
        date_patterns = [
            r'ก่อตั้ง.*?พ\.ศ\.?\s*\d{4}',
            r'จัดตั้ง.*?พ\.ศ\.?\s*\d{4}',
            r'เริ่มต้น.*?พ\.ศ\.?\s*\d{4}',
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text_content, re.IGNORECASE)
            for match in matches:
                # ขยายบริบทรอบๆ วันที่
                start_pos = text_content.find(match)
                if start_pos != -1:
                    context_start = max(0, start_pos - 200)
                    context_end = min(len(text_content), start_pos + len(match) + 200)
                    context = text_content[context_start:context_end]
                    
                    cleaned_content = clean_background_content(context)
                    if cleaned_content and len(cleaned_content) > 50:
                        background_sections.append({
                            "section": "ความเป็นมาสโมสรนักศึกษา",
                            "content": cleaned_content
                        })
                        break
    
    print(f"📊 พบข้อมูลความเป็นมา: {len(background_sections)} รายการ")
    return background_sections

def is_unwanted_text(text):
    """ตรวจสอบว่าเป็นข้อความที่ไม่ต้องการหรือไม่"""
    unwanted_patterns = [
        "ค้นหา",
        "เกี่ยวกับเรา",
        "ประวัติความเป็นมา",
        "โครงสร้างองค์กร",
        "วิสัยทัศน์/พันธกิจ",
        "ผู้บริหาร",
        "บุคลากร",
        "สิ่งอำนวยความสะดวก",
        "ภาคการศึกษา",
        "หลักสูตร",
        "การรับสมัคร",
        "ติดต่อเรา",
        "A-AA+",
        "navigation",
        "menu",
        "footer",
        "header"
    ]
    
    text_lower = text.lower()
    
    # ตรวจสอบว่ามีคำที่ไม่ต้องการหรือไม่
    for pattern in unwanted_patterns:
        if pattern.lower() in text_lower:
            return True
    
    # ตรวจสอบว่าเป็น navigation หรือ menu
    if len(text) < 20 or text.count("|") > 3:
        return True
    
    return False

def clean_background_content(content):
    """ทำความสะอาดเนื้อหาความเป็นมา"""
    if not content:
        return ""
    
    # ลบข้อความที่ไม่ต้องการ
    unwanted_phrases = [
        "ค้นหาA-AA+",
        "เกี่ยวกับเราประวัติความเป็นมาโครงสร้างองค์กร",
        "วิสัยทัศน์/พันธกิจผู้บริหารบุคลากร",
        "สิ่งอำนวยความสะดวก",
        "ติดต่อเรา",
        "123 ถ.มิตรภาพ",
        "043-009700",
        "computing.kku@kku.ac.th"
    ]
    
    cleaned = content
    for phrase in unwanted_phrases:
        cleaned = cleaned.replace(phrase, "")
    
    # ลบบรรทัดที่มีแต่เครื่องหมาย
    lines = cleaned.split('\n')
    clean_lines = []
    
    for line in lines:
        line = line.strip()
        # เก็บเฉพาะบรรทัดที่มีเนื้อหาที่มีความหมาย
        if (len(line) > 10 and 
            not line.startswith('|') and 
            not all(c in '|-+=' for c in line) and
            not is_unwanted_text(line)):
            clean_lines.append(line)
    
    # รวมบรรทัดที่สะอาดแล้ว
    result = '\n'.join(clean_lines).strip()
    
    # ลบช่องว่างเกิน
    result = re.sub(r'\n\s*\n', '\n\n', result)
    result = re.sub(r'\s+', ' ', result)
    
    return result

def extract_committee_members(soup):
    """ดึงข้อมูลคณะกรรมการแต่ละคนจากหน้าเว็บ"""
    members = []
    
    print("🔍 กำลังค้นหาข้อมูลคณะกรรมการ...")
    
    # วิธีที่ 1: หาจากการ์ดหรือ div ที่มีข้อมูลแต่ละคน (ตามรูปที่เห็น)
    member_cards = extract_from_member_cards(soup)
    if member_cards:
        members.extend(member_cards)
    
    # วิธีที่ 2: หาจากตารางคณะกรรมการ
    table_members = extract_from_committee_table(soup)
    if table_members:
        members.extend(table_members)
    
    # วิธีที่ 3: หาจากข้อความที่มีรูปแบบ "ตำแหน่ง: ชื่อ"
    text_members = extract_from_text_patterns(soup)
    if text_members:
        members.extend(text_members)
    
    # ลบข้อมูลที่ซ้ำกัน
    unique_members = []
    seen_names = set()
    
    for member in members:
        name_key = member.get("name", "").lower().strip()
        if name_key and name_key not in seen_names and len(name_key) > 3:
            unique_members.append(member)
            seen_names.add(name_key)
    
    print(f"✅ พบข้อมูลคณะกรรมการ: {len(unique_members)} คน")
    for member in unique_members:
        print(f"   - {member.get('position', 'ไม่ระบุตำแหน่ง')}: {member.get('name', 'ไม่ระบุชื่อ')}")
    
    return unique_members

def extract_from_member_cards(soup):
    """ดึงข้อมูลจากการ์ดสมาชิก (ตามรูปที่เห็น)"""
    members = []
    
    # หาการ์ดที่มีข้อมูลสมาชิก - ตามโครงสร้างในรูป
    cards = soup.find_all(['div'], class_=re.compile(r'card|member|profile'))
    
    for card in cards:
        card_text = card.get_text(strip=True)
        
        # ตรวจสอบว่าเป็นการ์ดสมาชิกหรือไม่
        if any(keyword in card_text for keyword in ["นักศึกษาชั้นปีที่", "ประธาน", "รองประธาน", "เลขานุการ", "เหรัญญิก"]):
            member_info = parse_member_card_text(card_text)
            if member_info:
                members.append(member_info)
    
    # หาจาก overlay text ในรูป (ตามที่เห็นในรูป)
    overlays = soup.find_all(['div'], class_=re.compile(r'overlay'))
    for overlay in overlays:
        overlay_text = overlay.get_text(strip=True)
        if any(keyword in overlay_text for keyword in ["นักศึกษาชั้นปีที่", "ประธาน", "รอง"]):
            member_info = parse_member_card_text(overlay_text)
            if member_info:
                members.append(member_info)
    
    return members

def parse_member_card_text(card_text):
    """แยกข้อมูลจากข้อความในการ์ด"""
    try:
        # รูปแบบที่เห็นในรูป: "นางสาวอุทัยพร ศรีนะ นักศึกษาชั้นปีที่ 3 สาขาวิชาวิทยาการคอมพิวเตอร์"
        
        # Pattern 1: ชื่อ + ข้อมูลนักศึกษา
        student_pattern = r'(นาย|นาง|นางสาว)\s*([^\s]+\s+[^\s]+)\s*นักศึกษาชั้นปีที่\s*(\d+)\s*สาขาวิชา([^\n]*)'
        match = re.search(student_pattern, card_text)
        
        if match:
            title = match.group(1)
            name = match.group(2).strip()
            year = match.group(3)
            major = match.group(4).strip()
            
            full_name = f"{title}{name}"
            student_info = f"นักศึกษาชั้นปีที่ {year} สาขาวิชา{major}"
            
            # หาตำแหน่งจากบริบท
            position = "สมาชิกคณะกรรมการ"
            if "ประธาน" in card_text and "รอง" not in card_text:
                position = "ประธานสโมสร"
            elif "รองประธาน" in card_text:
                position = "รองประธานสโมสร"
            elif "เลขานุการ" in card_text:
                position = "เลขานุการ"
            elif "เหรัญญิก" in card_text:
                position = "เหรัญญิก"
            
            return {
                "name": full_name,
                "position": position,
                "student_info": student_info,
                "year": year,
                "major": major,
                "details": f"{position}: {full_name} ({student_info})"
            }
    
    except Exception as e:
        print(f"❌ Error parsing member card: {e}")
    
    return None

def extract_from_committee_table(soup):
    """ดึงข้อมูลจากตารางคณะกรรมการ"""
    members = []
    
    tables = soup.find_all('table')
    for table in tables:
        # ตรวจสอบว่าเป็นตารางคณะกรรมการหรือไม่
        table_text = table.get_text().lower()
        if any(keyword in table_text for keyword in ["คณะกรรมการ", "ประธาน", "รอง", "เลขา"]):
            
            rows = table.find_all('tr')
            for row in rows[1:]:  # ข้ามหัวตาราง
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    position = cells[0].get_text(strip=True)
                    name = cells[1].get_text(strip=True)
                    
                    if position and name and len(name) > 2:
                        members.append({
                            "name": name,
                            "position": position,
                            "details": f"{position}: {name}"
                        })
    
    return members

def extract_from_text_patterns(soup):
    """ดึงข้อมูลจากรูปแบบข้อความ"""
    members = []
    
    text_content = soup.get_text()
    
    # รูปแบบ: "ตำแหน่ง: ชื่อ"
    position_patterns = [
        r'ประธาน[^:]*:?\s*([^\n]+)',
        r'รองประธาน[^:]*:?\s*([^\n]+)',
        r'เลขานุการ[^:]*:?\s*([^\n]+)',
        r'เหรัญญิก[^:]*:?\s*([^\n]+)',
    ]
    
    for pattern in position_patterns:
        matches = re.findall(pattern, text_content, re.IGNORECASE)
        for name in matches:
            name = name.strip()
            if len(name) > 2 and not any(skip in name.lower() for skip in ["ประกอบด้วย", "ดังนี้", "คือ"]):
                position = "สมาชิกคณะกรรมการ"
                if "ประธาน" in pattern and "รอง" not in pattern:
                    position = "ประธานสโมสร"
                elif "รองประธาน" in pattern:
                    position = "รองประธานสโมสร"
                elif "เลขานุการ" in pattern:
                    position = "เลขานุการ"
                elif "เหรัญญิก" in pattern:
                    position = "เหรัญญิก"
                
                members.append({
                    "name": name,
                    "position": position,
                    "details": f"{position}: {name}"
                })
    
    return members

def create_documents_from_scraped_data(club_info: Dict[str, Any]) -> List[Document]:
    """สร้าง Documents จากข้อมูลที่ scrape ได้ - แยก chunk ความเป็นมาและคณะกรรมการแต่ละคน"""
    documents = []
    
    print("📝 กำลังสร้าง Documents จากข้อมูลที่ scrape ได้...")
    
    # 1. ข้อมูลพื้นฐาน
    if club_info.get("basic_info"):
        basic_content = []
        for key, value in club_info["basic_info"].items():
            basic_content.append(f"{key}: {value}")
        
        if basic_content:
            doc = Document(
                page_content="\n".join(basic_content),
                metadata={
                    "type": "student_club_basic_info",
                    "category": "basic_information",
                    "source": "web_scraping"
                }
            )
            documents.append(doc)
            print(f"✅ สร้าง Document: ข้อมูลพื้นฐาน")
    
    # 2. ความเป็นมา (แยก chunk)
    if club_info.get("background"):
        for i, bg_item in enumerate(club_info["background"]):
            content = f"ความเป็นมาสโมสรนักศึกษาวิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น\n\n"
            content += f"หัวข้อ: {bg_item.get('section', 'ความเป็นมา')}\n\n"
            content += bg_item.get('content', '')
            
            doc = Document(
                page_content=content,
                metadata={
                    "type": "student_club_background",
                    "category": "background_history",
                    "section": bg_item.get('section', 'ความเป็นมา'),
                    "item_id": i + 1,
                    "source": "web_scraping"
                }
            )
            documents.append(doc)
            print(f"✅ สร้าง Document: ความเป็นมา - {bg_item.get('section', 'ความเป็นมา')}")
    
    # 3. คณะกรรมการแต่ละคน (แยก chunk แต่ละคน)
    if club_info.get("individual_members"):
        for i, member in enumerate(club_info["individual_members"]):
            # สร้าง content สำหรับแต่ละคน
            member_content = f"คณะกรรมการสโมสรนักศึกษาวิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น\n\n"
            member_content += f"ตำแหน่ง: {member.get('position', 'ไม่ระบุตำแหน่ง')}\n"
            member_content += f"ชื่อ-นามสกุล: {member.get('name', 'ไม่ระบุชื่อ')}\n"
            
            if member.get('student_info'):
                member_content += f"ข้อมูลการศึกษา: {member['student_info']}\n"
            if member.get('year'):
                member_content += f"ชั้นปี: {member['year']}\n"
            if member.get('major'):
                member_content += f"สาขาวิชา: {member['major']}\n"
            
            member_content += f"\nรายละเอียด: {member.get('details', '')}"
            
            doc = Document(
                page_content=member_content,
                metadata={
                    "type": "student_club_committee_member",
                    "category": "committee_members",
                    "position": member.get('position', 'ไม่ระบุตำแหน่ง'),
                    "name": member.get('name', 'ไม่ระบุชื่อ'),
                    "year": member.get('year', ''),
                    "major": member.get('major', ''),
                    "member_id": i + 1,
                    "source": "web_scraping"
                }
            )
            documents.append(doc)
            print(f"✅ สร้าง Document: {member.get('position', 'สมาชิก')} - {member.get('name', 'ไม่ระบุชื่อ')}")
        
        # สร้าง document รวมรายชื่อทั้งหมด
        if len(club_info["individual_members"]) > 0:
            all_members_list = []
            all_members_list.append("รายชื่อคณะกรรมการสโมสรนักศึกษาวิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น:")
            all_members_list.append("")
            
            for member in club_info["individual_members"]:
                member_line = f"• {member.get('position', 'สมาชิก')}: {member.get('name', 'ไม่ระบุชื่อ')}"
                if member.get('student_info'):
                    member_line += f" ({member['student_info']})"
                all_members_list.append(member_line)
            
            combined_content = "\n".join(all_members_list)
            
            doc = Document(
                page_content=combined_content,
                metadata={
                    "type": "student_club_committee_all",
                    "category": "committee_members",
                    "total_members": len(club_info["individual_members"]),
                    "source": "web_scraping"
                }
            )
            documents.append(doc)
            print(f"✅ สร้าง Document: รายชื่อคณะกรรมการทั้งหมด ({len(club_info['individual_members'])} คน)")
    
    return documents


def main():
    """Main function สำหรับดึงข้อมูลด้วย Web Scraping และบันทึกลง AstraDB"""
    print("🚀 Starting Student Club Web Scraping data ingestion...")
    
    # Check environment variables
    token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
    keyspace = os.getenv("ASTRA_DB_KEYSPACE", "default_keyspace")
    
    if not token or not api_endpoint:
        print("❌ Error: Missing AstraDB credentials in .env file")
        print("Please add:")
        print("ASTRA_DB_APPLICATION_TOKEN=your_token_here")
        print("ASTRA_DB_API_ENDPOINT=your_endpoint_here")
        return False
    
    print(f"🔑 Using endpoint: {api_endpoint}")
    print(f"🏠 Using keyspace: {keyspace}")
    
    # Initialize AstraDB client
    try:
        client = DataAPIClient(token=token)
        database = client.get_database_by_api_endpoint(api_endpoint)
        print("✅ Connected to AstraDB successfully!")
    except Exception as e:
        print(f"❌ Failed to connect to AstraDB: {e}")
        return False
    
    # Get existing collection
    collection_name = "student_club_embedding"
    try:
        existing_collections = list(database.list_collection_names())
        print(f"📂 Existing collections: {existing_collections}")
        
        if collection_name in existing_collections:
            collection = database.get_collection(collection_name)
            print(f"📂 Using existing collection: {collection_name}")
            
            # Clear existing data first
            print("🗑️ Clearing existing data from collection...")
            try:
                delete_result = collection.delete_many({})
                print(f"🗑️ Deleted existing documents from collection")
            except Exception as e:
                print(f"⚠️ Warning: Could not clear collection: {e}")
                
        else:
            print(f"❌ Collection {collection_name} not found!")
            print("Please create the collection via AstraDB UI with vector support:")
            print(f"  - Collection Name: {collection_name}")
            print("  - Vector Dimension: 384")
            print("  - Vector Metric: cosine")
            return False
            
    except Exception as e:
        print(f"❌ Failed to access collection: {e}")
        return False
    
    # Scrape data from website
    print("🌐 Scraping student club data from website...")
    soup = scrape_student_club_data()
    
    if not soup:
        print("❌ Failed to scrape data from website")
        return False
    
    # Extract information from scraped data
    print("📝 Extracting student club information...")
    club_info = extract_student_club_info(soup)
    
    if not club_info or not any(club_info.values()):
        print("❌ No student club information extracted")
        return False
    
    print(f"📊 สรุปข้อมูลที่ดึงได้:")
    print(f"   - ข้อมูลพื้นฐาน: {len(club_info.get('basic_info', {}))} รายการ")
    print(f"   - ความเป็นมา: {len(club_info.get('background', []))} รายการ")
    print(f"   - คณะกรรมการ: {len(club_info.get('individual_members', []))} คน")
    
    # Create documents
    print("📝 Creating documents from scraped data...")
    documents = create_documents_from_scraped_data(club_info)
    
    if not documents:
        print("❌ No documents created from scraped data")
        return False
    
    print(f"📊 Created {len(documents)} documents")
    
    # Show sample documents
    for i, doc in enumerate(documents[:3]):
        print(f"\n--- Document {i+1} ({doc.metadata.get('type', 'unknown')}) ---")
        print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
    
    # Split documents into chunks (smaller size for AstraDB limits)
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    
    # Filter out chunks that are too large for AstraDB (max 8000 bytes)
    filtered_chunks = []
    for chunk in chunks:
        content_size = len(chunk.page_content.encode('utf-8'))
        if content_size <= 7000:  # Leave some buffer
            filtered_chunks.append(chunk)
        else:
            print(f"⚠️ Skipping chunk with size {content_size} bytes (too large)")
    
    chunks = filtered_chunks
    print(f"📄 Created {len(chunks)} chunks (after filtering)")
    
    # Initialize embeddings
    print("🧠 Initializing embeddings model...")
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Use incremental indexing
    print("\n🚀 Starting incremental indexing...")
    try:
        # Convert chunks back to documents for incremental processing
        from langchain.schema import Document as LangChainDoc
        docs_for_incremental = []
        for chunk in chunks:
            docs_for_incremental.append(LangChainDoc(
                page_content=chunk.page_content,
                metadata=chunk.metadata
            ))
        
        stats = enable_incremental_mode(
            collection=collection,
            embedding_model=embedding,
            new_documents=docs_for_incremental,
            metadata_filter={"type": "student_club"},  # Filter สำหรับดึงเอกสารชมรม
            hash_keys=["club_name", "club_id"],  # Keys สำหรับสร้าง unique hash
            delete_missing=False  # ไม่ลบเอกสารเก่า
        )
        
        print(f"\n✅ Incremental indexing completed!")
        print(f"   - New documents inserted: {stats['inserted']}")
        print(f"   - Existing documents skipped: {stats['skipped']}")
        
    except Exception as e:
        print(f"❌ Failed to process incremental indexing: {e}")
        return False
    
    # Verify insertion
    try:
        count = collection.count_documents({})
        print(f"🔍 Total documents in student_club collection: {count}")
    except Exception as e:
        print(f"⚠️ Could not verify document count: {e}")
    
    print("🎉 Student Club Web Scraping data ingestion completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
