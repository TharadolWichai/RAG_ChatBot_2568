import time

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

# ตั้งค่า Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # ไม่เปิดหน้าต่าง browser
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# ระบุ path ของ chromedriver (เปลี่ยนให้ตรงกับเครื่อง)
service = Service("D:/CS YEAR 4/chatbot_kkucp2568/RAG_ChatBot_2568/chromedriver-win64/chromedriver.exe")
driver = webdriver.Chrome(service=service, options=chrome_options)

# เข้าเว็บ
url = "https://computing.kku.ac.th/bsc-entrance"
driver.get(url)

# รอโหลด JS
time.sleep(3)

# ดึง HTML หลังโหลดเสร็จ
soup = BeautifulSoup(driver.page_source, "html.parser")
driver.quit()

# ดึง div class="kku-content"
content_divs = soup.find_all("div", class_="kku-content")
for div_index, div in enumerate(content_divs, start=1):
    print(f"=== DIV {div_index} ===")
    
    for child in div.find_all(recursive=False):
        # h3, h4
        if child.name in ["h3", "h4"]:
            print(child.get_text(strip=True))
        
        # p มี <a> อยู่ข้างใน
        elif child.name == "p":
            texts = []
            for elem in child.children:
                if getattr(elem, "name", None) == "a":
                    href = elem.get("href")
                    link_text = elem.get_text(strip=True)
                    print(f"- {link_text} ({href})")
                else:
                    text = elem.get_text(strip=True) if hasattr(elem, "get_text") else str(elem).strip()
                    if text:
                        texts.append(text)
            if texts:
                print(" ".join(texts))
        
        # ol + li + year-links
        elif child.name == "ol":
            for li in child.find_all("li"):
                li_text = li.get_text(" ", strip=True)
                print(f"- {li_text}")
                
                year_links_divs = li.find_all("div", class_="year-links")
                for yl_div in year_links_divs:
                    for a in yl_div.find_all("a"):
                        href = a.get("href")
                        link_text = a.get_text(strip=True)
                        print(f"  > Link: {link_text} ({href})")
        
        # table
        elif child.name == "table" and "kku-table" in child.get("class", []):
            print("Table:")
            for row in child.find_all("tr"):
                cells = row.find_all(["th", "td"])
                cell_texts = [cell.get_text(" ", strip=True) for cell in cells]
                print("\t".join(cell_texts))
        
        # kku-note
        elif child.name == "div" and "kku-note" in child.get("class", []):
            print("Note:")
            strong_tags = child.find_all("strong")
            links = child.find_all("a")
            
            for i, a in enumerate(links):
                description = ""
                if i < len(strong_tags):
                    strong = strong_tags[i]
                    u_tag = strong.find("u")
                    if u_tag:
                        description = u_tag.get_text(strip=True)
                    else:
                        description = strong.get_text(strip=True)
                        # ตรวจสอบว่าซ้ำกับ link หรือ href
                        if description.strip() == a.get_text(strip=True).strip() or description.strip() == a.get("href").strip():
                            description = ""  # ถ้าซ้ำ → ไม่ต้อง print
                
                if description:
                    print(f"  Description: {description} ")
                link_text = a.get_text(strip=True)
                print(f"- {link_text}")
    
    print("-" * 50)
