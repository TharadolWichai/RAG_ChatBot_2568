import time

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

# -------------------------
# ตั้งค่า Chrome options
# -------------------------
chrome_options = Options()
chrome_options.add_argument("--headless")  # ไม่เปิดหน้าต่าง browser
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# ระบุ path ของ chromedriver
service = Service("D:/CS YEAR 4/chatbot_kkucp2568/RAG_ChatBot_2568/chromedriver-win64/chromedriver.exe")
driver = webdriver.Chrome(service=service, options=chrome_options)

# -------------------------
# ลิสต์ URLs
# -------------------------
urls = [
    "https://computing.kku.ac.th/hardware-human",
    "https://computing.kku.ac.th/mlislab",
    "https://computing.kku.ac.th/aiii",
    "https://computing.kku.ac.th/agtlab",
    "https://computing.kku.ac.th/asclab",
    "https://computing.kku.ac.th/nlsplab",
    "https://computing.kku.ac.th/aidalab",
    "https://computing.kku.ac.th/i-serg"
]

# -------------------------
# ดึงข้อมูลจากแต่ละ URL
# -------------------------
for url_index, url in enumerate(urls, start=1):
    print(f"\n🌐 [URL {url_index}] กำลังดึงข้อมูลจาก: {url}\n")
    driver.get(url)
    time.sleep(3)  # รอให้ JS โหลด (ปรับเวลาได้)

    soup = BeautifulSoup(driver.page_source, "html.parser")

    # -------------------------
    # หา div class="w-100 h-100" แรก
    # -------------------------
    content_div = soup.find("div", class_="w-100 h-100")

    if not content_div:
        print("❌ ไม่เจอ div.w-100.h-100")
        print("=" * 80)
        continue

    # -------------------------
    # ดึง tag h2, h3, p, ul, li และตัดซ้ำ
    # -------------------------
    seen_texts = set()
    for tag in content_div.find_all(["h2", "h3", "p", "ul", "li"], recursive=True):
        text = tag.get_text(" ", strip=True)
        if text and text not in seen_texts:
            print(f"[{tag.name.upper()}] {text}")
            seen_texts.add(text)

    print("=" * 80)

driver.quit()
