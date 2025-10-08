import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# -------------------------
# ตั้งค่า Chrome options
# -------------------------
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# ระบุ path ของ chromedriver
service = Service("D:/CS YEAR 4/chatbot_kkucp2568/RAG_ChatBot_2568/drivers/chromedriver.exe")
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

    try:
        # รอให้ div class="w-100 h-100" โหลด (สูงสุด 10 วินาที)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.w-100.h-100"))
        )
    except:
        print("❌ ไม่เจอ div.w-100.h-100")
        print("=" * 80)
        continue

    soup = BeautifulSoup(driver.page_source, "html.parser")
    content_div = soup.select_one("div.w-100.h-100")

    if not content_div:
        print("❌ ไม่เจอ div.w-100.h-100 (หลัง parse)")
        print("=" * 80)
        continue

    # -------------------------
    # ดึง h2>strong, h2, h3, p, ul, li และตัดซ้ำ
    # -------------------------
    seen_texts = set()
    for tag in content_div.find_all(["h2", "h3", "p", "ul", "li"], recursive=True):
        text = ""
        if tag.name == "h2":
            strong_tag = tag.find("strong")
            if strong_tag:
                text = strong_tag.get_text(strip=True)
            else:
                text = tag.get_text(" ", strip=True)
        else:
            text = tag.get_text(" ", strip=True)

        if text and text not in seen_texts:
            print(f"[{tag.name.upper()}] {text}")
            seen_texts.add(text)

    print("=" * 80)

driver.quit()
