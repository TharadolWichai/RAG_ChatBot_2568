"""
Core scraping engine ที่รองรับทั้ง web scraping และ API
"""
import requests
from bs4 import BeautifulSoup
import time
import urllib3
from typing import List, Dict, Any, Optional
from langchain.schema import Document

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Selenium imports
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("⚠️ Selenium not available - will use requests only")


class WebScraper:
    """Web scraper ที่รองรับทั้ง requests และ Selenium"""
    
    def __init__(self, use_selenium: bool = True, wait_time: int = 3, 
                 custom_headers: Optional[Dict[str, str]] = None):
        """
        Args:
            use_selenium: ใช้ Selenium สำหรับ JavaScript rendering
            wait_time: เวลารอ (วินาที) สำหรับ JavaScript loading
            custom_headers: Custom HTTP headers
        """
        self.use_selenium = use_selenium and SELENIUM_AVAILABLE
        self.wait_time = wait_time
        self.custom_headers = custom_headers or {}
        self.driver = None
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default HTTP headers"""
        return {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'th-TH,th;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br'
        }
    
    def _setup_selenium_driver(self):
        """Setup Selenium Chrome driver"""
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium is not available")
        
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--ignore-certificate-errors')
        chrome_options.add_argument('--ignore-ssl-errors')
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        try:
            # Try webdriver-manager first
            from selenium.webdriver.chrome.service import Service as ChromeService
            from webdriver_manager.chrome import ChromeDriverManager
            service = ChromeService(ChromeDriverManager().install())
        except (ImportError, Exception):
            # Fallback to local chromedriver
            import os
            driver_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                      'drivers', 'chromedriver.exe')
            if os.path.exists(driver_path):
                service = Service(driver_path)
            else:
                # Try without .exe for Linux/Mac
                driver_path_no_exe = driver_path.replace('.exe', '')
                if os.path.exists(driver_path_no_exe):
                    service = Service(driver_path_no_exe)
                else:
                    raise FileNotFoundError(
                        f"ChromeDriver not found. Please install webdriver-manager: "
                        f"pip install webdriver-manager"
                    )
        
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
    
    def scrape_with_selenium(self, url: str) -> BeautifulSoup:
        """Scrape webpage using Selenium"""
        if not self.driver:
            self._setup_selenium_driver()
        
        print(f"   🌐 กำลังโหลดหน้าเว็บด้วย Selenium: {url}")
        self.driver.get(url)
        
        # รอให้ JavaScript โหลด
        print(f"   ⏳ รอ JavaScript rendering ({self.wait_time} วินาที)...")
        time.sleep(self.wait_time)
        
        # Parse HTML
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        return soup
    
    def scrape_with_requests(self, url: str) -> BeautifulSoup:
        """Scrape webpage using requests"""
        headers = {**self._get_default_headers(), **self.custom_headers}
        
        print(f"   🌐 กำลังโหลดหน้าเว็บด้วย requests: {url}")
        response = requests.get(url, headers=headers, verify=False, timeout=30)
        response.encoding = 'utf-8'
        
        if response.status_code != 200:
            raise Exception(f"HTTP Error: {response.status_code}")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup
    
    def scrape(self, url: str) -> BeautifulSoup:
        """
        Scrape webpage (auto-detect whether to use Selenium or requests)
        
        Args:
            url: URL ของหน้าเว็บที่ต้องการ scrape
            
        Returns:
            BeautifulSoup object
        """
        if self.use_selenium:
            try:
                return self.scrape_with_selenium(url)
            except Exception as e:
                print(f"⚠️ Selenium failed: {e}")
                print("   Falling back to requests...")
                return self.scrape_with_requests(url)
        else:
            return self.scrape_with_requests(url)
    
    def fetch_api(self, url: str, method: str = "GET", 
                  params: Optional[Dict] = None, 
                  json_data: Optional[Dict] = None) -> Any:
        """
        Fetch data from API endpoint
        
        Args:
            url: API endpoint URL
            method: HTTP method (GET, POST, etc.)
            params: Query parameters
            json_data: JSON payload for POST/PUT requests
            
        Returns:
            API response data (dict or list)
        """
        headers = {**self._get_default_headers(), **self.custom_headers}
        headers['Content-Type'] = 'application/json'
        
        print(f"   🔌 กำลังเรียก API: {method} {url}")
        
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, params=params, verify=False, timeout=30)
        elif method.upper() == "POST":
            response = requests.post(url, headers=headers, json=json_data, verify=False, timeout=30)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        response.encoding = 'utf-8'
        
        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
        
        try:
            return response.json()
        except:
            return response.text
    
    def close(self):
        """Close Selenium driver if opened"""
        if self.driver:
            self.driver.quit()
            self.driver = None
            print("   🔚 ปิด Chrome driver แล้ว")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


def is_api_url(url: str) -> bool:
    """
    Check if URL is likely an API endpoint
    
    Args:
        url: URL to check
        
    Returns:
        True if URL appears to be an API endpoint
    """
    api_indicators = ['/api/', '/json', '.json', '/rest/', '/graphql']
    return any(indicator in url.lower() for indicator in api_indicators)

