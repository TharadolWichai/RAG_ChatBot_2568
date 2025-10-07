"""
Utility functions for ChromeDriver management
"""
import os
import platform
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service


def get_chrome_driver_path():
    """
    Get the appropriate ChromeDriver path based on the operating system
    """
    current_dir = os.path.dirname(__file__)
    drivers_dir = os.path.join(current_dir, "..", "drivers")
    
    system = platform.system().lower()
    if system == "windows":
        driver_name = "chromedriver.exe"
    else:
        driver_name = "chromedriver"
    
    driver_path = os.path.join(drivers_dir, driver_name)
    return os.path.abspath(driver_path)


def check_chrome_driver_exists():
    """
    Check if ChromeDriver exists in the drivers folder
    """
    driver_path = get_chrome_driver_path()
    exists = os.path.exists(driver_path)
    
    if exists:
        print(f"✅ ChromeDriver found: {driver_path}")
    else:
        print(f"❌ ChromeDriver not found: {driver_path}")
        print("📋 Please download ChromeDriver from: https://chromedriver.chromium.org/downloads")
        print("📁 Place it in the 'drivers/' folder")
    
    return exists


def create_chrome_driver(headless=True):
    """
    Create and return a configured Chrome WebDriver instance
    
    Args:
        headless (bool): Whether to run Chrome in headless mode
    
    Returns:
        webdriver.Chrome: Configured Chrome WebDriver instance
    """
    if not check_chrome_driver_exists():
        raise FileNotFoundError("ChromeDriver not found. Please install it first.")
    
    # Chrome options
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    
    # Create service
    driver_path = get_chrome_driver_path()
    service = Service(driver_path)
    
    try:
        driver = webdriver.Chrome(service=service, options=chrome_options)
        print(f"🚀 Chrome WebDriver created successfully")
        return driver
    except Exception as e:
        print(f"❌ Failed to create Chrome WebDriver: {e}")
        raise


def test_chrome_driver():
    """
    Test ChromeDriver functionality
    """
    print("🧪 Testing ChromeDriver...")
    
    try:
        driver = create_chrome_driver(headless=True)
        driver.get("https://www.google.com")
        title = driver.title
        print(f"✅ ChromeDriver test successful! Page title: {title}")
        driver.quit()
        return True
    except Exception as e:
        print(f"❌ ChromeDriver test failed: {e}")
        return False


if __name__ == "__main__":
    # Run tests when script is executed directly
    print("🔧 ChromeDriver Utility Test")
    print("-" * 40)
    
    # Check if driver exists
    check_chrome_driver_exists()
    
    # Test driver functionality
    test_chrome_driver()
