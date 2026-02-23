"""
Script สำหรับเช็คโมเดลที่มีใน KKU IntelSphere API

วิธีรัน:
    python check_kku_models.py
    
Output:
    - รายการโมเดลทั้งหมดที่ใช้ได้
    - หรือ error message ถ้าไม่สามารถดึงได้
"""
import os
import requests
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

def check_models_endpoint():
    """เช็คโมเดลผ่าน /models endpoint (OpenAI-compatible)"""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://gen.ai.kku.ac.th/api/v1")
    
    if not api_key:
        print("❌ ไม่พบ OPENAI_API_KEY ใน .env")
        return None
    
    print(f"🔍 กำลังเช็คโมเดลจาก: {base_url}")
    print(f"🔑 API Key: {api_key[:20]}...")
    print()
    
    # Try /models endpoint
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        print("📡 ลอง 1: GET /models (OpenAI standard)")
        response = requests.get(
            f"{base_url}/models",
            headers=headers,
            timeout=10
        )
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            models_data = response.json()
            print("   ✅ ดึงข้อมูลสำเร็จ!\n")
            return models_data
        else:
            print(f"   ⚠️ Response: {response.text[:500]}\n")
            
    except requests.exceptions.Timeout:
        print("   ⏱️ Timeout - API ไม่ตอบกลับภายใน 10 วินาที\n")
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
    
    return None

def test_with_wrong_model():
    """ทดสอบเรียกใช้โมเดลที่ไม่มี เพื่อดู error message"""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://gen.ai.kku.ac.th/api/v1")
    
    if not api_key:
        return None
    
    print("📡 ลอง 2: POST /chat/completions ด้วยโมเดลที่ไม่มี (เพื่อดู error)")
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json={
                "model": "test-invalid-model-xyz",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10
            },
            timeout=10
        )
        
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text[:500]}\n")
        
        return response.json() if response.status_code != 200 else None
        
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
        return None

def test_with_current_model():
    """ทดสอบเรียกใช้โมเดลปัจจุบัน (gemini-2.5-flash-lite)"""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://gen.ai.kku.ac.th/api/v1")
    model_name = os.getenv("OPENAI_MODEL", "gemini-2.5-flash-lite")
    
    if not api_key:
        return None
    
    print(f"📡 ลอง 3: POST /chat/completions ด้วยโมเดล '{model_name}'")
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "สวัสดี"}],
                "max_tokens": 50
            },
            timeout=30
        )
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("   ✅ API ทำงานได้!")
            if "choices" in result and len(result["choices"]) > 0:
                message = result["choices"][0].get("message", {}).get("content", "")
                print(f"   💬 คำตอบ: {message[:100]}...")
            print()
            return result
        else:
            print(f"   ⚠️ Response: {response.text[:500]}\n")
            return None
            
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
        return None

def parse_models_data(models_data):
    """Parse และแสดงรายการโมเดล"""
    if not models_data:
        return []
    
    print("="*60)
    print("📋 รายการโมเดลที่พบ:")
    print("="*60)
    
    models = []
    
    # OpenAI format: {"data": [{"id": "model-name", ...}, ...]}
    if isinstance(models_data, dict) and "data" in models_data:
        for model in models_data["data"]:
            model_id = model.get("id", "unknown")
            models.append(model_id)
            print(f"  ✅ {model_id}")
            
    # Simple list format: ["model1", "model2", ...]
    elif isinstance(models_data, list):
        for model in models_data:
            if isinstance(model, str):
                models.append(model)
                print(f"  ✅ {model}")
            elif isinstance(model, dict):
                model_id = model.get("id", model.get("name", "unknown"))
                models.append(model_id)
                print(f"  ✅ {model_id}")
    
    # Dict with models key
    elif isinstance(models_data, dict) and "models" in models_data:
        for model in models_data["models"]:
            if isinstance(model, str):
                models.append(model)
                print(f"  ✅ {model}")
            elif isinstance(model, dict):
                model_id = model.get("id", model.get("name", "unknown"))
                models.append(model_id)
                print(f"  ✅ {model_id}")
    
    else:
        print("  ⚠️ ไม่สามารถ parse ข้อมูลได้")
        print(f"  Raw data: {json.dumps(models_data, indent=2, ensure_ascii=False)[:500]}")
    
    print("="*60)
    print(f"📊 รวมทั้งหมด: {len(models)} โมเดล\n")
    
    return models

def save_models_to_file(models):
    """บันทึกรายการโมเดลลงไฟล์"""
    if not models:
        print("⚠️ ไม่มีโมเดลให้บันทึก")
        return
    
    output_file = "kku_available_models.json"
    data = {
        "source": "KKU IntelSphere API",
        "base_url": os.getenv("OPENAI_BASE_URL"),
        "models": models,
        "timestamp": None  # จะถูกเติมตอน import datetime
    }
    
    try:
        from datetime import datetime
        data["timestamp"] = datetime.now().isoformat()
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 บันทึกรายการโมเดลลงไฟล์: {output_file}")
    except Exception as e:
        print(f"❌ ไม่สามารถบันทึกไฟล์ได้: {e}")

def main():
    print("="*60)
    print("🔍 KKU IntelSphere - Model Checker")
    print("="*60)
    print()
    
    # Method 1: Try /models endpoint
    models_data = check_models_endpoint()
    
    if models_data:
        models = parse_models_data(models_data)
        if models:
            save_models_to_file(models)
            print("\n✅ สรุป: ดึงรายการโมเดลสำเร็จ!")
            print("📋 สามารถนำไปใช้ใน dashboard แบบ dropdown ได้")
            return
    
    # Method 2: Try with wrong model to see error
    print("="*60)
    print("🔍 ไม่สามารถดึงรายการโมเดลได้ - ลองวิธีอื่น")
    print("="*60)
    print()
    
    test_with_wrong_model()
    
    # Method 3: Test current model
    test_result = test_with_current_model()
    
    if test_result:
        print("="*60)
        print("✅ สรุป: API ทำงานได้ แต่ไม่สามารถดึงรายการโมเดลได้")
        print("="*60)
        print()
        print("💡 แนะนำ:")
        print("   1. ใช้ text input ในหน้า dashboard แทน dropdown")
        print("   2. ระบุโมเดล default: gemini-2.5-flash-lite")
        print("   3. ให้ผู้ใช้พิมพ์ชื่อโมเดลเองถ้าต้องการเปลี่ยน")
        print()
        print("📝 โมเดลที่แนะนำ (จากภาพที่เห็น):")
        print("   - gemini-2.5-flash-lite")
        print("   - gemini-pro")
        print("   - gemini-flash")
    else:
        print("="*60)
        print("❌ สรุป: ไม่สามารถเชื่อมต่อ API ได้")
        print("="*60)
        print()
        print("💡 ตรวจสอบ:")
        print("   1. API Key ถูกต้องหรือไม่")
        print("   2. Base URL ถูกต้องหรือไม่")
        print("   3. มี network access ไปยัง gen.ai.kku.ac.th หรือไม่")

if __name__ == "__main__":
    main()
