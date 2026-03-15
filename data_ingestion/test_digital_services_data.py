# test_digital_services_data.py - Test Script for Digital Services Auto-Discovery
# ทดสอบระบบ Auto-discover บริการดิจิตอล

import os
import sys
from typing import Dict, Set

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_auto_discovery():
    """
    ทดสอบการ Auto-discover บริการจากหน้าเว็บ
    """
    print("\n" + "="*70)
    print("🧪 Testing Digital Services Auto-Discovery")
    print("="*70 + "\n")
    
    # Import the function
    try:
        from digital_services_data import auto_discover_services, FALLBACK_SERVICES
        print("✅ Successfully imported auto_discover_services")
    except Exception as e:
        print(f"❌ Failed to import: {e}")
        return False
    
    # Run auto-discovery
    print("\n📡 Running auto-discovery...")
    try:
        discovered = auto_discover_services()
        print(f"✅ Auto-discovery completed")
    except Exception as e:
        print(f"❌ Auto-discovery failed: {e}")
        return False
    
    # Analyze results
    print(f"\n📊 Discovery Results:")
    print(f"   Total services found: {len(discovered)}")
    print(f"   Expected count: 10 (digital services only)")
    
    # List all discovered services
    print("\n📋 Discovered Services:")
    for idx, (slug, info) in enumerate(discovered.items(), 1):
        print(f"   {idx}. {info['name']}")
        print(f"      Slug: {slug}")
        print(f"      URL: {info['url']}")
        print(f"      Category: {info['category']}")
        print(f"      Keywords: {', '.join(info.get('keywords', [])[:3])}")
        print()
    
    return discovered


def test_service_validation(services: Dict):
    """
    ทดสอบว่าบริการที่ค้นพบเป็นบริการจริงๆ (ไม่ใช่หน้าอื่น)
    """
    print("\n" + "="*70)
    print("🔍 Validating Discovered Services")
    print("="*70 + "\n")
    
    # Expected service patterns (อย่างน้อยต้องมีอันใดอันหนึ่ง)
    valid_service_patterns = [
        'hosting', 'virtual', 'vm', 'apple', 'google', 
        'grammarly', 'chatgpt', 'gpt', 'ai', 'server',
        'nas', 'storage', 'gpu', 'h100', 'snapdrop',
        'digital', 'cloud', 'backup'
    ]
    
    # Invalid patterns (ต้องไม่มี)
    invalid_patterns = [
        'history', 'structure', 'vision', 'mission',
        'course', 'admission', 'scholarship', 'alumni',
        'club', 'staff', 'student', 'people', 'board',
        'academics', 'graduate', 'entrance', 'mikrotik'
    ]
    
    valid_count = 0
    invalid_count = 0
    
    for slug, info in services.items():
        name = info['name'].lower()
        url = info['url'].lower()
        
        # Check if it's a valid service
        is_valid_service = any(pattern in url or pattern in name for pattern in valid_service_patterns)
        
        # Check if it contains invalid patterns
        has_invalid = any(pattern in url or pattern in name for pattern in invalid_patterns)
        
        if is_valid_service and not has_invalid:
            valid_count += 1
            print(f"✅ VALID: {info['name']}")
            print(f"   URL: {url}")
        else:
            invalid_count += 1
            print(f"❌ INVALID: {info['name']}")
            print(f"   URL: {url}")
            if has_invalid:
                print(f"   Reason: Contains invalid patterns")
            else:
                print(f"   Reason: No service indicators")
        print()
    
    print(f"\n📊 Validation Summary:")
    print(f"   ✅ Valid services: {valid_count}")
    print(f"   ❌ Invalid services: {invalid_count}")
    print(f"   📈 Accuracy: {(valid_count / len(services) * 100):.1f}%")
    
    # Test passes if no invalid services found
    if invalid_count == 0:
        print("\n🎉 All services are valid!")
        return True
    else:
        print(f"\n⚠️ Found {invalid_count} invalid service(s)")
        return False


def test_expected_services(services: Dict):
    """
    ทดสอบว่ามีบริการที่คาดหวังครบหรือไม่
    """
    print("\n" + "="*70)
    print("✅ Checking Expected Services")
    print("="*70 + "\n")
    
    # Expected services (ชื่อหรือ URL pattern)
    expected_services = {
        "Web Hosting": ["hosting", "digitalsv-webhosting"],
        "Virtual Machine": ["virtual", "vm", "virtual-machine"],
        "Apple Store": ["apple"],
        "Google Play": ["google"],
        "Grammarly": ["grammarly"],
        "ChatGPT Plus": ["chatgpt", "gpt"],
        "AI Server": ["server", "ds-ai", "ai"],
        "NAS": ["nas"],
        "Snapdrop": ["snapdrop"],
        "H100": ["h100", "gpu"]
    }
    
    found_services = {}
    
    for expected_name, patterns in expected_services.items():
        found = False
        for slug, info in services.items():
            name = info['name'].lower()
            url = info['url'].lower()
            
            if any(pattern in url or pattern in name for pattern in patterns):
                found = True
                found_services[expected_name] = info['name']
                break
        
        if found:
            print(f"✅ Found: {expected_name} → {found_services[expected_name]}")
        else:
            print(f"❌ Missing: {expected_name}")
    
    print(f"\n📊 Coverage Summary:")
    print(f"   Found: {len(found_services)}/{len(expected_services)} services")
    print(f"   Coverage: {(len(found_services) / len(expected_services) * 100):.1f}%")
    
    if len(found_services) == len(expected_services):
        print("\n🎉 All expected services found!")
        return True
    else:
        print(f"\n⚠️ Missing {len(expected_services) - len(found_services)} service(s)")
        return False


def test_fallback_mechanism():
    """
    ทดสอบ Fallback mechanism
    """
    print("\n" + "="*70)
    print("🔄 Testing Fallback Mechanism")
    print("="*70 + "\n")
    
    try:
        from digital_services_data import FALLBACK_SERVICES
        print(f"✅ Fallback services loaded: {len(FALLBACK_SERVICES)} services")
        
        # List fallback services
        print("\n📋 Fallback Services:")
        for idx, (slug, info) in enumerate(FALLBACK_SERVICES.items(), 1):
            print(f"   {idx}. {info['name']}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to load fallback services: {e}")
        return False


def test_service_categories(services: Dict):
    """
    ทดสอบการจัดหมวดหมู่บริการ
    """
    print("\n" + "="*70)
    print("📂 Testing Service Categories")
    print("="*70 + "\n")
    
    categories = {}
    for slug, info in services.items():
        category = info.get('category', 'unknown')
        if category not in categories:
            categories[category] = []
        categories[category].append(info['name'])
    
    print("📊 Services by Category:")
    for category, service_list in categories.items():
        print(f"\n   {category.upper()} ({len(service_list)} services):")
        for service in service_list:
            print(f"      • {service}")
    
    return True


def run_all_tests():
    """
    รันทุก test
    """
    print("\n" + "="*70)
    print("🚀 STARTING ALL TESTS")
    print("="*70)
    
    results = {}
    
    # Test 1: Auto-discovery
    discovered = test_auto_discovery()
    if discovered:
        results['auto_discovery'] = True
        
        # Test 2: Service validation
        results['validation'] = test_service_validation(discovered)
        
        # Test 3: Expected services
        results['expected'] = test_expected_services(discovered)
        
        # Test 4: Categories
        results['categories'] = test_service_categories(discovered)
    else:
        results['auto_discovery'] = False
    
    # Test 5: Fallback
    results['fallback'] = test_fallback_mechanism()
    
    # Final summary
    print("\n" + "="*70)
    print("📊 FINAL TEST SUMMARY")
    print("="*70 + "\n")
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {status}: {test_name}")
    
    total_tests = len(results)
    passed_tests = sum(1 for p in results.values() if p)
    
    print(f"\n   Total: {passed_tests}/{total_tests} tests passed")
    print(f"   Success Rate: {(passed_tests / total_tests * 100):.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️ {total_tests - passed_tests} test(s) failed")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    run_all_tests()
