"""
Script เพื่อค้นหา Keyspace และ Collection ที่ถูกต้อง
"""
import os
from dotenv import load_dotenv
from astrapy import DataAPIClient

load_dotenv()

print("="*70)
print("🔍 กำลังค้นหา Keyspace และ Collection ใน AstraDB...")
print("="*70)

token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")

if not token or not api_endpoint:
    print("❌ Error: Missing credentials in .env file")
    exit(1)

print(f"\n📍 Endpoint: {api_endpoint}")

client = DataAPIClient(token=token)
database = client.get_database_by_api_endpoint(api_endpoint)

print(f"\n✅ เชื่อมต่อ AstraDB สำเร็จ!")

# ลิสต์ keyspace ที่จะลอง
keyspaces_to_try = [
    None,  # ไม่ระบุ (default)
    "default_keyspace",
    "ragchatbot",
    "ragchatbot_main",
    "RagChatbot",
    "RagChatbot_Main",
]

print(f"\n{'='*70}")
print("🔍 กำลังค้นหา Collections ใน Keyspace ต่างๆ...")
print(f"{'='*70}\n")

found_keyspaces = []

for keyspace in keyspaces_to_try:
    try:
        if keyspace:
            db_test = database.with_options(keyspace=keyspace)
            display_ks = keyspace
        else:
            db_test = database
            display_ks = "(default/no keyspace specified)"
        
        collections = list(db_test.list_collection_names())
        
        if collections:
            print(f"✅ Keyspace: {display_ks}")
            print(f"   📂 Collections found:")
            for col in collections:
                print(f"      - {col}")
            print()
            
            found_keyspaces.append({
                "keyspace": keyspace,
                "display": display_ks,
                "collections": collections
            })
        else:
            print(f"⚪ Keyspace: {display_ks}")
            print(f"   📂 No collections (keyspace exists but empty)")
            print()
            
    except Exception as e:
        error_msg = str(e)
        if "does not exist" in error_msg.lower():
            print(f"❌ Keyspace: {display_ks}")
            print(f"   ⚠️  Keyspace does not exist")
            print()
        else:
            print(f"❌ Keyspace: {display_ks}")
            print(f"   ⚠️  Error: {error_msg[:60]}...")
            print()

print(f"{'='*70}")
print("📊 สรุปผลการค้นหา")
print(f"{'='*70}\n")

if found_keyspaces:
    print(f"✅ พบ {len(found_keyspaces)} keyspace(s) ที่มี collections:\n")
    
    for i, ks_info in enumerate(found_keyspaces, 1):
        print(f"{i}. Keyspace: {ks_info['display']}")
        print(f"   Collections: {', '.join(ks_info['collections'])}\n")
    
    # หา keyspace ที่มี allpeople_embedding
    allpeople_ks = None
    for ks_info in found_keyspaces:
        if "allpeople_embedding" in ks_info['collections']:
            allpeople_ks = ks_info
            break
    
    if allpeople_ks:
        print("="*70)
        print("🎯 แนะนำ: ใช้ keyspace นี้!")
        print("="*70)
        print(f"\nKeyspace ที่มี 'allpeople_embedding': {allpeople_ks['display']}\n")
        
        if allpeople_ks['keyspace']:
            print("แก้ไขไฟล์ .env เป็น:")
            print("-"*70)
            print(f"ASTRA_DB_KEYSPACE={allpeople_ks['keyspace']}")
            print("-"*70)
        else:
            print("แก้ไขไฟล์ .env โดย:")
            print("-"*70)
            print("# ลบหรือ comment บรรทัด ASTRA_DB_KEYSPACE")
            print("# ASTRA_DB_KEYSPACE=...")
            print("-"*70)
    else:
        print("⚠️  ไม่พบ collection 'allpeople_embedding' ใน keyspace ใดเลย")
        print("\n💡 แนะนำ: สร้าง collection ใหม่ด้วยคำสั่ง:")
        print("-"*70)
        print("python create_collection.py")
        print("-"*70)
else:
    print("❌ ไม่พบ keyspace ใดๆ ที่มี collections")
    print("\n💡 คำแนะนำ:")
    print("1. ตรวจสอบว่า Database มี keyspace หรือยัง (ใน AstraDB UI)")
    print("2. ตรวจสอบว่า Token มี permission ที่ถูกต้อง")
    print("3. ลองสร้าง collection ใหม่ผ่าน AstraDB UI")

print(f"\n{'='*70}")
print("✅ เสร็จสิ้นการตรวจสอบ")
print(f"{'='*70}\n")

