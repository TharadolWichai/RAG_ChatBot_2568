"""
สร้าง Keyspace และ Collection ใน AstraDB ผ่านโค้ด
"""
import os
from dotenv import load_dotenv
from astrapy import DataAPIClient

load_dotenv()

print("="*70)
print("🚀 กำลังสร้าง Keyspace และ Collection ใน AstraDB...")
print("="*70)

# Load credentials
token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")

if not token or not api_endpoint:
    print("❌ Error: Missing credentials in .env")
    exit(1)

print(f"\n📍 Endpoint: {api_endpoint}")

# Connect to AstraDB
try:
    client = DataAPIClient(token=token)
    database = client.get_database_by_api_endpoint(api_endpoint)
    print("✅ เชื่อมต่อ AstraDB สำเร็จ!")
except Exception as e:
    print(f"❌ ไม่สามารถเชื่อมต่อ: {e}")
    exit(1)

# ===========================
# ขั้นตอนที่ 1: สร้าง Keyspace
# ===========================
keyspace_name = "rag_data"
print(f"\n{'='*70}")
print(f"🔧 ขั้นตอนที่ 1: สร้าง Keyspace '{keyspace_name}'")
print(f"{'='*70}")

try:
    # ลองสร้าง keyspace (ถ้ามีอยู่แล้วจะ skip)
    # Note: AstraDB Serverless ไม่รองรับ create keyspace ผ่าน Data API
    # ต้องใช้ DevOps API แทน
    
    # ลองเชื่อมต่อกับ keyspace ก่อน
    try:
        test_db = database.with_options(keyspace=keyspace_name)
        collections = list(test_db.list_collection_names())
        print(f"✅ Keyspace '{keyspace_name}' มีอยู่แล้ว")
        database = test_db
    except Exception as e:
        if "does not exist" in str(e).lower():
            print(f"⚠️  Keyspace '{keyspace_name}' ยังไม่มี กำลังสร้าง...")
            
            # สำหรับ AstraDB Serverless - ใช้ admin API
            try:
                admin = client.get_admin()
                # Extract database ID from endpoint
                db_id = api_endpoint.split("https://")[1].split("-")[0]
                
                admin.create_keyspace(
                    database=db_id,
                    keyspace=keyspace_name
                )
                print(f"✅ สร้าง Keyspace '{keyspace_name}' สำเร็จ!")
                
                # รอสักครู่ให้ keyspace พร้อมใช้งาน
                import time
                print("⏳ รอ keyspace พร้อมใช้งาน...")
                time.sleep(3)
                
                database = database.with_options(keyspace=keyspace_name)
                
            except Exception as create_error:
                print(f"⚠️  ไม่สามารถสร้าง keyspace ผ่าน API: {create_error}")
                print(f"\n💡 วิธีแก้: ใช้ keyspace เริ่มต้นแทน")
                print(f"   จะสร้าง collection โดยไม่ระบุ keyspace")
                # ไม่ระบุ keyspace ใช้ default
                keyspace_name = None
        else:
            raise

except Exception as e:
    print(f"❌ Error: {e}")
    print(f"\n💡 จะลองสร้าง collection โดยไม่ระบุ keyspace")
    keyspace_name = None

# ===========================
# ขั้นตอนที่ 2: สร้าง Collection
# ===========================
collection_name = "allpeople_embedding"
print(f"\n{'='*70}")
print(f"🔧 ขั้นตอนที่ 2: สร้าง Collection '{collection_name}'")
if keyspace_name:
    print(f"   ใน Keyspace: {keyspace_name}")
print(f"{'='*70}")

try:
    # ตรวจสอบว่า collection มีอยู่แล้วหรือไม่
    existing_collections = list(database.list_collection_names())
    
    if collection_name in existing_collections:
        print(f"✅ Collection '{collection_name}' มีอยู่แล้ว")
        collection = database.get_collection(collection_name)
    else:
        print(f"📦 กำลังสร้าง collection '{collection_name}'...")
        
        # สร้าง collection พร้อม vector search
        collection = database.create_collection(
            name=collection_name,
            dimension=384,  # สำหรับ sentence-transformers/all-MiniLM-L6-v2
            metric="cosine"
        )
        
        print(f"✅ สร้าง Collection '{collection_name}' สำเร็จ!")
        print(f"   - Vector Dimension: 384")
        print(f"   - Similarity Metric: cosine")
        
except Exception as e:
    print(f"❌ ไม่สามารถสร้าง collection: {e}")
    exit(1)

# ===========================
# ขั้นตอนที่ 3: ตรวจสอบ
# ===========================
print(f"\n{'='*70}")
print(f"🔍 ขั้นตอนที่ 3: ตรวจสอบการตั้งค่า")
print(f"{'='*70}")

try:
    # ตรวจสอบ collection
    collections = list(database.list_collection_names())
    print(f"\n✅ Collections ที่มีอยู่:")
    for col in collections:
        print(f"   - {col}")
    
    # นับจำนวน documents
    count = collection.count_documents({})
    print(f"\n📊 จำนวน documents ใน '{collection_name}': {count}")
    
except Exception as e:
    print(f"⚠️  ไม่สามารถตรวจสอบ: {e}")

# ===========================
# สรุป
# ===========================
print(f"\n{'='*70}")
print(f"🎉 การตั้งค่าเสร็จสมบูรณ์!")
print(f"{'='*70}")

if keyspace_name:
    print(f"\n✅ Keyspace: {keyspace_name}")
else:
    print(f"\n✅ Keyspace: (default - ไม่ได้ระบุ)")

print(f"✅ Collection: {collection_name}")
print(f"✅ Vector Dimension: 384")
print(f"✅ Similarity Metric: cosine")

print(f"\n📝 แก้ไขไฟล์ .env:")
print(f"{'='*70}")
if keyspace_name:
    print(f"ASTRA_DB_KEYSPACE={keyspace_name}")
else:
    print(f"# ไม่ต้องระบุ ASTRA_DB_KEYSPACE หรือปล่อยว่าง")
    print(f"# ASTRA_DB_KEYSPACE=")
print(f"{'='*70}")

print(f"\n🚀 พร้อมใช้งาน! สามารถรันคำสั่งนี้ได้เลย:")
print(f"   python allpeople_data.py")
print(f"\n{'='*70}")

