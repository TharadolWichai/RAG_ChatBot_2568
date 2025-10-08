import requests
from bs4 import BeautifulSoup
import json
from dotenv import load_dotenv
import os
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from astrapy import DataAPIClient
import uuid

load_dotenv()

def main():
    print("🚀 Starting Links scraping and AstraDB ingestion...")
    
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
    
    # Use services collection (รวมกับ contact และ students)
    collection_name = "services_embedding"
    try:
        # List existing collections first
        existing_collections = list(database.list_collection_names())
        print(f"📂 Existing collections: {existing_collections}")
        
        if collection_name in existing_collections:
            collection = database.get_collection(collection_name)
            print(f"📂 Using existing collection: {collection_name}")
            
            # Clear only links data (not all data)
            print("🗑️ Clearing existing links from collection...")
            try:
                # Delete only documents with category="links"
                delete_result = collection.delete_many({"metadata.category": "links"})
                print(f"🗑️ Deleted {delete_result.deleted_count if hasattr(delete_result, 'deleted_count') else 'existing'} link documents")
            except Exception as e:
                print(f"⚠️ Warning: Could not clear links: {e}")
                
        else:
            print(f"❌ Collection {collection_name} not found!")
            print("Please create the collection via AstraDB UI with vector support:")
            print("  - Collection Name: services_embedding")
            print("  - Vector Dimension: 384")
            print("  - Vector Metric: cosine")
            return False
            
    except Exception as e:
        print(f"❌ Failed to access collection: {e}")
        return False
    
    # Create links data manually (since website uses SPA/JavaScript)
    print("🌐 Creating links data from known services...")
    try:
        # Manual links data based on link.txt file - Updated with complete data
        links_data = [
            {
                "text": "ระบบจองห้องประชุม (Reservation)",
                "url": "https://appcs.kku.ac.th/rroom",
                "keywords": ["จองห้อง", "จองห้องประชุม", "booking", "meeting room", "reservation"]
            },
            {
                "text": "ระบบการจองใช้ห้องปฏิบัติการทางคอมพิวเตอร์",
                "url": "https://appcs.kku.ac.th/rlab",
                "keywords": ["จองห้อง", "ห้องแล็บ", "ปฏิบัติการ", "laboratory", "lab booking"]
            },
            {
                "text": "ระบบวิทยานิพนธ์",
                "url": "https://infocs.kku.ac.th/etis/",
                "keywords": ["วิทยานิพนธ์", "thesis", "research", "graduate"]
            },
            {
                "text": "ระบบจัดการโครงงานนักศึกษา",
                "url": "https://projectcs.kku.ac.th/e-project/",
                "keywords": ["โครงงาน", "นักศึกษา", "project", "student project"]
            },
            {
                "text": "ระบบจัดการโครงงานนักศึกษา(ใหม่)",
                "url": "https://eproject.computing.kku.ac.th/",
                "keywords": ["โครงงาน", "นักศึกษา", "project", "student project", "ใหม่", "new"]
            },
            {
                "text": "ระบบจัดการเอกสาร",
                "url": "https://infocs.kku.ac.th/uploadmanagement/",
                "keywords": ["จัดการเอกสาร", "เอกสาร", "document management", "upload"]
            },
            {
                "text": "ระบบแจ้งซ่อมออนไลน์",
                "url": "https://l.facebook.com/l.php?u=https%3A%2F%2Fsites.google.com%2Fkku.ac.th%2Fcprp%3Ffbclid%3DIwAR3L_IhcrTYn85pzjXreAiTqOf7Cu_iPlJP_UR2bgwRAH-o3cTKVhQ0bamo&h=AT3QUeLejw1fW8JFbyRi5G-H43oyG7Gpg2aLHmC-qoNlDyECmoebsAntCpo6bQg4VVPTClEIY374YTbO08_wHxijtiqiDuQ9gLXxV2jzmE3eJ7P6cd5nlzGgGwWKQXh3wSI",
                "keywords": ["แจ้งซ่อม", "ซ่อม", "maintenance", "repair", "online"]
            },
            {
                "text": "ระบบห้องสมุด",
                "url": "https://appcs.kku.ac.th/library/web",
                "keywords": ["ห้องสมุด", "library", "หนังสือ", "book"]
            },
            {
                "text": "ระบบเบิกจ่ายค่าสอนโครงการพิเศษ",
                "url": "https://tdbm.computing.kku.ac.th/login",
                "keywords": ["เบิกจ่าย", "ค่าสอน", "โครงการพิเศษ", "teaching", "special project"]
            },
            {
                "text": "ระบบภาระงานและการประเมินบุคลากร",
                "url": "http://10.199.36.11/cs-eoffice/",
                "keywords": ["ภาระงาน", "ประเมิน", "บุคลากร", "workload", "evaluation"]
            },
            {
                "text": "แบบฟอร์มลงทะเบียนข้อมูลส่วนตัวสำหรับบุคลากรใหม่",
                "url": "https://docs.google.com/forms/d/1mfySoDrIrYP3wrY91aZRuAcnGprwZVBeg614QCQK_xM/edit",
                "keywords": ["แบบฟอร์ม", "ลงทะเบียน", "บุคลากรใหม่", "registration", "new staff"]
            },
            {
                "text": "ระบบเบิกเงินสวัสดิการของผู้ปฏิบัติงานในมหาวิทยาลัยขอนแก่น",
                "url": "https://benefits.kku.ac.th/",
                "keywords": ["เบิกเงิน", "สวัสดิการ", "benefits", "welfare"]
            },
            {
                "text": "ระบบออกหนังสือรับรองบุคลากร",
                "url": "https://forms.hr.kku.ac.th/",
                "keywords": ["หนังสือรับรong", "บุคลากร", "certificate", "staff"]
            },
            {
                "text": "ระบบติดตามการขอกำหนดตำแหน่งบุคลากรสายวิชาการ",
                "url": "https://hr.kku.ac.th/tsap/",
                "keywords": ["ติดตาม", "กำหนดตำแหน่ง", "สายวิชาการ", "academic", "position"]
            },
            {
                "text": "ระบบติดตามการขอกำหนดตำแหน่งสายสนับสนุน",
                "url": "http://202.28.117.84/kgpkku/app",
                "keywords": ["ติดตาม", "กำหนดตำแหน่ง", "สายสนับสนุน", "support", "position"]
            },
            {
                "text": "ระบบลาอิเล็กทรอนิกส์",
                "url": "https://office.kku.ac.th/",
                "keywords": ["ลา", "อิเล็กทรอนิกส์", "leave", "electronic"]
            },
            {
                "text": "การขอกำหนดตำแหน่งทางวิชาการ (ระดับอุดมศึกษา)",
                "url": "https://hr.kku.ac.th/wphrdkku/?page_id=2732",
                "keywords": ["กำหนดตำแหน่ง", "วิชาการ", "อุดมศึกษา", "academic", "higher education"]
            },
            {
                "text": "การขอกำหนดตำแหน่งสูงขึ้น",
                "url": "https://hr.kku.ac.th/wphrdkku/?page_id=1506",
                "keywords": ["กำหนดตำแหน่ง", "สูงขึ้น", "promotion", "position"]
            },
            {
                "text": "สิทธิการลาของบุคลากร",
                "url": "https://hr.kku.ac.th/wphrdkku/?page_id=1522",
                "keywords": ["สิทธิ", "ลา", "บุคลากร", "leave rights", "staff"]
            },
            {
                "text": "แบบฟอร์มบันทึกข้อความ-ขออนุมัติไปต่างประเทศในระหว่างลาพักผ่อน",
                "url": "https://docs.google.com/document/d/1_nfGdXl6pJi6xMN9DbvFuGAuc5n9jmNK/edit?usp=share_link&ouid=111733404484503575729&rtpof=true&sd=true",
                "keywords": ["แบบฟอร์ม", "ต่างประเทศ", "ลาพักผ่อน", "travel abroad", "vacation"]
            },
            {
                "text": "ขั้นตอนการบันทึกการลาในระบบ-ikku",
                "url": "https://drive.google.com/file/d/1QdYr8fF-02TvHUjL4qKwzQcPpbwWt6Bk/view?usp=sharing",
                "keywords": ["ขั้นตอน", "บันทึกการลา", "ikku", "leave recording", "tutorial"]
            },
            {
                "text": "แนวปฏิบัติการขอรับเงิน-และเบิกจ่ายเงินสวัสดิการผู้ปฏิบัติงาน",
                "url": "https://drive.google.com/file/d/1yYs7ZjdfF9kBuIsG9EdrINWmu2-KP8_B/view?usp=sharing",
                "keywords": ["แนวปฏิบัติ", "เบิกจ่าย", "สวัสดิการ", "welfare", "guidelines"]
            },
            {
                "text": "ใบสมัครกองทุนสำรองเลี้ยงชีพ-มข.",
                "url": "https://drive.google.com/file/d/1qoEiibDA2zknhucTRIwPFcl3pShIXLK7/view?usp=sharing",
                "keywords": ["ใบสมัคร", "กองทุน", "เลี้ยงชีพ", "provident fund", "application"]
            },
            {
                "text": "แบบแจ้งความประสงค์สะสมเงินเพิ่ม-กองทุนสำรองเลี้ยงชีพ-มข.",
                "url": "https://drive.google.com/file/d/1FebTbAP41LjDDxq7m_CJ8FeUiO-jet-R/view?usp=share_link",
                "keywords": ["แจ้งความประสงค์", "กองทุน", "เลี้ยงชีพ", "provident fund", "savings"]
            },
            {
                "text": "กบม.9-2566-การกำหนดกรอบระดับตำแหน่งพนักงานมหาวิทยาลัย-ข้าราชการ",
                "url": "https://drive.google.com/file/d/1Oc5pfvWpYwAJg_I-XHGSNGT5gXtf89NP/view?usp=sharing",
                "keywords": ["กรอบ", "ตำแหน่ง", "พนักงาน", "ข้าราชการ", "position framework"]
            },
            {
                "text": "แบบฟอร์มขออนุญาตลงเวลาเนื่องจากเหตุสุดวิสัย (สายสนับสนุน)",
                "url": "https://drive.google.com/file/d/1UOpLvTi0xXNSSPuwQM8IjeorT3A3Ytte/view",
                "keywords": ["แบบฟอร์ม", "ลงเวลา", "เหตุสุดวิสัย", "สายสนับสนุน", "time recording"]
            },
            {
                "text": "เรียนรู้การใช้งานด้วยตัวเอง (Self Service) HCM",
                "url": "https://hr2.kku.ac.th/?page_id=8803",
                "keywords": ["Self Service", "HCM", "เรียนรู้", "tutorial", "self learning"]
            },
            {
                "text": "เข้าสู่ระบบ HCM (Self service)",
                "url": "https://login-iaajtj.fa.ocs.oraclecloud.com/oam/server/obrareq.cgi?encquery%3D%2Fst8%2BhaIc%2FuMG8VSYdSjEWGAtLJahfjbyAOTPL9Y6FTRboAhps5DsOO1S1KW8RAoWHfTpo%2FJO%2Fv9VC2yWVX4j75hp9SUhmPxziY2j3Kl961wrl%2FB4dvC8oTNPoWyskAvLJh83l%2FzjpPWCtkiBaWln99E%2B4gJJO9RWY6u1lK%2B4oQmSLX3C1z5gdx%2Bt%2FQ5q6JFAZqR1tUs9fgSvwdHlk8id81c4y%2BYU5svY7MDMGMoYshB%2Fp1C7pB60HSVAn6duCjj4LHMrdn9WxmUR6vkho1YahWRzAx0gSSjWTrlzD1CPUrexVAlErzs4UkTXaWEaeXgbfXTtx53Hud6xIhjeVVPFL9arEPjh%2FOdVxMLLbHHrm2CWQpYCFwCjjLxDtO2luuwygDU%2Bq%2F2ZOvTZFIFgucunJMmB3zwV69DwR2juSrHGrfBEQfuRxBD3O3VBbxc07%2BoEZHFlU%2F7okHTfMdqpPpKcU0C%2F%2FkRWgSVERcX5GW7RwZCKZ5jWNc5xZtNPWBIj6t%2FU6ZZFUQwevpSO%2Bbo1m2wDUiHns9ciedKxm4rxM47UToHybkb4jvfaoP8hC%2FCI9sWyTwrWuX%2FSsxGkGJCIrINSYqu2tO7hFF3XcOvuvrjc25%2FmUuayfwuhUi1yMjywXZeUDdmYc6jpo%2F3Sygq0SgzM0VaNDReMjtou9szYvaf%2FWEQl%2FzmUtWEdGTaZxBKh9U8H4dC5YWuNvYfVtnyNj0O1smhz7JIdWJpsstOFMU0HYMPbzfi7kDWysiZ3KoYH%2B4N2Xv3kSQhN2b%2BNdlXaXB%2BtISpJU9pYimVwjOy6sRuzU6UjGU9nQM0W6DogHfOaRH1mHMiBImsbAvYQo08k3Tdhhl4yLCDpbhdw04xu6NJUOrDv3HvPgGMPubxIAnqfIZ%2Fn%2FL6MfCuaO6tmsM3ERCaGekPV1oNHsWC5NukPMDq1z6IUFUQg8InPdGnLIosCDvSlMaG8y6XSgsR2OKRbwKkBwt3WAuxzssVAAncd0ZRCHnH%2BF%2FKP4%2BiBPE6KrCd9muQu9dovl24mF2F%2FyLbOkJX05y87bcPPbfkQkQF%2FbNi4guWdXnZF6J6VOXiCOtJtcUzuMuFaZMckE%2BBM3cAV%2Bm%2BxHruniCFSSWuf9U4UZ%2B5jina2eCMEpTbzvJBJ4MQuBHLShsNlwPfVoVFxsyiG%2BQBw93HJJ2xQiOvqjBJYkiMPyfYV7aqD8gKiFeAUhx9mkANh5iX%2FRT1IwmGkOfStX581LIrinS2IUjMOe7Qa%2FVp9jY7ojbhdvhMe8mcl0hbsrexx0yAG8Pa6H94qkH%2BxXAcZ%2F62c3d3O4nI3CMnGLyTagc3G15%2BPC85v2urVf%2BLeUjW6EDuJtq4lMUdGs15YJJ1%2FumlUuM86IRwJzqiTwQWVPCti8oNpDqjGWHBT5p7o2pUZ35iZzb9s%2BRpsEno0W9LBzVCy3UsBUSZe1GvUK2thHJq0%2BYN8YSCS%2Bge%2B3t4pO4f73Z3hlXLKWF087J%2B2Oz2%2BmGnJBMXW3xLtSWLOSsaESDBrtmjl%2FoJpK6JfG6y9h9IhOlXhHi6ALeXJXV55ai7juZaHs0KsAxzQP9ZLbS%2BXwKbLN7CpMbAfjRmZ2qdjtFET84RhGI26duhKrB5xxg%2Fo1KGOcp5dPBtbMdmSXLL%2FtM%3D%20agentid%3DOraFusionApp_11AG%20ver%3D1%20crmethod%3D2%26cksum%3Dd065cdad14040cb92a6755312fa6b01774460f21&ECID-Context=1.0069T0X0vwsDg%5EK6yVrY6G00DeTw0000At%3BkXjE",
                "keywords": ["HCM", "Self service", "เข้าสู่ระบบ", "login", "Oracle"]
            },
            {
                "text": "การใช้งานระบบ HCM โมดูลต่างๆ",
                "url": "https://hr2.kku.ac.th/?page_id=8803",
                "keywords": ["HCM", "โมดูล", "การใช้งาน", "modules", "usage"]
            },
            {
                "text": "คลิปวิดีโอ การลงเวลาปฏิบัติงานรูปแบบ Time Card",
                "url": "https://www.youtube.com/watch?v=vqxiRaa9njE",
                "keywords": ["คลิป", "วิดีโอ", "Time Card", "ลงเวลา", "tutorial"]
            },
            {
                "text": "คลิปวิดีโอ การลงเวลาปฏิบัติงานรูปแบบ App Sheet",
                "url": "https://www.youtube.com/watch?v=vqxiRaa9njE",
                "keywords": ["คลิป", "วิดีโอ", "App Sheet", "ลงเวลา", "tutorial"]
            },
            {
                "text": "คลิปวิดีโอ การขอเปลี่ยนแปลงเวลา",
                "url": "https://drive.google.com/file/d/18pnxRk39-eK1dj0qI8ApoiBfTjnhe4oW/view",
                "keywords": ["คลิป", "วิดีโอ", "เปลี่ยนแปลงเวลา", "time change", "tutorial"]
            },
            {
                "text": "คลิปวิดีโอ การบันทึกข้อมูลการลา",
                "url": "https://drive.google.com/file/d/1vUJn9XrnqCwHOJumLZYZbNEsxZd42_5V/view?t=48",
                "keywords": ["คลิป", "วิดีโอ", "บันทึกการลา", "leave recording", "tutorial"]
            },
            {
                "text": "คลิปวิดีโอ การบันทึกขออนุญาตไปต่างประเทศ (กรณีลาและขออนุญาตเดินทางไปต่างประเทศระหว่างลา โดยจะต้องบันทึกการลาปกติก่อน)",
                "url": "https://drive.google.com/file/d/1x3xPUTvFVfd7lGM1pOMou8ckQF1uPI-2/view",
                "keywords": ["คลิป", "วิดีโอ", "ต่างประเทศ", "ลา", "travel abroad", "tutorial"]
            },
            {
                "text": "ขั้นตอนการขออนุญาตไปต่างประเทศ (กรณีลากิจส่วนตัว ลาพักผ่อน วันหยุดราชการ วันหยุดนักขัตฤกษ์)",
                "url": "https://drive.google.com/file/d/1o0HiZ9kQ8Ldb0KkreL-HezyYk4SSi176/view",
                "keywords": ["ขั้นตอน", "ต่างประเทศ", "ลากิจ", "ลาพักผ่อน", "travel abroad"]
            },
            {
                "text": "ขั้นตอนการขอลาพักผ่อนไปต่างประเทศ โดยบุคลากร",
                "url": "https://drive.google.com/file/d/1Q-PP3rf-qKiFjGnA2BzlcEH5EhAahpf3/view",
                "keywords": ["ขั้นตอน", "ลาพักผ่อน", "ต่างประเทศ", "บุคลากร", "vacation abroad"]
            },
            {
                "text": "แบบฟอร์มขอขยายเวลาแก้เกรด I",
                "url": "https://docs.google.com/document/d/1m3psh4A0V4f_2buii9KDScdnlAN-ihKo/edit?usp=sharing&ouid=111733404484503575729&rtpof=true&sd=true",
                "keywords": ["แบบฟอร์ม", "ขยายเวลา", "แก้เกรด", "grade extension", "form"]
            },
            {
                "text": "แบบฟอร์มขอชี้แจงส่งเกรดช้า",
                "url": "https://docs.google.com/document/d/12polM7WoXwh9xEBTgx46GOVQpuQ6uC3x/edit?usp=sharing&ouid=111733404484503575729&rtpof=true&sd=true",
                "keywords": ["แบบฟอร์ม", "ชี้แจง", "ส่งเกรดช้า", "late grade", "explanation"]
            },
            {
                "text": "แบบฟอร์มขอเปลี่ยนแปลงการคุมสอบ",
                "url": "https://docs.google.com/document/d/1cVq2vCizT8zUtAVsVRd9u_EFuGev5q6p/edit?usp=sharing&ouid=111733404484503575729&rtpof=true&sd=true",
                "keywords": ["แบบฟอร์ม", "เปลี่ยนแปลง", "คุมสอบ", "exam supervision", "change"]
            },
            {
                "text": "แบบฟอร์มขอเปลี่ยนแปลงเกรด",
                "url": "https://docs.google.com/document/d/1p3k28ME7zm3UdAQCapsb5NVYiGKTr2yG/edit?usp=sharing&ouid=111733404484503575729&rtpof=true&sd=true",
                "keywords": ["แบบฟอร์ม", "เปลี่ยนแปลง", "เกรด", "grade change", "form"]
            },
            {
                "text": "แบบฟอร์มชี้แจงสาเหตุการให้เกรด I",
                "url": "https://docs.google.com/document/d/1DFYObCyt6yfTKbKJdfb3sNwlASY6rmVE/edit?usp=sharing&ouid=111733404484503575729&rtpof=true&sd=true",
                "keywords": ["แบบฟอร์ม", "ชี้แจง", "เกรด I", "incomplete grade", "explanation"]
            },
            {
                "text": "แบบฟอร์มขออนุมัติถ่ายเอกสาร",
                "url": "https://api.computing.kku.ac.th//storage/documents/2023-12-5-1703735229-undefined.pdf",
                "keywords": ["แบบฟอร์ม", "อนุมัติ", "ถ่ายเอกสาร", "document copy", "approval"]
            },
            {
                "text": "การยื่นจดสิทธิบัตรการประดิษฐ์/อนุสิทธิบัตร",
                "url": "https://ip.kku.ac.th/how-to-patent/1902",
                "keywords": ["สิทธิบัตร", "การประดิษฐ์", "อนุสิทธิบัตร", "patent", "invention"]
            },
            {
                "text": "การยื่นจดสิทธิบัตรการออกแบบผลิตภัณฑ์",
                "url": "https://ip.kku.ac.th/how-to-patent/1991",
                "keywords": ["สิทธิบัตร", "ออกแบบผลิตภัณฑ์", "product design", "patent"]
            },
            {
                "text": "การยื่นคำขอแจ้งข้อมูลลิขสิทธิ์",
                "url": "https://ip.kku.ac.th/copyright/1984",
                "keywords": ["ลิขสิทธิ์", "copyright", "แจ้งข้อมูล", "information"]
            }
        ]
        
        docs = []
        link_count = 0
        
        for link_data in links_data:
            link_text = link_data["text"]
            link_url = link_data["url"]
            keywords = link_data["keywords"]
            
            # Create comprehensive content for better search
            content_parts = [
                f"ชื่อลิงก์: {link_text}",
                f"URL: {link_url}",
                f"ประเภท: ลิงก์บริการคณะคอมพิวเตอร์"
            ]
            
            if keywords:
                content_parts.append(f"คำสำคัญ: {', '.join(keywords)}")
            
            content = "\n".join(content_parts)
            
            # Create metadata with category
            metadata = {
                "link_text": link_text,
                "url": link_url,
                "type": "service_link",
                "keywords": keywords,
                "category": "links",  # Category สำหรับแยกประเภท
                "subcategory": "general_services"
            }
            
            docs.append(Document(page_content=content, metadata=metadata))
            link_count += 1
            print(f"✅ Created link {link_count}: {link_text}")
        
        print(f"📊 Total links created: {len(docs)}")
        
    except Exception as e:
        print(f"❌ Failed to create links data: {e}")
        return False
    
    if len(docs) == 0:
        print("🚨 No links found for vector creation")
        return False

    print(f"📝 Processed {len(docs)} link documents")

    # Split documents (though links are usually short, this ensures consistency)
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"📄 Created {len(chunks)} chunks")

    # Initialize embeddings
    print("🧠 Initializing embeddings model...")
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Generate embeddings and insert to AstraDB
    print("💾 Inserting links data into AstraDB...")
    
    documents_to_insert = []
    for i, chunk in enumerate(chunks):
        # Generate embedding
        vector = embedding.embed_query(chunk.page_content)
        
        # Prepare document for insertion
        doc = {
            "_id": str(uuid.uuid4()),
            "content": chunk.page_content,
            "$vector": vector,
            "metadata": chunk.metadata
        }
        documents_to_insert.append(doc)
        
        if i % 5 == 0:
            print(f"📊 Processed {i+1}/{len(chunks)} chunks...")
    
    # Insert all documents
    try:
        result = collection.insert_many(documents_to_insert)
        print(f"✅ Successfully inserted {len(result.inserted_ids)} link documents into AstraDB!")
    except Exception as e:
        print(f"❌ Failed to insert documents: {e}")
        return False
    
    # Verify insertion
    try:
        count = collection.count_documents({})
        print(f"🔍 Total documents in links collection: {count}")
    except Exception as e:
        print(f"⚠️ Could not verify document count: {e}")
    
    print("🎉 Links scraping and AstraDB ingestion completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
