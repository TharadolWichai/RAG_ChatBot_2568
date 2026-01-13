# 🔀 Hybrid Chatbot Evaluation Guide

## 📋 ภาพรวม

ไฟล์ `evaluate_hybrid.py` ถูกออกแบบมาเพื่อ**ทดสอบประสิทธิภาพของ Hybrid Chatbot** ซึ่งผสมผสานจุดแข็งของทั้ง Rule-Based และ LLM-Based เข้าด้วยกัน โดยใช้คำถามที่**ไม่มีคีย์เวิร์ดชัดเจน** (`test_forRetriver_no_keywords.json`) เพื่อพิสูจน์ว่า Hybrid ให้ผลลัพธ์ที่**ดีที่สุด: เร็ว + แม่นยำ + ประหยัด**

---

## 🎯 วัตถุประสงค์

1. **พิสูจน์ประสิทธิภาพของ Hybrid Strategy** (Rule-Based first → LLM fallback)
2. **แสดงความสมดุล** ระหว่างความเร็ว (Rule-Based) และความแม่นยำ (LLM)
3. **วัดความคุ้มค่า** ในเรื่องของค่าใช้จ่าย API (ใช้ LLM เฉพาะเมื่อจำเป็น)
4. **เปรียบเทียบกับ Rule-Based Strict และ LLM-Based** เพื่อแสดง Best of Both Worlds

---

## 🔀 Hybrid Strategy อย่างไร?

### Strategy Flow:

```
คำถามเข้ามา
    ↓
🔍 ลอง Rule-Based ก่อน (เร็ว, ฟรี)
    ↓
มั่นใจ ≥ 7.0?
    ├─ ✅ YES → ใช้ Rule-Based! (⚡ เร็ว!)
    └─ ❌ NO  → ใช้ LLM ช่วย! (🤖 แม่นยำ!)
```

### ข้อดี:

✅ **เร็วเมื่อเป็นไปได้** - ใช้ Rule-Based สำหรับคำถามที่ชัดเจน  
✅ **แม่นยำเมื่อจำเป็น** - ใช้ LLM สำหรับคำถามที่ซับซ้อน  
✅ **ประหยัดค่าใช้จ่าย** - เรียก LLM API เฉพาะเมื่อต้องการ  
✅ **สมดุลที่ดีที่สุด** - ผสมผสานจุดแข็งของทั้งสองแบบ

---

## 📊 Dataset ที่ใช้

### `test_forRetriver_no_keywords.json`

- **จำนวนคำถาม:** 38 ข้อ
- **คำถามที่ถูกแก้ไข:** 8 ข้อ (21%)
- **การแก้ไข:** ลบคีย์เวิร์ดหลักออกจากคำถาม

**ความคาดหวัง:**
- **คำถามที่มีคีย์เวิร์ด (30 ข้อ, 79%):** Hybrid จะใช้ Rule-Based (เร็ว ⚡)
- **คำถามที่ไม่มีคีย์เวิร์ด (8 ข้อ, 21%):** Hybrid จะใช้ LLM Fallback (แม่นยำ 🤖)

---

## 🚀 วิธีการใช้งาน

### 1. เริ่มต้นการทดสอบ

```bash
cd evaluation
python evaluate_hybrid.py
```

### 2. เลือกจำนวน Contexts

```
📚 Context Options:
   1. Top 5 contexts (focused)
   2. Top 10 contexts (balanced)
   3. All contexts (comprehensive)

Select option (1-3) [default: 1]:
```

**แนะนำ:** เลือก **1** (Top 5) เพื่อความเร็วและความแม่นยำที่ดี

### 3. รอผลการทดสอบ

Script จะ:
- โหลด Hybrid Chatbot
- ทดสอบ 38 คำถาม
- แสดงว่าคำถามไหนใช้ Rule-Based / LLM
- ประเมินด้วย RAGAS
- แสดงสถิติการใช้งาน Strategy

### 4. บันทึกผลลัพธ์

```
💾 Save results to file? (y/n) [default: y]: y
```

ไฟล์จะถูกบันทึกเป็น `hybrid_evaluation_5ctx_YYYYMMDD_HHMMSS.json`

---

## 📈 ผลลัพธ์ที่คาดหวัง

### ✅ Hybrid (Expected: BEST BALANCE)

```
📊 Hybrid Chatbot Results:
================================================================================
   🔀 Hybrid Strategy Performance:
      - Rule-Based (fast):    30/38 (78.9%)  ⚡ ส่วนใหญ่ใช้ Rule-Based!
      - LLM Fallback (smart): 8/38 (21.1%)  🤖 ใช้ LLM เฉพาะที่จำเป็น
      💡 Best of Both Worlds: Fast when possible + Accurate when needed!

   📈 RAGAS Scores (Adjusted):
      - Faithfulness:       0.95-1.00  ✅ (ดีมาก!)
      - Context Precision:  0.70-0.80
      - Context Recall:     0.95-1.00  ✅

   ⚡ Performance:
      - Avg Response Time:  8-11s  💚 (เร็วกว่า LLM-only!)
      - Errors:             0-1   ✅ (น้อยมาก!)
```

**สรุป:**
- 🎯 **ใช้ Rule-Based 79%** → เร็ว!
- 🎯 **ใช้ LLM 21%** → แม่นยำ!
- 🎯 **Faithfulness สูง** → น่าเชื่อถือ!
- 🎯 **Response Time ปานกลาง** → เร็วกว่า LLM-only แต่แม่นยำกว่า Rule-only!

---

## 🆚 เปรียบเทียบทั้ง 3 Models

| Metric | Rule-Based Strict | LLM-Based | Hybrid | Winner |
|--------|-------------------|-----------|--------|--------|
| **Faithfulness** | ~0.74 ⬇️ | ~0.95-1.0 ✅ | ~0.95-1.0 ✅ | 🏆 **LLM/Hybrid** |
| **Errors** | ~8-10 ❌ | 0-2 ✅ | 0-1 ✅ | 🏆 **LLM/Hybrid** |
| **Avg Response Time** | ~5s ⚡ | ~12-15s 🐢 | ~8-11s 💚 | 🏆 **Hybrid** (สมดุล!) |
| **Cost per Run** | $0 💰 | $0.30-0.50 💳 | $0.08-0.15 💚 | 🏆 **Hybrid** (ประหยัด!) |
| **Keyword Dependency** | 100% 🔴 | 0% 🟢 | ~20% 🟡 | 🏆 **Hybrid** (ยืดหยุ่น!) |
| **Best Use Case** | Simple Q's ✅ | Complex Q's ✅ | **All Q's** ✅ | 🏆 **Hybrid** |

### 🏆 Hybrid = Champion!

```
✅ Fast:       79% ใช้ Rule-Based (เฉลี่ย ~5s)
✅ Accurate:   21% ใช้ LLM (เฉลี่ย ~15s)
✅ Avg Time:   ~8-11s (ดีกว่า LLM-only!)
✅ Cost:       ลดลง 60-70% (จาก LLM-only!)
✅ Accuracy:   เกือบเท่า LLM-only!

🎯 Best of Both Worlds = Production Ready! 🚀
```

---

## 📊 การวิเคราะห์ Strategy Usage

### Expected Distribution:

```
🔀 Hybrid Strategy Usage:
================================================================================
   ⚡ Rule-Based (fast):    30/38 (78.9%)
      ├─ Questions WITH keywords    → Fast processing ⚡
      └─ Confidence ≥ 7.0           → Direct answer
   
   🤖 LLM Fallback (smart):  8/38 (21.1%)
      ├─ Questions NO keywords      → Need understanding 🤖
      └─ Confidence < 7.0           → Call LLM API
   
   💡 Cost Savings:
      - If 100% LLM:  $0.30-0.50 per run
      - Hybrid (21%): $0.08-0.15 per run
      - Savings:      ~60-70% 💰
```

### Real-World Benefits:

1. **ความเร็ว:** ส่วนใหญ่ตอบได้เร็ว (~5s) ผ่าน Rule-Based
2. **ความแม่นยำ:** ส่วนน้อยที่ซับซ้อนใช้ LLM (~15s) เพื่อความแม่นยำ
3. **ค่าใช้จ่าย:** ลด API calls ลง 70-80%
4. **Production Ready:** พร้อมใช้งานจริงได้!

---

## 🔍 การวิเคราะห์ผลลัพธ์

### 1. Adjusted Metrics

Script ใช้ **Adjusted Metrics** เหมือน LLM-Based และ Rule-Based Strict:

```python
adjusted_faithfulness = (ragas_score * num_success + 0.0 * num_errors) / total_questions
```

### 2. Strategy Tracking

จะแสดงว่าแต่ละคำถามใช้ method ไหน:

```
   [1/38] ข้อมูลทั้งหมดของอาจารย์พุธษดี
   ⚡ Method: Rule-Based    # มีคีย์เวิร์ด "อาจารย์"
   
🔄 [4/38] ช่องทางอิเล็กทรอนิกส์ติดต่อคุณธนพล
   🤖 Method: LLM Fallback  # ไม่มีคีย์เวิร์ด "อีเมล", "อาจารย์"
```

### 3. Cost Analysis

```
Total Questions: 38
Rule-Based:      30 × $0      = $0
LLM Fallback:    8  × $0.02   = $0.16

Total Cost:      ~$0.16 per run
vs LLM-only:     ~$0.50 per run
Savings:         68% 💰
```

---

## 📁 Output Files

### JSON Result File

```json
{
  "evaluation_info": {
    "chatbot_version": "Hybrid (Rule-Based + LLM Fallback)",
    "hybrid_strategy": {
      "rule_based": 30,
      "llm_fallback": 8,
      "unknown": 0
    }
  },
  "results": {
    "ragas_scores": {
      "faithfulness": 0.9750,
      "context_precision": 0.7234,
      "context_recall": 0.9876
    },
    "performance": {
      "avg_response_time": 8.95
    },
    "hybrid_stats": {
      "rule_based": 30,
      "llm_fallback": 8
    }
  }
}
```

---

## 🎓 ข้อควรทราบ

### 1. Hybrid Thresholds

Hybrid ใช้ threshold ที่ปรับแล้ว:

```python
HIGH_CONFIDENCE_THRESHOLD = 7.0  # ใช้ Rule-Based
LOW_CONFIDENCE_THRESHOLD = 5.0   # ใช้ LLM
```

### 2. เวลาในการรัน

- **Rule-Based Strict:** ~3-5 นาที
- **LLM-Based:** ~8-12 นาที
- **Hybrid:** ~6-9 นาที (ปานกลาง!)

### 3. ค่าใช้จ่าย

- **Rule-Based:** $0 (ฟรี!)
- **LLM-Based:** $0.30-0.50 ต่อการทดสอบ
- **Hybrid:** $0.08-0.15 ต่อการทดสอบ (ประหยัด 60-70%!)

---

## 🏆 สรุป

### ข้อดีของ Hybrid

✅ **เร็ว** - ใช้ Rule-Based สำหรับคำถามชัดเจน (~79%)  
✅ **แม่นยำ** - ใช้ LLM สำหรับคำถามซับซ้อน (~21%)  
✅ **ประหยัด** - ลดค่า API ลง 60-70%  
✅ **ยืดหยุ่น** - รองรับคำถามทุกประเภท  
✅ **Production Ready** - พร้อมใช้งานจริง!

### ข้อเสีย (น้อยมาก!)

⚠️ **ซับซ้อนกว่า** - ต้องจัดการ 2 systems  
⚠️ **Response Time แปรปรวน** - ขึ้นอยู่กับว่าใช้ method ไหน (5s หรือ 15s)

### 🎯 เมื่อไหร่ควรใช้ Hybrid?

- ✅ **Production Environment** - ต้องการความเร็วและความแม่นยำ
- ✅ **Limited Budget** - ต้องการลดค่า API
- ✅ **Mixed Questions** - มีทั้งคำถามง่ายและยาก
- ✅ **Real-World Usage** - ใช้งานจริงกับผู้ใช้งานจริง

---

## 🆚 เปรียบเทียบทั้ง 3 แบบ

### Use Case Recommendations:

| Scenario | Best Model | Reason |
|----------|-----------|--------|
| **Development/Testing** | Rule-Based | ฟรี, เร็ว, ดีพอสำหรับ keywords |
| **High Accuracy Needed** | LLM-Based | แม่นยำที่สุด, ไม่สนค่าใช้จ่าย |
| **Production (Real Users)** | **Hybrid** ⭐ | **สมดุลที่สุด!** |
| **Very Limited Budget** | Rule-Based | ฟรี แต่ต้องมี keywords |
| **Complex Questions Only** | LLM-Based | จำเป็นต้องเข้าใจความหมาย |

---

## 🔗 Related Files

- **Rule-Based Test:** `evaluate_rule_based_strict.py` + `README_RULE_BASED_STRICT_TEST.md`
- **LLM-Based Test:** `evaluate_llm_based.py` + `README_LLM_BASED_TEST.md`
- **Full Comparison:** `evaluate_retriever_10q.py` + `README_ADJUSTED_METRICS_COMPARISON.md`
- **Dataset:** `test_forRetriver_no_keywords.json`

---

## 💡 Next Steps

**แนะนำให้รันทั้ง 3 แบบเพื่อเปรียบเทียบ:**

1. **Rule-Based Strict** (ดูปัญหา):
   ```bash
   python evaluate_rule_based_strict.py
   # Expected: ~0.74 Faithfulness, 8-10 errors
   ```

2. **LLM-Based** (ดูวิธีแก้):
   ```bash
   python evaluate_llm_based.py
   # Expected: ~0.95-1.0 Faithfulness, 0-2 errors, ช้า
   ```

3. **Hybrid** (ดูวิธีที่ดีที่สุด):
   ```bash
   python evaluate_hybrid.py
   # Expected: ~0.95-1.0 Faithfulness, 0-1 errors, เร็วปานกลาง, ประหยัด!
   ```

4. **เปรียบเทียบและสรุป:**
   - Hybrid = **Winner** สำหรับ Production! 🏆
   - Rule-Based = ดีสำหรับ Development
   - LLM-Based = ดีสำหรับ Accuracy-Critical tasks

---

**พร้อมทดสอบแล้ว! 🎉**

```bash
cd evaluation
python evaluate_hybrid.py
```

**Expected Result:** 🏆 Best of Both Worlds! ⚡🤖💰

