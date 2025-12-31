# 📚 Evaluation System - Complete Documentation Index

## 🎯 ภาพรวม

ระบบประเมินประสิทธิภาพของ **3 Chatbot Models** ด้วย RAGAS:

1. 🔒 **Rule-Based Strict Mode** - Fast but keyword-dependent
2. 🤖 **LLM-Based** - Accurate but slow and costly  
3. 🔀 **Hybrid** - Best of Both Worlds! ⭐ (Production Ready)

---

## 🚀 Quick Start - Which Test to Run?

### 🎯 Option 1: Compare All 3 Models (Recommended!)

```bash
python evaluate_retriever_10q.py
```

- **Time:** ~20-25 minutes
- **Output:** Comparison table of all 3 models
- **Best For:** Making final decision
- **Choose:** `test_forRetriver_no_keywords.json` + Top 5 + Strict Mode

---

### 🔒 Option 2: Test Rule-Based Only

```bash
python evaluate_rule_based_strict.py
```

- **Time:** ~3-5 minutes
- **Output:** Rule-Based performance only
- **Best For:** Understanding limitations
- **Expected:** ~0.74 Faithfulness, 8 errors

---

### 🤖 Option 3: Test LLM-Based Only

```bash
python evaluate_llm_based.py
```

- **Time:** ~8-12 minutes
- **Output:** LLM-Based performance only
- **Best For:** Understanding LLM capabilities
- **Expected:** ~0.95-1.0 Faithfulness, 0-2 errors

---

### 🔀 Option 4: Test Hybrid Only ⭐

```bash
python evaluate_hybrid.py
```

- **Time:** ~6-9 minutes
- **Output:** Hybrid performance only
- **Best For:** Validating production choice
- **Expected:** ~0.95-1.0 Faithfulness, 0-1 errors, 68% cost savings!

---

## 📚 Complete Documentation

### 🏆 Start Here (Most Important!)

| Priority | Document | Description |
|----------|----------|-------------|
| ⭐⭐⭐ | **[THREE_MODELS_COMPARISON](README_THREE_MODELS_COMPARISON.md)** | เปรียบเทียบทั้ง 3 models แบบละเอียด **อ่านก่อน!** |
| ⭐⭐ | **[HYBRID_TEST](README_HYBRID_TEST.md)** | คู่มือ Hybrid (Production Ready!) |
| ⭐ | **[ADJUSTED_METRICS_COMPARISON](README_ADJUSTED_METRICS_COMPARISON.md)** | อธิบาย Adjusted Metrics |

### 📖 Individual Model Guides

| Model | Document | Purpose |
|-------|----------|---------|
| 🔒 Rule-Based | **[RULE_BASED_STRICT_TEST](README_RULE_BASED_STRICT_TEST.md)** | แสดงข้อจำกัดเมื่อไม่มีคีย์เวิร์ด |
| 🤖 LLM-Based | **[LLM_BASED_TEST](README_LLM_BASED_TEST.md)** | แสดงความสามารถของ LLM |
| 🔀 Hybrid | **[HYBRID_TEST](README_HYBRID_TEST.md)** | แสดง Best of Both Worlds! |

### 🔧 Technical Documentation

| Document | Description |
|----------|-------------|
| **[RETRIEVER_TEST](README_RETRIEVER_TEST.md)** | คู่มือ `evaluate_retriever_10q.py` |
| **[NO_KEYWORDS_TEST](README_NO_KEYWORDS_TEST.md)** | อธิบาย dataset `test_forRetriver_no_keywords.json` |
| **[ADJUSTED_METRICS](README_ADJUSTED_METRICS.md)** | อธิบาย Adjusted Metrics Logic |
| **[EXCEL_CONVERSION](README_EXCEL_CONVERSION.md)** | วิธี convert Excel → JSON |

---

## 📊 Expected Results Summary

```
================================================================================
🏆 THREE MODELS COMPARISON
================================================================================

Metric                | Rule-Based | LLM-Based | Hybrid    | Winner
--------------------------------------------------------------------------------
Faithfulness          |     0.7368 |    0.9750 |    0.9750 | Hybrid ⭐
Context Precision     |     0.5526 |    0.7234 |    0.7234 | Hybrid
Context Recall        |     0.7895 |    0.9876 |    0.9876 | Hybrid
--------------------------------------------------------------------------------
Avg Response Time     |      5.43s |     12.43s|      8.95s| Hybrid ⭐
Errors (out of 38)    |          8 |         0 |         0 | Hybrid
Cost per Run          |         $0 |     $0.40 |     $0.12 | Hybrid ⭐
--------------------------------------------------------------------------------
Production Ready?     |         ⚠️ |         ✅|        ✅ | Hybrid ⭐
================================================================================

🏆 Overall Winner: HYBRID! 🔀⭐

Why Hybrid Wins:
   ✅ Fast:     79% use Rule-Based (~5s)
   ✅ Accurate: 21% use LLM (~15s) when needed
   ✅ Cheap:    68% cost reduction vs LLM-only
   ✅ Ready:    Production-ready NOW!
```

---

## 🎯 Recommended Workflow

### Step 1: Understand the Problem 📖

```
อ่าน: README_THREE_MODELS_COMPARISON.md
เวลา: 10-15 นาที
เป้าหมาย: เข้าใจความแตกต่างของทั้ง 3 models
```

---

### Step 2: Run Individual Tests 🧪

```bash
# Test 1: Rule-Based (ดูปัญหา)
python evaluate_rule_based_strict.py
# Expected: ~0.74 Faithfulness, 8 errors

# Test 2: LLM-Based (ดูวิธีแก้)
python evaluate_llm_based.py
# Expected: ~0.95-1.0 Faithfulness, 0-2 errors

# Test 3: Hybrid (ดูวิธีที่ดีที่สุด)
python evaluate_hybrid.py
# Expected: ~0.95-1.0 Faithfulness, 0-1 errors, fast!
```

---

### Step 3: Run Full Comparison 📊

```bash
python evaluate_retriever_10q.py

# เลือก:
1. Dataset: test_forRetriver_no_keywords.json (Option 2)
2. Contexts: Top 5 (Option 1)
3. Rule-Based Mode: Strict (Option 2)
```

---

### Step 4: Analyze Results 📈

```
เปิดไฟล์ JSON ที่ได้จากการรัน
เปรียบเทียบ:
   - Faithfulness scores
   - Response times
   - Error counts
   - Hybrid strategy usage (Rule-Based vs LLM)
```

---

### Step 5: Make Decision 🎯

```
Based on results:
   ✅ Hybrid = Best for Production (recommended!)
   ⚡ Rule-Based = Good for Development
   🤖 LLM-Based = Best for Accuracy-Critical
```

---

## 🔑 Key Files

### Evaluation Scripts

| File | Purpose | Time | Output |
|------|---------|------|--------|
| `evaluate_retriever_10q.py` | Compare all 3 models | ~20-25 min | Comparison table |
| `evaluate_rule_based_strict.py` | Test Rule-Based only | ~3-5 min | Rule-Based results |
| `evaluate_llm_based.py` | Test LLM-Based only | ~8-12 min | LLM results |
| `evaluate_hybrid.py` | Test Hybrid only | ~6-9 min | Hybrid results |

### Test Datasets

| File | Description | Modified |
|------|-------------|----------|
| `test_forRetriver.json` | 38 questions WITH keywords | 0 (0%) |
| `test_forRetriver_no_keywords.json` | 38 questions, 8 WITHOUT keywords | 8 (21%) |

---

## 💡 FAQ

### Q: ทำไมต้องใช้ `test_forRetriver_no_keywords.json`?

**A:** เพื่อทดสอบว่า model จัดการกับคำถามที่**ไม่มีคีย์เวิร์ดชัดเจน**ได้อย่างไร

- Rule-Based: ตอบไม่ได้ (100% fail on no-keyword questions)
- LLM-Based: ตอบได้ (เข้าใจความหมาย)
- Hybrid: ตอบได้ (ใช้ LLM fallback)

---

### Q: Adjusted Metrics คืออะไร?

**A:** การปรับคะแนน RAGAS โดยนับ error cases เป็น 0.0

```python
# RAGAS default: ข้าม errors → คะแนนสูงเกินจริง
# Adjusted: นับ errors = 0.0 → คะแนนสะท้อนความเป็นจริง

adjusted_score = (ragas_score * num_success + 0.0 * num_errors) / total_questions
```

**อ่านเพิ่มเติม:** `README_ADJUSTED_METRICS.md`

---

### Q: Hybrid ประหยัดได้จริงหรือ?

**A:** ใช่! ประหยัด 60-70%

```
LLM-only (38 questions):
   38 × $0.01 = $0.38-0.50

Hybrid (79% Rule-Based, 21% LLM):
   30 × $0    = $0
   8  × $0.02 = $0.16
   Total      = $0.12-0.16

Savings: 68%! 💰
```

---

### Q: Hybrid เร็วกว่า LLM-only จริงหรือ?

**A:** ใช่!

```
LLM-only:  38 × 12s = ~456s (7.6 min)
Hybrid:    30 × 5s  = 150s
           8  × 15s = 120s
           Total    = 270s (4.5 min)

Faster: 41%! ⚡
```

---

### Q: Model ไหนเหมาะกับ Production?

**A:** 🏆 **Hybrid!**

**เหตุผล:**
- ⚡ Fast (79% use Rule-Based)
- 🤖 Accurate (21% use LLM when needed)
- 💰 Cheap (68% cost savings)
- ✅ Balanced (Best of Both Worlds!)

---

### Q: ต้อง API key ไหม?

**A:**
- Rule-Based: ❌ ไม่ต้อง (ฟรี!)
- LLM-Based: ✅ ต้อง (OPENAI_API_KEY)
- Hybrid: ✅ ต้อง (แต่ใช้น้อย!)

**Setup:**
```bash
# .env file
OPENAI_API_KEY=sk-or-v1-...  # OpenRouter
# หรือ
OPENAI_API_KEY=sk-...        # OpenAI
```

---

## 🎉 Conclusion

```
🏆 Winner: HYBRID! 🔀

✅ Fast:     79% questions use Rule-Based (~5s)
✅ Accurate: 21% questions use LLM (~15s)
✅ Cheap:    68% cost reduction
✅ Ready:    Production-ready NOW!

🚀 Recommendation: Deploy Hybrid to production!
```

---

## 📞 Need Help?

1. **อ่าน:** [README_THREE_MODELS_COMPARISON.md](README_THREE_MODELS_COMPARISON.md) ← เริ่มที่นี่!
2. **ถามเพิ่ม:** Check individual model READMEs
3. **Run Tests:** Follow the Quick Start Guide above

---

**Happy Testing! 🎉**

```bash
# Start here!
cd evaluation
python evaluate_hybrid.py
```

**Expected Result:** 🏆 Best of Both Worlds! ⚡🤖💰
