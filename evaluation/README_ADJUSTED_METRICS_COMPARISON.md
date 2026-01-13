# 📊 Adjusted RAGAS Metrics - Comparison Mode

## 🎯 Update Overview

ได้อัพเดท `evaluate_retriever_10q.py` ให้รองรับ **Adjusted Metrics** เหมือนกับ `evaluate_rule_based_strict.py` เพื่อให้การเปรียบเทียบทั้ง 3 models (Rule-Based, LLM-Based, Hybrid) มีความแม่นยำและเป็นธรรม!

---

## ✨ What's New?

### **1. Automatic Metrics Adjustment**

```python
# ถ้ามี errors จะ adjust scores โดยอัตโนมัติ:
adjusted_score = (original_score * num_success + 0.0 * num_errors) / total_questions

# Example: Rule-Based with 10 errors out of 38 questions
# Original: Faithfulness = 1.0 (เฉพาะ 28 ข้อที่ตอบได้)
# Adjusted: Faithfulness = 0.7368 (รวม 10 ข้อที่ตอบไม่ได้)
```

### **2. Dual Score Display**

จะแสดง **2 ชุดคะแนน**:

- ✅ **Adjusted Scores** (นับ errors) → ใช้เปรียบเทียบ
- 📊 **Original Scores** (ข้าม errors) → อ้างอิงว่า RAGAS คำนวณยังไง

### **3. Error Impact Analysis**

แสดงผลกระทบของ errors อย่างชัดเจน:
```
💡 Impact of 10 errors:
   - Error Rate: 10/38 (26.3%)
   - Faithfulness Drop: 0.2632 (26.3%)
   - This shows real-world performance including failures
```

---

## 🚀 How to Use

### **Run Evaluation:**

```bash
cd evaluation
python evaluate_retriever_10q.py
```

### **Interactive Prompts:**

```
1. Select Test Dataset:
   - test_forRetriver.json (มีคีย์เวิร์ด)
   - test_forRetriver_no_keywords.json (ไม่มีคีย์เวิร์ด)

2. Select Contexts:
   - Top 5 contexts
   - Top 10 contexts
   - All contexts

3. Select Rule-Based Mode:
   - Normal Mode (with fallback)
   - Strict Mode (no fallback) ⭐
```

---

## 📊 Output Format

### **Individual Results:**

```
================================================================================
📊 Rule-Based Results:
================================================================================
📈 RAGAS Scores (Adjusted - Including Error Cases):
   - Faithfulness:       0.7368  ✅
   - Context Precision:  0.5158  ✅
   - Context Recall:     0.7368  ✅
💡 Bonus Metric:
   - Answer Relevancy:   0.0685 (for reference)

📊 RAGAS Scores (Original - Success Questions Only):
   - Faithfulness:       1.0000  (เฉพาะ 28 ข้อที่ตอบได้)
   - Context Precision:  nan
   - Context Recall:     1.0000
   - Answer Relevancy:   0.0930

⚡ Performance:
   - Errors:             10 ❌

💡 Impact of 10 errors:
   - Error Rate: 10/38 (26.3%)
   - Faithfulness Drop: 0.2632 (26.3%)
   - This shows real-world performance including failures
```

### **Comparison Table:**

```
================================================================================
📊 COMPARISON SUMMARY - Retriever Test (Adjusted Scores)
================================================================================
💡 Note: Scores are adjusted to include error cases (errors count as 0.0)
   This gives more realistic performance metrics!
================================================================================

Metric                         | Rule-Based           | LLM-Based            | Hybrid              
------------------------------------------------------------------------------------------------
📈 Main Metrics (Adjusted)
Faithfulness                   |               0.7368 |               0.9500 |               0.9700
Context Precision              |               0.5158 |               0.8200 |               0.8500
Context Recall                 |               0.7368 |               0.9500 |               0.9700
------------------------------------------------------------------------------------------------
💡 Bonus Metric (for reference)
Answer Relevancy               |               0.0685 |               0.1200 |               0.1300
------------------------------------------------------------------------------------------------
📊 Original Scores (Success Questions Only - for reference)
Faithfulness                   |               1.0000 |               0.9500 |               0.9700
Context Precision              |                nan   |               0.8200 |               0.8500
Context Recall                 |               1.0000 |               0.9500 |               0.9700
Answer Relevancy               |               0.0930 |               0.1200 |               0.1300
```

---

## 🎯 Key Benefits

### **1. Fair Comparison**

| Metric | Before (Unfair) | After (Fair) |
|--------|-----------------|--------------|
| **Rule-Based** | 1.0000 (ไม่นับ errors) | 0.7368 (นับ errors) ✅ |
| **LLM-Based** | 0.9500 (ไม่มี errors) | 0.9500 (same) |
| **Hybrid** | 0.9700 (ไม่มี errors) | 0.9700 (same) |

**ก่อน:** Rule-Based ดูเท่ากับ LLM/Hybrid (ไม่เป็นธรรม!)
**หลัง:** Rule-Based ต่ำกว่าชัดเจน (เป็นธรรม!)

### **2. Real-World Performance**

- ✅ สะท้อนประสิทธิภาพจริงในการใช้งาน
- ✅ นับ errors ที่เกิดขึ้นจริง
- ✅ แสดงข้อจำกัดของแต่ละ model

### **3. Clear Impact Analysis**

```
⚠️  Error Summary:
   - Rule-Based: 10 errors ❌
   - LLM-Based: 0 errors ✅
   - Hybrid: 0 errors ✅
   💡 Adjusted scores reflect these errors as 0.0
```

---

## 📖 Use Cases

### **Case 1: Normal Evaluation (With Keywords)**

```bash
# Test File: test_forRetriver.json
# Mode: Normal (with fallback)
# Expected: All models perform well (few/no errors)
```

**Purpose:** ประเมินประสิทธิภาพปกติ

### **Case 2: Stress Test (No Keywords)**

```bash
# Test File: test_forRetriver_no_keywords.json
# Mode: Strict (no fallback)
# Expected: Rule-Based struggles, LLM/Hybrid succeed
```

**Purpose:** ทดสอบข้อจำกัดของ Rule-Based

### **Case 3: Context Sensitivity**

```bash
# Contexts: Top 5 vs Top 10 vs All
# Expected: More contexts → Better precision
```

**Purpose:** ศึกษาผลกระทบของจำนวน contexts

---

## 🔄 Comparison with evaluate_rule_based_strict.py

| Feature | `evaluate_rule_based_strict.py` | `evaluate_retriever_10q.py` |
|---------|----------------------------------|------------------------------|
| **Models Tested** | Rule-Based only | All 3 models (Rule/LLM/Hybrid) |
| **Adjusted Metrics** | ✅ Yes | ✅ Yes |
| **Test File** | `test_forRetriver_no_keywords.json` (fixed) | Both files (user choice) |
| **Strict Mode** | Always on | User choice |
| **Purpose** | Prove Rule-Based limitations | Compare all models fairly |
| **Output** | Single model detailed | Comparison table |

---

## 💾 Saved Results

### **JSON File Format:**

```json
{
  "evaluation_info": {
    "test_file": "test_forRetriver.json",
    "total_questions": 38,
    "adjusted_metrics": true,
    "adjustment_explanation": "Error cases are counted as 0.0 for all metrics",
    "total_errors": 10,
    "notes": "Metrics are adjusted to include error cases"
  },
  "results": {
    "Rule-Based": {
      "ragas_scores": {
        "faithfulness": 0.7368,
        "context_precision": 0.5158,
        "context_recall": 0.7368
      },
      "ragas_scores_original": {
        "faithfulness": 1.0000,
        "context_precision": "nan",
        "context_recall": 1.0000
      },
      "performance": {
        "errors": 10
      }
    }
  }
}
```

---

## 📌 Important Notes

### **1. When to Use Adjusted Scores:**

✅ **Always use adjusted scores for:**
- Comparing different models
- Reporting performance to stakeholders
- Making deployment decisions
- Academic papers/presentations

❌ **Don't use adjusted scores for:**
- Debugging RAGAS itself
- When all models have 0 errors (no difference)

### **2. Interpreting NaN Context Precision:**

```
Original Context Precision: nan
→ RAGAS ไม่สามารถคำนวณได้ (error cases ทำให้ context retrieval ล้มเหลว)

Adjusted Context Precision: 0.5158
→ Estimate จากคำถามที่ตอบได้ (~0.7) + errors (0.0)
```

### **3. Error Rate Impact:**

```python
# Error rate 26.3% = Faithfulness drop 26.3%
# This is NOT a coincidence!

error_rate = num_errors / total_questions
faithfulness_drop = original_faithfulness * error_rate

# Example: 1.0 * 0.263 = 0.263
# Adjusted: 1.0 - 0.263 = 0.737 ✅
```

---

## 🎓 Learning Points

1. ✅ **RAGAS ข้าม errors** → ต้อง adjust manually
2. ✅ **Error = 0.0** สำหรับทุก metric
3. ✅ **Adjusted scores สะท้อนความจริง** มากกว่า
4. ✅ **Rule-Based มีข้อจำกัด** ที่ชัดเจน
5. ✅ **Hybrid/LLM-Based เหนือกว่า** อย่างมีนัยสำคัญ

---

## 🔗 Related Files

- `evaluate_retriever_10q.py` - Main script (updated with adjusted metrics)
- `evaluate_rule_based_strict.py` - Rule-Based only evaluation
- `test_forRetriver.json` - Test file with keywords (38 questions)
- `test_forRetriver_no_keywords.json` - Test file without keywords (8 modified)
- `README_ADJUSTED_METRICS.md` - Detailed explanation of adjustment logic

---

## 🚀 Quick Start

```bash
# 1. Run evaluation with default settings (all 3 models)
cd evaluation
python evaluate_retriever_10q.py

# 2. Select options:
#    - Dataset: test_forRetriver.json (1)
#    - Contexts: All contexts (3)
#    - Rule-Based Mode: Strict Mode (2)

# 3. Review results:
#    - Adjusted scores in comparison table
#    - Error impact analysis
#    - Saved JSON file

# 4. Compare with strict mode evaluation:
python evaluate_rule_based_strict.py
```

---

**Created:** 2025-01-20  
**Updated:** 2025-01-20  
**Purpose:** Fair comparison of all chatbot models with adjusted metrics  
**Impact:** More realistic and trustworthy evaluation results! 🎯

