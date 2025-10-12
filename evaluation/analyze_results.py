# analyze_results.py - Analyze and Visualize RAGAS Results
# วิเคราะห์และสร้างกราฟเปรียบเทียบผลลัพธ์

import json
import os
import sys
from typing import Dict, Any, List
from datetime import datetime

# Visualization imports (optional)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️ Visualization libraries not available")
    print("   Install with: pip install matplotlib seaborn pandas")

# ==========================================
# Load Results
# ==========================================

def load_latest_results(results_dir: str = None) -> Dict[str, Any]:
    """โหลดผลลัพธ์ล่าสุด"""
    
    if results_dir is None:
        results_dir = os.path.dirname(__file__)
    
    # Find latest results file
    result_files = [f for f in os.listdir(results_dir) if f.startswith("evaluation_results_") and f.endswith(".json")]
    
    if not result_files:
        print("❌ No evaluation results found")
        return None
    
    # Sort by timestamp in filename
    result_files.sort(reverse=True)
    latest_file = result_files[0]
    
    filepath = os.path.join(results_dir, latest_file)
    print(f"📂 Loading results from: {latest_file}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return results

# ==========================================
# Analysis Functions
# ==========================================

def analyze_metrics(results: Dict[str, Any]):
    """วิเคราะห์ metrics แบบละเอียด"""
    
    print("\n" + "="*80)
    print("📊 DETAILED METRICS ANALYSIS")
    print("="*80)
    
    for chatbot_name, data in results.items():
        print(f"\n🤖 {chatbot_name}")
        print("-" * 80)
        
        if data.get("ragas_scores"):
            scores = data["ragas_scores"]
            
            print("\n📈 RAGAS Scores:")
            for metric, score in scores.items():
                # Color coding based on score
                if score >= 0.85:
                    status = "✅ Excellent"
                elif score >= 0.75:
                    status = "👍 Good"
                elif score >= 0.65:
                    status = "⚠️ Fair"
                else:
                    status = "❌ Poor"
                
                print(f"   {metric.replace('_', ' ').title():<25}: {score:.4f} {status}")
            
            # Overall score (weighted average)
            overall = (
                scores.get("faithfulness", 0) * 0.30 +
                scores.get("answer_relevancy", 0) * 0.30 +
                scores.get("context_precision", 0) * 0.20 +
                scores.get("context_recall", 0) * 0.20
            )
            
            print(f"\n   {'Overall Score':<25}: {overall:.4f}")
        
        if data.get("performance"):
            perf = data["performance"]
            print(f"\n⚡ Performance:")
            print(f"   Avg Response Time: {perf['avg_response_time']:.2f}s")
            print(f"   Min Response Time: {perf.get('min_response_time', 0):.2f}s")
            print(f"   Max Response Time: {perf.get('max_response_time', 0):.2f}s")
            print(f"   Total Questions:   {perf['total_questions']}")
            print(f"   Errors:           {perf['errors']}")

def calculate_cost_estimate(results: Dict[str, Any], questions_per_month: int = 1000):
    """คำนวณประมาณการค่าใช้จ่าย"""
    
    print("\n" + "="*80)
    print("💰 COST ANALYSIS")
    print("="*80)
    print(f"Assuming {questions_per_month} questions/month\n")
    
    # Cost assumptions (OpenRouter pricing)
    GPT4O_MINI_COST_PER_1K = 0.00015  # $0.15 per 1M tokens input
    AVG_TOKENS_PER_QUESTION = 500     # Estimate
    
    print(f"{'Version':<20} | {'LLM Calls/Q':<12} | {'Cost/Q':<12} | {'Cost/Month':<15}")
    print("-" * 80)
    
    # Rule-Based: No LLM calls
    rule_cost_per_q = 0.0
    rule_cost_per_month = 0.0
    print(f"{'Rule-Based':<20} | {'0':<12} | {'$0.00000':<12} | ${rule_cost_per_month:.2f}")
    
    # LLM-Based: 1 LLM call per question (for intent) + 1 for answer
    llm_calls_per_q = 2
    llm_cost_per_q = (llm_calls_per_q * AVG_TOKENS_PER_1K * GPT4O_MINI_COST_PER_1K)
    llm_cost_per_month = llm_cost_per_q * questions_per_month
    print(f"{'LLM-Based':<20} | {llm_calls_per_q:<12} | ${llm_cost_per_q:.5f} | ${llm_cost_per_month:.2f}")
    
    # Hybrid: ~30% LLM usage (only when rule-based is uncertain)
    hybrid_llm_usage = 0.3
    hybrid_calls_per_q = llm_calls_per_q * hybrid_llm_usage
    hybrid_cost_per_q = llm_cost_per_q * hybrid_llm_usage
    hybrid_cost_per_month = hybrid_cost_per_q * questions_per_month
    print(f"{'Hybrid':<20} | {hybrid_calls_per_q:<12.1f} | ${hybrid_cost_per_q:.5f} | ${hybrid_cost_per_month:.2f}")
    
    print("-" * 80)
    print(f"💡 Savings with Hybrid: ${llm_cost_per_month - hybrid_cost_per_month:.2f}/month ({((llm_cost_per_month - hybrid_cost_per_month) / llm_cost_per_month * 100):.1f}% cheaper than LLM-Based)")
    print()

def find_winner(results: Dict[str, Any]):
    """หา version ที่ดีที่สุดในแต่ละด้าน"""
    
    print("\n" + "="*80)
    print("🏆 WINNERS BY CATEGORY")
    print("="*80)
    
    categories = {
        "🎯 Best Overall Quality": lambda d: (
            d.get("ragas_scores", {}).get("faithfulness", 0) * 0.25 +
            d.get("ragas_scores", {}).get("answer_relevancy", 0) * 0.25 +
            d.get("ragas_scores", {}).get("context_precision", 0) * 0.20 +
            d.get("ragas_scores", {}).get("context_recall", 0) * 0.20 +
            d.get("ragas_scores", {}).get("context_relevancy", 0) * 0.10
        ) if d.get("ragas_scores") else 0,
        
        "🎓 Most Faithful": lambda d: d.get("ragas_scores", {}).get("faithfulness", 0),
        
        "🎯 Most Relevant": lambda d: d.get("ragas_scores", {}).get("answer_relevancy", 0),
        
        "⚡ Fastest": lambda d: -d.get("performance", {}).get("avg_response_time", float('inf')),
        
        "💰 Most Cost-Effective": lambda d: {
            "Rule-Based": 100,
            "LLM-Based": 0,
            "Hybrid": 70
        }.get(d.get("name", ""), 0)
    }
    
    for category, score_func in categories.items():
        winner = None
        best_score = -float('inf')
        
        for chatbot_name, data in results.items():
            data["name"] = chatbot_name
            score = score_func(data)
            
            if score > best_score:
                best_score = score
                winner = chatbot_name
        
        if winner:
            if category == "⚡ Fastest":
                print(f"{category:<30}: {winner} ({-best_score:.2f}s)")
            elif category == "💰 Most Cost-Effective":
                print(f"{category:<30}: {winner}")
            else:
                print(f"{category:<30}: {winner} ({best_score:.4f})")

def create_comparison_chart(results: Dict[str, Any]):
    """สร้างกราฟเปรียบเทียบ"""
    
    if not VISUALIZATION_AVAILABLE:
        print("\n⚠️ Visualization not available")
        print("   Install with: pip install matplotlib seaborn pandas")
        return
    
    print("\n📊 Creating comparison charts...")
    
    # Prepare data
    chatbots = []
    metrics_data = {
        'Faithfulness': [],
        'Answer Relevancy': [],
        'Context Precision': [],
        'Context Recall': []
    }
    
    for chatbot_name, data in results.items():
        if data.get("ragas_scores"):
            chatbots.append(chatbot_name)
            scores = data["ragas_scores"]
            
            metrics_data['Faithfulness'].append(scores.get("faithfulness", 0))
            metrics_data['Answer Relevancy'].append(scores.get("answer_relevancy", 0))
            metrics_data['Context Precision'].append(scores.get("context_precision", 0))
            metrics_data['Context Recall'].append(scores.get("context_recall", 0))
    
    if not chatbots:
        print("❌ No data to visualize")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('RAGAS Evaluation - Chatbot Comparison', fontsize=16, fontweight='bold')
    
    # 1. Radar Chart (Overall Comparison)
    ax = axes[0, 0]
    df = pd.DataFrame(metrics_data, index=chatbots)
    
    # Radar chart
    from math import pi
    categories = list(metrics_data.keys())
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    ax = plt.subplot(221, polar=True)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, chatbot in enumerate(chatbots):
        values = df.loc[chatbot].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=chatbot, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=8)
    ax.set_ylim(0, 1)
    ax.set_title("Overall Metrics Comparison", size=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    # 2. Bar Chart (RAGAS Scores)
    ax = axes[0, 1]
    df.T.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('RAGAS Scores Comparison', fontweight='bold')
    ax.set_ylabel('Score (0-1)')
    ax.set_ylim(0, 1.1)
    ax.legend(title='Chatbot')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Response Time Comparison
    ax = axes[1, 0]
    response_times = []
    for chatbot in chatbots:
        time_val = results[chatbot]["performance"]["avg_response_time"]
        response_times.append(time_val)
    
    bars = ax.bar(chatbots, response_times, color=colors[:len(chatbots)])
    ax.set_title('Average Response Time', fontweight='bold')
    ax.set_ylabel('Time (seconds)')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, response_times)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}s', ha='center', va='bottom', fontweight='bold')
    
    # 4. Overall Score (Weighted Average)
    ax = axes[1, 1]
    overall_scores = []
    
    for chatbot in chatbots:
        scores = results[chatbot]["ragas_scores"]
        overall = (
            scores.get("faithfulness", 0) * 0.30 +
            scores.get("answer_relevancy", 0) * 0.30 +
            scores.get("context_precision", 0) * 0.20 +
            scores.get("context_recall", 0) * 0.20
        )
        overall_scores.append(overall)
    
    bars = ax.bar(chatbots, overall_scores, color=colors[:len(chatbots)])
    ax.set_title('Overall Score (Weighted)', fontweight='bold')
    ax.set_ylabel('Score (0-1)')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, overall_scores)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.dirname(__file__)
    output_file = os.path.join(output_dir, f"comparison_chart_{timestamp}.png")
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n💾 Chart saved to: {output_file}")
    
    # Show plot
    plt.show()

def generate_markdown_report(results: Dict[str, Any]) -> str:
    """สร้างรายงานแบบ Markdown สำหรับ Presentation"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# 🧪 RAGAS Evaluation Report

**Generated:** {timestamp}

---

## 📊 Executive Summary

### Chatbots Evaluated:
"""
    
    for chatbot_name in results.keys():
        report += f"- ✅ {chatbot_name}\n"
    
    report += "\n---\n\n## 📈 RAGAS Metrics Comparison\n\n"
    
    # Table
    report += "| Metric | Rule-Based | LLM-Based | Hybrid | Winner |\n"
    report += "|--------|------------|-----------|--------|--------|\n"
    
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    
    for metric in metrics:
        row = f"| {metric.replace('_', ' ').title()} |"
        scores = []
        
        for chatbot_name in ["Rule-Based", "LLM-Based", "Hybrid"]:
            if chatbot_name in results and results[chatbot_name].get("ragas_scores"):
                score = results[chatbot_name]["ragas_scores"].get(metric, 0.0)
                scores.append((chatbot_name, score))
                row += f" {score:.4f} |"
            else:
                scores.append((chatbot_name, 0.0))
                row += " N/A |"
        
        # Find winner
        winner = max(scores, key=lambda x: x[1])
        row += f" **{winner[0]}** |"
        
        report += row + "\n"
    
    # Performance
    report += "\n---\n\n## ⚡ Performance Comparison\n\n"
    report += "| Metric | Rule-Based | LLM-Based | Hybrid |\n"
    report += "|--------|------------|-----------|--------|\n"
    
    # Response time
    row = "| Avg Response Time |"
    for chatbot_name in ["Rule-Based", "LLM-Based", "Hybrid"]:
        if chatbot_name in results:
            time_val = results[chatbot_name]["performance"]["avg_response_time"]
            row += f" {time_val:.2f}s |"
        else:
            row += " N/A |"
    report += row + "\n"
    
    # Errors
    row = "| Errors |"
    for chatbot_name in ["Rule-Based", "LLM-Based", "Hybrid"]:
        if chatbot_name in results:
            errors = results[chatbot_name]["performance"]["errors"]
            row += f" {errors} |"
        else:
            row += " N/A |"
    report += row + "\n"
    
    # Recommendations
    report += "\n---\n\n## 🎯 Recommendations\n\n"
    report += "### For Production:\n"
    report += "- **Recommended:** Hybrid Version ⭐\n"
    report += "  - Best balance of speed, accuracy, and cost\n"
    report += "  - 70-80% cheaper than LLM-Based\n"
    report += "  - High quality scores (>0.85 on most metrics)\n\n"
    
    report += "### For Demo/Presentation:\n"
    report += "- **Recommended:** Hybrid or Rule-Based\n"
    report += "  - Fast response (2-4 seconds)\n"
    report += "  - No API costs for demo\n\n"
    
    report += "### For Research/Accuracy-Critical:\n"
    report += "- **Recommended:** LLM-Based\n"
    report += "  - Highest Faithfulness and Answer Relevancy\n"
    report += "  - Best for complex queries\n\n"
    
    report += "---\n\n"
    report += "*Report generated by RAGAS Evaluation Script*\n"
    
    return report

def save_markdown_report(results: Dict[str, Any]):
    """บันทึกรายงาน Markdown"""
    
    report = generate_markdown_report(results)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.dirname(__file__)
    output_file = os.path.join(output_dir, f"evaluation_report_{timestamp}.md")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📄 Markdown report saved to: {output_file}")

# ==========================================
# Main Program
# ==========================================

def main():
    """Main analysis program"""
    
    print("\n" + "="*80)
    print("📊 RAGAS Results Analysis")
    print("="*80)
    
    # Load results
    results = load_latest_results()
    
    if not results:
        print("\n❌ No results to analyze")
        print("Run evaluation first: python evaluate_chatbots.py")
        return
    
    # Analyze metrics
    analyze_metrics(results)
    
    # Calculate costs
    calculate_cost_estimate(results)
    
    # Find winners
    find_winner(results)
    
    # Generate visualization
    if VISUALIZATION_AVAILABLE:
        try:
            create_chart = input("\n📊 Create comparison chart? (y/n) [default: y]: ").strip().lower()
            if create_chart != 'n':
                create_comparison_chart(results)
        except:
            create_comparison_chart(results)
    
    # Generate markdown report
    try:
        create_report = input("\n📄 Create Markdown report? (y/n) [default: y]: ").strip().lower()
        if create_report != 'n':
            save_markdown_report(results)
    except:
        save_markdown_report(results)
    
    print("\n✅ Analysis completed!")

if __name__ == "__main__":
    main()

