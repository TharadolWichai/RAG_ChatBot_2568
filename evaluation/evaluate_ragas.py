"""
evaluate_ragas.py — Ragas Evaluation Script
รัน: python evaluation/evaluate_ragas.py
"""

import json
import math
import os
import pathlib
import warnings
from datetime import datetime

from dotenv import load_dotenv
from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig

# กด deprecation ของ Langchain*Wrapper (ยังจำเป็นสำหรับ embed_query + OpenRouter แบบชัดเจน)
warnings.filterwarnings(
    "ignore",
    message=".*LangchainEmbeddingsWrapper is deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*LangchainLLMWrapper is deprecated.*",
    category=DeprecationWarning,
)

# ── 1. Config ────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EVAL_MODEL = "anthropic/claude-3-haiku"  # judge LLM via OpenRouter

# Embedding ให้ตรงกับระบบ deploy
EMBED_MODEL = "intfloat/multilingual-e5-large"

RUN_CONFIG = RunConfig(
    timeout=300,
    max_retries=8,
    max_workers=8,
)

EVAL_RESULTS_DIR = PROJECT_ROOT / "evaluation" / "eval_results"
INPUT_FILE = EVAL_RESULTS_DIR / "ragas_input.json"
SAMPLE_SIZE = 5  # ทดสอบแบบประหยัดค่าใช้จ่าย

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing. Please set it in .env")

# ── 2. LLM (OpenRouter) — explicit key/base_url ไม่แตะ env ของ KKU ────────────
_lc_llm = ChatOpenAI(
    model=EVAL_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    temperature=0,
    timeout=RUN_CONFIG.timeout,
    max_tokens=2048,
    default_headers={
        "HTTP-Referer": "https://computing.kku.ac.th",
        "X-Title": "KKU-CS-Chatbot-Ragas-Eval",
    },
)
llm = LangchainLLMWrapper(_lc_llm, run_config=RUN_CONFIG)

# ── 3. Embeddings (Local HF) — ให้ตรง production deployment ────────────────
_lc_embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
embeddings = LangchainEmbeddingsWrapper(_lc_embeddings, run_config=RUN_CONFIG)

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


def load_dataset(path: pathlib.Path) -> Dataset:
    data = json.load(open(path, encoding="utf-8"))
    if SAMPLE_SIZE > 0:
        data = data[:SAMPLE_SIZE]
    return Dataset.from_list(data)


def _print_summary(result) -> None:
    """Ragas EvaluationResult ไม่มี .items() — ใช้ค่าเฉลี่ยใน _repr_dict"""
    agg = getattr(result, "_repr_dict", None)
    if not agg:
        print("  (ไม่มีสรุปคะแนน)")
        return
    for metric_name, score in sorted(agg.items()):
        if isinstance(score, float) and math.isnan(score):
            print(f"  {metric_name:<30} nan")
        elif isinstance(score, float):
            print(f"  {metric_name:<30} {score:.4f}")
        else:
            print(f"  {metric_name:<30} {score}")


def main() -> None:
    print(f"Loading input: {INPUT_FILE}")
    dataset = load_dataset(INPUT_FILE)
    print(f"Rows: {len(dataset)}  |  Columns: {dataset.column_names}")
    print(f"Sample mode: first {SAMPLE_SIZE} questions")

    print(f"Judge LLM : {EVAL_MODEL} (OpenRouter)")
    print(f"Embeddings: {EMBED_MODEL} (local HuggingFace)")
    print(f"\nRunning Ragas evaluation with {len(METRICS)} metrics...")
    result = evaluate(
        dataset=dataset,
        metrics=METRICS,
        llm=llm,
        embeddings=embeddings,
        run_config=RUN_CONFIG,
        raise_exceptions=False,
        show_progress=True,
    )

    print("\n" + "=" * 50)
    print("  Ragas Evaluation Summary")
    print("=" * 50)
    _print_summary(result)
    print("=" * 50)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = EVAL_RESULTS_DIR / f"ragas_result_{ts}.json"

    try:
        result_df = result.to_pandas()
        result_df.to_json(out_path, orient="records", force_ascii=False, indent=2)
        print(f"\nSaved detailed results -> {out_path}")
    except Exception as exc:
        fallback = EVAL_RESULTS_DIR / f"ragas_scores_only_{ts}.json"
        with open(fallback, "w", encoding="utf-8") as f:
            json.dump(result.scores, f, ensure_ascii=False, indent=2)
        print(f"\n[to_pandas failed: {exc}]")
        print(f"Saved per-row scores only -> {fallback}")


if __name__ == "__main__":
    main()
