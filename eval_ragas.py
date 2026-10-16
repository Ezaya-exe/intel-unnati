    #!/usr/bin/env python3
"""
RAGAS Evaluation Script for NCERT Doubt Solver
Uses RAGAS 0.2.x API to compute reference-free metrics.

Usage:
    python eval_ragas.py                 # full 50 questions
    python eval_ragas.py -n 5            # quick test with 5 questions
"""
import sys
import io
import json
import time
import os
import argparse
from pathlib import Path

# Force UTF-8 on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import AnswerRelevancy
from ragas import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from core.doubt_solver import NCERTDoubtSolver

# Paths
EVAL_DATASET_PATH = Path(__file__).parent / "data" / "evaluation" / "eval_dataset.json"
RAGAS_CSV = Path(__file__).parent / "data" / "evaluation" / "ragas_scores.csv"
RAGAS_JSON = Path(__file__).parent / "data" / "evaluation" / "ragas_summary.json"

# Config
SLEEP_BETWEEN_QUESTIONS = 20   # Groq TPM safety
JUDGE_MODEL = "llama-3.3-70b-versatile"


def load_dataset():
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def run_pipeline(solver, entry):
    """Run pipeline on one question, return question/answer/contexts."""
    result = solver.ask(
        question=entry["question"],
        grade=entry.get("grade"),
        subject=entry.get("subject"),
        n_context_chunks=5,
    )

    # Re-run the search to get raw chunks
    filters = {}
    if entry.get("grade"):
        filters["grade"] = entry["grade"]
    if entry.get("subject"):
        filters["subject"] = entry["subject"]

    search = solver.vector_store.advanced_search(
        query=entry["question"],
        n_results=5,
        filters=filters if filters else None,
        use_hybrid=True,
        use_rerank=True,
        use_expansion=False,
    )
    contexts = [c["text"] for c in search.get("results", [])]

    return {
        "question": entry["question"],
        "answer": result["answer"],
        "contexts": contexts,
        "ground_truth": ", ".join(entry.get("expected_keywords", [])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num", type=int, default=None,
                        help="Number of questions to evaluate (default: all)")
    args = parser.parse_args()

    print("=" * 70)
    print("RAGAS EVALUATION")
    print("=" * 70)
    print(f"Judge model: {JUDGE_MODEL}")
    print()

    # Load questions
    questions = load_dataset()
    if args.num:
        questions = questions[:args.num]
    print(f"Loaded {len(questions)} questions.\n")

    # Init solver
    print("Initializing solver...")
    solver = NCERTDoubtSolver()
    print("✓ Ready.\n")

    # Warm up
    print("Warming up...")
    try:
        solver.ask("What is photosynthesis?", grade=10, subject="Science")
    except Exception:
        pass

    # Collect responses
    print(f"\nCollecting responses (with {SLEEP_BETWEEN_QUESTIONS}s sleep between)...\n")
    records = []
    for i, entry in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {entry['question'][:60]}")
        try:
            rec = run_pipeline(solver, entry)
            records.append(rec)
            print(f"    ✓ answer: {len(rec['answer'])} chars, contexts: {len(rec['contexts'])}")
        except Exception as e:
            print(f"    ❌ {e}")

        if i < len(questions):
            time.sleep(SLEEP_BETWEEN_QUESTIONS)

    print(f"\n✓ Collected {len(records)} records.\n")

    # Build dataset
    dataset = Dataset.from_list(records)

    # Judge LLM + embeddings
    judge_llm = LangchainLLMWrapper(ChatGroq(
        model=JUDGE_MODEL,
        groq_api_key=os.environ["GROQ_API_KEY"],
        temperature=0.0,
        max_tokens=1024,
    ))

    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    
    # RAGAS run config — critical for Groq
    config = RunConfig(
        timeout=120,
        max_retries=5,
        max_workers=1,
    )

# Metric instances (must be instantiated with the judge)
    
    m2 = AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings)
    m2.strictness = 1   # Prevents "n=3" issue with Groq
    print("Running RAGAS evaluation...\n")
    result = evaluate(
    dataset=dataset,
    metrics=[m2],
    llm=judge_llm,
    embeddings=judge_embeddings,
    run_config=config,
    raise_exceptions=False,
    )

    # Save
    df = result.to_pandas()
    df.to_csv(RAGAS_CSV, index=False, encoding="utf-8")
    print(f"✓ Saved per-question to {RAGAS_CSV}")

    summary = {}
    for m in ["faithfulness", "answer_relevancy"]:
        if m in df.columns:
            summary[m] = round(float(df[m].mean()), 4)

    with open(RAGAS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print
    print("\n" + "=" * 70)
    print("RAGAS SUMMARY")
    print("=" * 70)
    for k, v in summary.items():
        if v != v:   # NaN check
            print(f"  {k:22s} NaN")
            continue
        bar = "█" * int(v * 30)
        print(f"  {k:22s} {v:.3f}  {bar}")
    print("=" * 70)
    print(f"\n✓ Summary: {RAGAS_JSON}")


if __name__ == "__main__":
    main()