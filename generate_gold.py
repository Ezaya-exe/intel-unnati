#!/usr/bin/env python3
"""
Generate gold answers for the first 10 eval questions using retrieved context.
Each answer is generated ONLY from the RAG's own retrieved chunks,
then saved back into eval_dataset.json as 'expected_answer'.
"""
import json, sys, os, time, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from core.doubt_solver import NCERTDoubtSolver
from langchain_groq import ChatGroq

solver = NCERTDoubtSolver()
llm = ChatGroq(
    model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b"),
    groq_api_key=os.environ["GROQ_API_KEY"],
    temperature=0,
)

path = Path(__file__).parent.parent / "data" / "evaluation" / "eval_dataset.json"
with open(path, encoding="utf-8") as f:
    data = json.load(f)

N = 10  # start with 10

for i, q in enumerate(data["questions"][:N], 1):
    print(f"\n[{i}/{N}] {q['question']}")
    try:
        r = solver.ask(
            question=q["question"],
            grade=q.get("grade"),
            subject=q.get("subject"),
        )
        context_chunks = [c.get("text", "") for c in r.get("citations", []) if c.get("text")]
        if not context_chunks:
            print("  ⚠️ no context retrieved — skipping")
            continue

        context = "\n\n---\n\n".join(context_chunks[:5])

        prompt = f"""You are writing a reference answer for a school textbook Q&A dataset.

Using ONLY the context below, write a clear 2-3 sentence answer to the question.
The answer should be factually grounded in the context. Do not add outside knowledge.
Use the same language as the question.

Context:
{context}

Question: {q['question']}

Answer:"""

        gold = llm.invoke(prompt).content.strip()
        gold = re.sub(r" thinking.*?", "", gold, flags=re.DOTALL).strip()

        q["expected_answer"] = gold
        print(f"  → {gold[:150]}...")

        time.sleep(1.2)  # rate limit
    except Exception as e:
        print(f"  ✗ error: {e}")

with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\n✅ Saved gold answers for {N} questions to {path}")
print("⚠️  MANUALLY REVIEW before running RAGAS.")