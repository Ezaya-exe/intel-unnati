"""
Profile latency across representative questions.
Usage: python profile_latency.py
"""
import sys
import io
import time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
from src.core.doubt_solver import NCERTDoubtSolver

solver = NCERTDoubtSolver()

test_questions = [
    ("What is photosynthesis?", 8, "Science", "English"),
    ("प्रकाश संश्लेषण क्या है?", 8, "Science", "Hindi"),
    ("Solve for x: 2x + 5 = 15", 9, "Maths", "English"),
    ("Explain the water cycle with examples from everyday life", 7, "Science", "English"),
    ("Why is it important?", 8, "Science", "English"),
]

print("\n" + "=" * 60)
print("PROFILING LATENCY ACROSS REPRESENTATIVE QUESTIONS")
print("=" * 60)

all_timings = []

for i, (q, grade, subject, lang) in enumerate(test_questions, 1):
    print(f"\n[{i}/{len(test_questions)}] Q: {q[:60]}")
    try:
        result = solver.ask(q, grade=grade, subject=subject, language=lang)
        all_timings.append(result["metadata"]["timings"])
        print(f"  Answer: {result['answer'][:120]}...")
        print(f"  Latency: {result['metadata']['latency_seconds']}s")
        print(f"  In-scope: {result['metadata']['in_scope']}")
    except Exception as e:
        import traceback
        print(f"  ❌ FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        continue   # ← skip to next question instead of crashing
    if i < len(test_questions):
        time.sleep(15)        
# Aggregate
print("\n" + "=" * 60)
print("AGGREGATE TIMINGS (averaged)")
print("=" * 60)

if all_timings:
    keys = all_timings[0].keys()
    avg = {k: sum(t.get(k, 0) for t in all_timings) / len(all_timings) for k in keys}
    total = sum(avg.values())
    for name, ms in sorted(avg.items(), key=lambda x: -x[1]):
        pct = (ms / total) * 100
        print(f"  {name:35s} {ms:8.1f} ms  ({pct:5.1f}%)")
    print("-" * 60)
    print(f"  {'TOTAL':35s} {total:8.1f} ms")