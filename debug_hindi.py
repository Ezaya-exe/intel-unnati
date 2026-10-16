import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.core.doubt_solver import NCERTDoubtSolver

solver = NCERTDoubtSolver()

# Hindi question
q = "प्रकाश संश्लेषण क्या है?"
print(f"\nQuestion: {q}\n")

# Call advanced_search directly to see raw results
results = solver.vector_store.advanced_search(
    query=q,
    n_results=5,
    filters={'grade': 8, 'subject': 'Science'},
    use_hybrid=True,
    use_rerank=True,
    use_expansion=True
)

print(f"Got {len(results.get('results', []))} chunks\n")
for i, chunk in enumerate(results.get('results', []), 1):
    meta = chunk.get('metadata', {})
    score = chunk.get('relevance_score', chunk.get('score', 'N/A'))
    print(f"--- Chunk {i} ---")
    print(f"  Grade: {meta.get('grade')}, Subject: {meta.get('subject')}, Lang: {meta.get('language')}")
    print(f"  Score: {score}")
    print(f"  Text: {chunk['text'][:200]}...")
    print()