import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, "src")
from core.doubt_solver import NCERTDoubtSolver
import re

solver = NCERTDoubtSolver()
data = solver.vector_store.collection.get(limit=100)
metas = data['metadatas']

print(f"Languages in DB: {set(m.get('language') for m in metas)}")

# Check for Devanagari in the first few chunks
for i, (doc, meta) in enumerate(zip(data['documents'][:5], metas[:5]), 1):
    has_devanagari = bool(re.search(r'[\u0900-\u097F]', doc))
    print(f"Chunk {i}: Devanagari={has_devanagari}, Text={doc[:80]}...")