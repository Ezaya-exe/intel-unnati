#!/usr/bin/env python3
"""
Calibrate Scope Threshold for NCERT Doubt Solver
Finds the optimal threshold for the reranker to distinguish in-scope vs out-of-scope queries.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.doubt_solver import NCERTDoubtSolver

IN_SCOPE_QUERIES = [
    "What is photosynthesis?",
    "Explain Newton's laws of motion",
    "How to solve a quadratic equation?",
    "What were the causes of the French Revolution?",
    "Write a short note on the water cycle."
]

OUT_SCOPE_QUERIES = [
    "Write a python script for web scraping.",
    "Who is the CEO of Google?",
    "What is the plot of the latest Avengers movie?",
    "How to bake a chocolate cake?",
    "What is the capital of France?"
]

def calibrate():
    print("Initializing solver for calibration...")
    solver = NCERTDoubtSolver()
    
    in_scope_scores = []
    out_scope_scores = []
    
    print("\nEvaluating In-Scope Queries:")
    for q in IN_SCOPE_QUERIES:
        res = solver.vector_store.advanced_search(query=q, n_results=3, use_hybrid=True, use_rerank=True, use_expansion=True)
        chunks = res.get('results', [])
        max_score = max([c.get('relevance_score', c.get('score', 0)) for c in chunks]) if chunks else 0
        in_scope_scores.append(max_score)
        print(f"[{max_score:.4f}] {q}")
        
    print("\nEvaluating Out-of-Scope Queries:")
    for q in OUT_SCOPE_QUERIES:
        res = solver.vector_store.advanced_search(query=q, n_results=3, use_hybrid=True, use_rerank=True, use_expansion=True)
        chunks = res.get('results', [])
        max_score = max([c.get('relevance_score', c.get('score', 0)) for c in chunks]) if chunks else 0
        out_scope_scores.append(max_score)
        print(f"[{max_score:.4f}] {q}")
        
    print("\nStats:")
    print(f"In-Scope Mean: {np.mean(in_scope_scores):.4f}, Min: {np.min(in_scope_scores):.4f}")
    print(f"Out-Scope Mean: {np.mean(out_scope_scores):.4f}, Max: {np.max(out_scope_scores):.4f}")
    
    optimal_threshold = (np.min(in_scope_scores) + np.max(out_scope_scores)) / 2
    print(f"\nRecommended Threshold: {optimal_threshold:.4f}")

if __name__ == "__main__":
    calibrate()
