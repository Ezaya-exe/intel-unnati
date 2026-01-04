#!/usr/bin/env python3
"""
NCERT Doubt-Solver Evaluation and Benchmarking Script
Measures: Latency, Citation Accuracy, Answer Relevance, and Keyword Coverage
"""

import json
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import statistics

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.doubt_solver import NCERTDoubtSolver

# Paths
EVAL_DATASET_PATH = Path(__file__).parent / "data" / "evaluation" / "eval_dataset.json"
RESULTS_PATH = Path(__file__).parent / "data" / "evaluation" / "benchmark_results.json"


def load_evaluation_dataset() -> Dict:
    """Load the evaluation dataset"""
    if not EVAL_DATASET_PATH.exists():
        print(f"❌ Evaluation dataset not found: {EVAL_DATASET_PATH}")
        sys.exit(1)
    
    with open(EVAL_DATASET_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_keyword_coverage(answer: str, expected_keywords: List[str]) -> float:
    """Calculate what percentage of expected keywords appear in the answer"""
    if not expected_keywords:
        return 1.0
    
    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return found / len(expected_keywords)


def check_citation_presence(result: Dict) -> bool:
    """Check if citations are present in the response"""
    citations = result.get('citations', [])
    return len(citations) > 0


def check_grade_match(result: Dict, expected_grades: List[int]) -> bool:
    """Check if returned citations match expected grade range"""
    citations = result.get('citations', [])
    
    if not citations:
        return False
    
    for citation in citations:
        grade = citation.get('grade')
        if grade in expected_grades:
            return True
    
    return False


def run_evaluation(solver: NCERTDoubtSolver, dataset: Dict, 
                   max_questions: int = None, verbose: bool = True) -> Dict:
    """
    Run evaluation on the dataset
    
    Returns:
        Dictionary with evaluation metrics
    """
    questions = dataset['questions']
    
    if max_questions:
        questions = questions[:max_questions]
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_questions': len(questions),
        'latencies_ms': [],
        'keyword_coverages': [],
        'citations_present': [],
        'grade_matches': [],
        'in_scope_count': 0,
        'out_of_scope_count': 0,
        'errors': [],
        'individual_results': []
    }
    
    print(f"\n{'='*60}")
    print(f"🧪 Running Evaluation on {len(questions)} Questions")
    print(f"{'='*60}\n")
    
    for i, q in enumerate(questions, 1):
        try:
            if verbose:
                print(f"[{i}/{len(questions)}] {q['question'][:50]}...")
            
            # Time the query
            start_time = time.time()
            
            response = solver.ask(
                question=q['question'],
                grade=q.get('grade'),
                subject=q.get('subject'),
                n_context_chunks=5
            )
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            # Extract metrics
            answer = response.get('answer', '')
            citations = response.get('citations', [])
            in_scope = response.get('in_scope', False)
            
            # Calculate metrics
            keyword_coverage = calculate_keyword_coverage(
                answer, q.get('expected_keywords', [])
            )
            has_citations = len(citations) > 0
            grade_match = check_grade_match(
                response, q.get('expected_grade_range', [q.get('grade', 0)])
            )
            
            # Store results
            results['latencies_ms'].append(latency_ms)
            results['keyword_coverages'].append(keyword_coverage)
            results['citations_present'].append(has_citations)
            results['grade_matches'].append(grade_match)
            
            if in_scope:
                results['in_scope_count'] += 1
            else:
                results['out_of_scope_count'] += 1
            
            # Store individual result
            results['individual_results'].append({
                'id': q['id'],
                'question': q['question'],
                'latency_ms': round(latency_ms, 2),
                'keyword_coverage': round(keyword_coverage, 2),
                'has_citations': has_citations,
                'grade_match': grade_match,
                'in_scope': in_scope,
                'answer_preview': answer[:200] + '...' if len(answer) > 200 else answer
            })
            
            if verbose:
                status = "✅" if keyword_coverage > 0.5 else "⚠️"
                print(f"   {status} Latency: {latency_ms:.0f}ms | Keywords: {keyword_coverage*100:.0f}% | Citations: {has_citations}")
            
        except Exception as e:
            results['errors'].append({
                'id': q['id'],
                'question': q['question'],
                'error': str(e)
            })
            if verbose:
                print(f"   ❌ Error: {e}")
    
    # Calculate summary statistics
    if results['latencies_ms']:
        results['summary'] = {
            'avg_latency_ms': round(statistics.mean(results['latencies_ms']), 2),
            'median_latency_ms': round(statistics.median(results['latencies_ms']), 2),
            'min_latency_ms': round(min(results['latencies_ms']), 2),
            'max_latency_ms': round(max(results['latencies_ms']), 2),
            'p95_latency_ms': round(sorted(results['latencies_ms'])[int(len(results['latencies_ms']) * 0.95)], 2),
            
            'avg_keyword_coverage': round(statistics.mean(results['keyword_coverages']) * 100, 2),
            'citation_rate': round(sum(results['citations_present']) / len(results['citations_present']) * 100, 2),
            'grade_match_rate': round(sum(results['grade_matches']) / len(results['grade_matches']) * 100, 2),
            
            'in_scope_rate': round(results['in_scope_count'] / len(questions) * 100, 2),
            'error_rate': round(len(results['errors']) / len(questions) * 100, 2)
        }
    
    return results


def print_summary(results: Dict):
    """Print evaluation summary"""
    summary = results.get('summary', {})
    
    print(f"\n{'='*60}")
    print("📊 EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n📈 Performance Metrics:")
    print(f"   Total Questions: {results['total_questions']}")
    print(f"   Errors: {len(results['errors'])}")
    
    print(f"\n⏱️ Latency (Target: ≤3000-5000ms):")
    print(f"   Average: {summary.get('avg_latency_ms', 0):.0f}ms")
    print(f"   Median: {summary.get('median_latency_ms', 0):.0f}ms")
    print(f"   P95: {summary.get('p95_latency_ms', 0):.0f}ms")
    print(f"   Min/Max: {summary.get('min_latency_ms', 0):.0f}ms / {summary.get('max_latency_ms', 0):.0f}ms")
    
    # Check if latency target met
    avg_latency = summary.get('avg_latency_ms', 0)
    if avg_latency <= 3000:
        print(f"   ✅ PASS: Average latency under 3 seconds")
    elif avg_latency <= 5000:
        print(f"   ⚠️ ACCEPTABLE: Average latency under 5 seconds")
    else:
        print(f"   ❌ FAIL: Average latency exceeds 5 seconds")
    
    print(f"\n📚 Answer Quality:")
    print(f"   Keyword Coverage: {summary.get('avg_keyword_coverage', 0):.1f}%")
    print(f"   Citation Rate: {summary.get('citation_rate', 0):.1f}%")
    print(f"   Grade Match Rate: {summary.get('grade_match_rate', 0):.1f}%")
    
    # Check if citation target met
    citation_rate = summary.get('citation_rate', 0)
    if citation_rate >= 85:
        print(f"   ✅ PASS: Citation rate >= 85%")
    else:
        print(f"   ❌ FAIL: Citation rate {citation_rate:.1f}% < 85%")
    
    print(f"\n🎯 Scope Detection:")
    print(f"   In-Scope: {summary.get('in_scope_rate', 0):.1f}%")
    
    print(f"\n{'='*60}\n")


def save_results(results: Dict):
    """Save evaluation results to JSON"""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Results saved to: {RESULTS_PATH}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='NCERT Doubt-Solver Evaluation')
    parser.add_argument('-n', '--num-questions', type=int, default=None,
                        help='Number of questions to evaluate (default: all)')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress per-question output')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to file')
    
    args = parser.parse_args()
    
    print("\n🚀 Initializing NCERT Doubt Solver...")
    solver = NCERTDoubtSolver()
    
    print("\n📂 Loading evaluation dataset...")
    dataset = load_evaluation_dataset()
    print(f"   Found {len(dataset['questions'])} questions")
    
    # Run evaluation
    results = run_evaluation(
        solver=solver,
        dataset=dataset,
        max_questions=args.num_questions,
        verbose=not args.quiet
    )
    
    # Print summary
    print_summary(results)
    
    # Save results
    if not args.no_save:
        save_results(results)
    
    # Return exit code based on performance
    summary = results.get('summary', {})
    if summary.get('avg_latency_ms', 10000) <= 5000 and summary.get('citation_rate', 0) >= 85:
        print("✅ All performance targets met!")
        return 0
    else:
        print("⚠️ Some performance targets not met")
        return 1


if __name__ == "__main__":
    sys.exit(main())
