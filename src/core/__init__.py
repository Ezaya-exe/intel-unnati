"""
NCERT Doubt Solver Core Module
"""
from .doubt_solver import NCERTDoubtSolver, detect_language, is_in_scope
from .feedback import save_feedback, get_feedback_stats, get_recent_feedback

__all__ = [
    'NCERTDoubtSolver',
    'detect_language',
    'is_in_scope',
    'save_feedback',
    'get_feedback_stats',
    'get_recent_feedback'
]
