"""
Feedback System for NCERT Doubt Solver
Stores student feedback in SQLite database
"""
import sqlite3
import os
from datetime import datetime
from typing import Optional, List, Dict
import json

DATABASE_PATH = "data/feedback.db"


def init_database():
    """Initialize the feedback database"""
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            rating INTEGER CHECK(rating >= 1 AND rating <= 5),
            helpful BOOLEAN,
            comment TEXT,
            grade INTEGER,
            subject TEXT,
            language TEXT,
            latency_seconds REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()


def save_feedback(
    question: str,
    answer: str,
    helpful: bool,
    session_id: Optional[str] = None,
    rating: Optional[int] = None,
    comment: Optional[str] = None,
    grade: Optional[int] = None,
    subject: Optional[str] = None,
    language: Optional[str] = None,
    latency_seconds: Optional[float] = None
) -> int:
    """
    Save feedback to database
    
    Args:
        question: The question asked
        answer: The answer provided
        helpful: True if thumbs up, False if thumbs down
        session_id: Optional session identifier
        rating: Optional 1-5 star rating
        comment: Optional text feedback
        grade: Grade level used
        subject: Subject used
        language: Language detected/used
        latency_seconds: Response time
        
    Returns:
        Feedback ID
    """
    init_database()
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO feedback 
        (session_id, question, answer, rating, helpful, comment, grade, subject, language, latency_seconds)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        session_id, question, answer, rating, helpful, comment,
        grade, subject, language, latency_seconds
    ))
    
    feedback_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return feedback_id


def get_feedback_stats() -> Dict:
    """Get aggregated feedback statistics"""
    init_database()
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Total count
    cursor.execute("SELECT COUNT(*) FROM feedback")
    total = cursor.fetchone()[0]
    
    # Helpful count
    cursor.execute("SELECT COUNT(*) FROM feedback WHERE helpful = 1")
    helpful = cursor.fetchone()[0]
    
    # Not helpful count
    cursor.execute("SELECT COUNT(*) FROM feedback WHERE helpful = 0")
    not_helpful = cursor.fetchone()[0]
    
    # Average rating
    cursor.execute("SELECT AVG(rating) FROM feedback WHERE rating IS NOT NULL")
    avg_rating = cursor.fetchone()[0]
    
    # Average latency
    cursor.execute("SELECT AVG(latency_seconds) FROM feedback WHERE latency_seconds IS NOT NULL")
    avg_latency = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        'total_feedback': total,
        'helpful_count': helpful,
        'not_helpful_count': not_helpful,
        'helpful_rate': round(helpful / total * 100, 1) if total > 0 else 0,
        'average_rating': round(avg_rating, 2) if avg_rating else None,
        'average_latency': round(avg_latency, 2) if avg_latency else None
    }


def get_recent_feedback(limit: int = 10) -> List[Dict]:
    """Get recent feedback entries"""
    init_database()
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, session_id, question, answer, helpful, rating, comment, 
               grade, subject, language, latency_seconds, timestamp
        FROM feedback
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    feedback_list = []
    for row in rows:
        feedback_list.append({
            'id': row[0],
            'session_id': row[1],
            'question': row[2],
            'answer': row[3][:200] + '...' if len(row[3]) > 200 else row[3],
            'helpful': row[4],
            'rating': row[5],
            'comment': row[6],
            'grade': row[7],
            'subject': row[8],
            'language': row[9],
            'latency_seconds': row[10],
            'timestamp': row[11]
        })
    
    return feedback_list


# Initialize database on import
init_database()
