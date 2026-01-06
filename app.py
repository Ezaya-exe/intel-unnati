"""
NCERT Multilingual Doubt Solver - Gradio Web Application
Premium UI with smooth animations and modern design
"""
import gradio as gr
import sys
import uuid
import time
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.doubt_solver import NCERTDoubtSolver, detect_language, LANGUAGE_NAMES
from core.feedback import save_feedback, get_feedback_stats
from generation.qwen_gguf import load_model

# Global solver instance
solver = None
session_id = str(uuid.uuid4())[:8]
last_response = {}

def initialize_app():
    """Initialize the backend components"""
    global solver
    print("⏳ Initializing Backend...")
    try:
        print("   - Loading Vector Database...")
        solver = NCERTDoubtSolver(use_wsl_server=False)
        print("   - Pre-loading GGUF LLM (WSL GPU)...")
        load_model()
        print("✅ Backend Initialized Successfully!")
        return True
    except Exception as e:
        print(f"❌ Error initializing backend: {e}")
        traceback.print_exc()
        return False

def format_citations_html(citations):
    """Format citations as beautiful cards"""
    if not citations:
        return ""
    
    html = """
    <div class="citations-container">
        <div class="citations-header" onclick="this.parentElement.classList.toggle('expanded')">
            <span class="citations-icon">📚</span>
            <span class="citations-title">Sources & Citations</span>
            <span class="citations-count">{} sources</span>
            <span class="citations-chevron">▼</span>
        </div>
        <div class="citations-content">
    """.format(len(citations))
    
    for cite in citations:
        grade_color = {5: '#10b981', 6: '#06b6d4', 7: '#3b82f6', 8: '#8b5cf6', 9: '#ec4899', 10: '#f59e0b'}.get(cite['grade'], '#6b7280')
        html += f"""
        <div class="citation-card">
            <div class="citation-badge" style="background: {grade_color}">Grade {cite['grade']}</div>
            <div class="citation-subject">{cite['subject']}</div>
            <div class="citation-text">{cite['text']}</div>
        </div>
        """
    html += "</div></div>"
    return html

def chat(message, history, grade, subject, language):
    """Process chat message and return response"""
    global last_response, solver
    
    if not message.strip():
        return "", history, "", "", gr.update(visible=False)
    
    if solver is None:
        return "", history or [], "", "⚠️ Backend not initialized", gr.update(visible=False)
    
    # Convert language selection
    lang_code = None
    if language and language != "Auto-detect":
        for code, name in LANGUAGE_NAMES.items():
            if name == language:
                lang_code = code
                break
    
    try:
        # Get response
        result = solver.ask(
            question=message,
            grade=int(grade) if grade else None,
            subject=subject if subject and subject != "All Subjects" else None,
            language=lang_code,
            n_context_chunks=5
        )
        
        # Store for feedback
        last_response = {
            'question': message,
            'answer': result['answer'],
            'grade': grade,
            'subject': subject,
            'language': result['metadata']['response_language'],
            'latency': result['metadata']['latency_seconds']
        }
        
        # Format response
        answer = result['answer']
        citations_html = format_citations_html(result['citations'])
        metadata = result['metadata']
        
        # Create beautiful metadata display
        latency = metadata['latency_seconds']
        latency_color = '#10b981' if latency < 3 else '#f59e0b' if latency < 5 else '#ef4444'
        meta_html = f"""
        <div class="meta-badges">
            <span class="meta-badge latency" style="--badge-color: {latency_color}">
                <span class="meta-icon">⚡</span> {latency:.1f}s
            </span>
            <span class="meta-badge language">
                <span class="meta-icon">🌐</span> {LANGUAGE_NAMES.get(metadata['detected_language'], 'Unknown')}
            </span>
            <span class="meta-badge sources">
                <span class="meta-icon">📖</span> {metadata['num_sources']} sources
            </span>
        </div>
        """
        
        history = history or []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})
        
        return "", history, citations_html, meta_html, gr.update(visible=True)
        
    except Exception as e:
        print(f"Error during chat: {e}")
        traceback.print_exc()
        error_msg = f"❌ Error: {str(e)}"
        history = history or []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return "", history, "", "", gr.update(visible=False)

def submit_feedback(is_helpful):
    global last_response
    if not last_response:
        return "❌ No response to rate yet"
    
    try:
        feedback_id = save_feedback(
            question=last_response.get('question', ''),
            answer=last_response.get('answer', ''),
            helpful=is_helpful,
            session_id=session_id,
            grade=int(last_response.get('grade')) if last_response.get('grade') else None,
            subject=last_response.get('subject'),
            language=last_response.get('language'),
            latency_seconds=last_response.get('latency')
        )
        emoji = "👍" if is_helpful else "👎"
        return f"""<div class="feedback-toast {'success' if is_helpful else 'noted'}">{emoji} Thank you for your feedback!</div>"""
    except Exception as e:
        return f"""<div class="feedback-toast error">❌ Error saving feedback</div>"""

def thumbs_up(): return submit_feedback(True)
def thumbs_down(): return submit_feedback(False)

def clear_chat():
    global solver, last_response
    if solver:
        solver.clear_history()
    last_response = {}
    return [], "", "", gr.update(visible=False)

def get_stats():
    global solver
    if not solver:
        return "Backend not initialized"
        
    try:
        db_stats = solver.get_stats()
        feedback_stats = get_feedback_stats()
        return f"""
### 📊 System Statistics
**Vector Database:**
- Total Chunks: {db_stats.get('total_chunks', 0):,}
- Grades: {db_stats.get('grades', {})}
- Subjects: {db_stats.get('subjects', {})}

**Feedback:**
- Total Ratings: {feedback_stats.get('total_feedback', 0)}
- Helpful Rate: {feedback_stats.get('helpful_rate', 0)}%
- Avg Latency: {feedback_stats.get('average_latency', 'N/A')}s
"""
    except Exception as e:
        return f"Error loading stats: {str(e)}"

# Premium CSS with animations
PREMIUM_CSS = """
/* ===== IMPORTS ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono&display=swap');

/* ===== CSS VARIABLES ===== */
:root {
    --primary: #6366f1;
    --primary-dark: #4f46e5;
    --primary-light: #818cf8;
    --secondary: #ec4899;
    --success: #10b981;
    --warning: #f59e0b;
    --error: #ef4444;
    --bg-dark: #0f172a;
    --bg-card: #1e293b;
    --bg-hover: #334155;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --border: #334155;
    --glass: rgba(255, 255, 255, 0.05);
    --shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    --radius: 16px;
    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ===== BASE STYLES ===== */
.gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%) !important;
    min-height: 100vh;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.dark {
    --background-fill-primary: transparent !important;
}

/* ===== ANIMATIONS ===== */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}

@keyframes glow {
    0%, 100% { box-shadow: 0 0 20px rgba(99, 102, 241, 0.3); }
    50% { box-shadow: 0 0 40px rgba(99, 102, 241, 0.6); }
}

/* ===== HEADER ===== */
.header-main {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(236, 72, 153, 0.2) 100%);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 24px;
    padding: 32px;
    margin-bottom: 24px;
    text-align: center;
    animation: fadeInUp 0.6s ease-out;
    position: relative;
    overflow: hidden;
}

.header-main::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.1) 0%, transparent 70%);
    animation: float 6s ease-in-out infinite;
}

.header-title {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #fff 0%, #c7d2fe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px 0;
    position: relative;
    z-index: 1;
}

.header-subtitle {
    color: var(--text-secondary);
    font-size: 1.1rem;
    margin: 0;
    position: relative;
    z-index: 1;
}

.header-badges {
    display: flex;
    justify-content: center;
    gap: 12px;
    margin-top: 16px;
    flex-wrap: wrap;
    position: relative;
    z-index: 1;
}

.header-badge {
    background: rgba(255, 255, 255, 0.1);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    color: var(--text-secondary);
    border: 1px solid rgba(255, 255, 255, 0.1);
    transition: var(--transition);
}

.header-badge:hover {
    background: rgba(255, 255, 255, 0.15);
    transform: translateY(-2px);
}

/* ===== CHATBOT ===== */
.chatbot-container {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden;
    box-shadow: var(--shadow);
}

.chatbot {
    background: transparent !important;
}

.message {
    animation: fadeInUp 0.3s ease-out !important;
}

.user-message, .bot-message {
    border-radius: 16px !important;
    padding: 16px 20px !important;
    margin: 8px 0 !important;
    line-height: 1.6 !important;
}

.user-message {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
    color: white !important;
    margin-left: 20% !important;
}

.bot-message {
    background: var(--bg-hover) !important;
    color: var(--text-primary) !important;
    margin-right: 20% !important;
    border: 1px solid var(--border) !important;
}

/* ===== INPUT AREA ===== */
.input-row {
    display: flex;
    gap: 12px;
    margin-top: 16px;
}

textarea {
    background: var(--bg-card) !important;
    border: 2px solid var(--border) !important;
    border-radius: 16px !important;
    color: var(--text-primary) !important;
    font-size: 16px !important;
    padding: 16px 20px !important;
    transition: var(--transition) !important;
    resize: none !important;
}

textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.2) !important;
    outline: none !important;
}

textarea::placeholder {
    color: var(--text-secondary) !important;
}

/* ===== BUTTONS ===== */
.primary-btn {
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%) !important;
    border: none !important;
    border-radius: 16px !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    padding: 16px 32px !important;
    cursor: pointer !important;
    transition: var(--transition) !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
    min-width: 120px !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5) !important;
}

.primary-btn:active {
    transform: translateY(0) !important;
}

.secondary-btn {
    background: var(--bg-hover) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-weight: 500 !important;
    padding: 12px 20px !important;
    transition: var(--transition) !important;
}

.secondary-btn:hover {
    background: var(--bg-card) !important;
    border-color: var(--primary) !important;
}

/* ===== FEEDBACK BUTTONS ===== */
.feedback-container {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px;
    background: var(--glass);
    border-radius: 16px;
    border: 1px solid var(--border);
    margin-top: 16px;
}

.feedback-label {
    color: var(--text-secondary);
    font-size: 0.9rem;
}

.feedback-btn {
    width: 50px !important;
    height: 50px !important;
    border-radius: 50% !important;
    background: var(--bg-hover) !important;
    border: 2px solid var(--border) !important;
    font-size: 24px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: var(--transition) !important;
    padding: 0 !important;
}

.feedback-btn:hover {
    transform: scale(1.15) !important;
    border-color: var(--primary) !important;
}

.feedback-btn.thumbs-up:hover {
    background: rgba(16, 185, 129, 0.2) !important;
    border-color: var(--success) !important;
}

.feedback-btn.thumbs-down:hover {
    background: rgba(239, 68, 68, 0.2) !important;
    border-color: var(--error) !important;
}

/* ===== FEEDBACK TOAST ===== */
.feedback-toast {
    padding: 12px 20px;
    border-radius: 12px;
    font-size: 0.9rem;
    animation: fadeInUp 0.3s ease-out;
}

.feedback-toast.success {
    background: rgba(16, 185, 129, 0.2);
    color: var(--success);
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.feedback-toast.noted {
    background: rgba(245, 158, 11, 0.2);
    color: var(--warning);
    border: 1px solid rgba(245, 158, 11, 0.3);
}

/* ===== META BADGES ===== */
.meta-badges {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 12px;
}

.meta-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
    background: var(--glass);
    border: 1px solid var(--border);
    color: var(--text-secondary);
    transition: var(--transition);
}

.meta-badge:hover {
    transform: translateY(-2px);
    background: rgba(255, 255, 255, 0.1);
}

.meta-badge.latency {
    color: var(--badge-color, var(--success));
    border-color: var(--badge-color, var(--success));
}

.meta-icon {
    font-size: 1rem;
}

/* ===== CITATIONS ===== */
.citations-container {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    margin-top: 16px;
    overflow: hidden;
    transition: var(--transition);
}

.citations-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px 20px;
    cursor: pointer;
    transition: var(--transition);
}

.citations-header:hover {
    background: var(--bg-hover);
}

.citations-icon {
    font-size: 1.2rem;
}

.citations-title {
    font-weight: 600;
    color: var(--text-primary);
    flex: 1;
}

.citations-count {
    background: var(--primary);
    color: white;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 500;
}

.citations-chevron {
    color: var(--text-secondary);
    transition: transform 0.3s ease;
}

.citations-container.expanded .citations-chevron {
    transform: rotate(180deg);
}

.citations-content {
    max-height: 0;
    overflow: hidden;
    transition: max-height 0.4s ease;
}

.citations-container.expanded .citations-content {
    max-height: 500px;
    overflow-y: auto;
}

.citation-card {
    padding: 16px 20px;
    border-top: 1px solid var(--border);
    animation: fadeInUp 0.3s ease-out;
}

.citation-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 0.75rem;
    font-weight: 600;
    color: white;
    margin-bottom: 8px;
}

.citation-subject {
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 6px;
}

.citation-text {
    color: var(--text-secondary);
    font-size: 0.9rem;
    line-height: 1.5;
}

/* ===== SIDEBAR ===== */
.sidebar {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    animation: fadeInUp 0.4s ease-out 0.2s backwards;
}

.sidebar-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ===== DROPDOWNS ===== */
.dropdown-container select,
.dropdown-container input {
    background: var(--bg-hover) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    padding: 12px 16px !important;
    font-size: 14px !important;
    transition: var(--transition) !important;
}

.dropdown-container select:focus,
.dropdown-container input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
}

.dropdown-container label {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    margin-bottom: 8px !important;
}

/* ===== FOOTER ===== */
.footer {
    text-align: center;
    padding: 24px;
    margin-top: 24px;
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    animation: fadeInUp 0.4s ease-out 0.3s backwards;
}

.footer-text {
    color: var(--text-secondary);
    font-size: 0.9rem;
}

.footer-text span {
    margin: 0 8px;
    opacity: 0.5;
}

/* ===== ACCORDION ===== */
.accordion {
    background: var(--bg-hover) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    overflow: hidden;
}

.accordion button {
    background: transparent !important;
    color: var(--text-primary) !important;
}

/* ===== LOADING STATE ===== */
.loading {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--text-secondary);
}

.loading-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--primary);
    animation: pulse 1.5s infinite;
}

.loading-dot:nth-child(2) { animation-delay: 0.2s; }
.loading-dot:nth-child(3) { animation-delay: 0.4s; }

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-dark);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--text-secondary);
}

/* ===== MOBILE RESPONSIVE ===== */
@media (max-width: 768px) {
    .header-title {
        font-size: 1.75rem;
    }
    
    .header-subtitle {
        font-size: 0.95rem;
    }
    
    .header-badges {
        gap: 8px;
    }
    
    .header-badge {
        font-size: 0.75rem;
        padding: 4px 10px;
    }
    
    .user-message {
        margin-left: 10% !important;
    }
    
    .bot-message {
        margin-right: 10% !important;
    }
    
    .meta-badges {
        flex-direction: column;
    }
    
    .feedback-container {
        flex-wrap: wrap;
        justify-content: center;
    }
}

/* ===== TRANSITIONS ===== */
* {
    transition: background 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}

/* ===== HIDE GRADIO FOOTER ===== */
footer {
    display: none !important;
}

/* ===== FIX GRADIO DEFAULTS ===== */
.contain {
    background: transparent !important;
}

.block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.wrap {
    background: transparent !important;
}

label {
    color: var(--text-secondary) !important;
}
"""

# Create Gradio interface
with gr.Blocks(title="NCERT Doubt Solver") as demo:
    
    # Header
    gr.HTML("""
        <style>""" + PREMIUM_CSS + """</style>
        <div class="header-main">
            <h1 class="header-title">📚 NCERT Doubt Solver</h1>
            <p class="header-subtitle">Your AI-powered study companion for Grades 5-10</p>
            <div class="header-badges">
                <span class="header-badge">🇮🇳 Hindi & English</span>
                <span class="header-badge">📖 465+ Textbooks</span>
                <span class="header-badge">⚡ Instant Answers</span>
                <span class="header-badge">🎯 Accurate Citations</span>
            </div>
        </div>
    """)
    
    with gr.Row():
        # Main chat column
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="",
                height=450,
                elem_classes=["chatbot-container"],
                show_label=False
            )
            
            with gr.Row(elem_classes=["input-row"]):
                msg = gr.Textbox(
                    placeholder="💬 Ask your doubt here... (e.g., What is photosynthesis?)",
                    show_label=False,
                    scale=5,
                    lines=1,
                    max_lines=3,
                    container=False
                )
                send_btn = gr.Button(
                    "Send ✨",
                    variant="primary",
                    scale=1,
                    elem_classes=["primary-btn"]
                )
            
            # Meta info
            meta_display = gr.HTML("", elem_id="meta-display")
            
            # Citations
            citations_html = gr.HTML("", elem_classes=["citations-wrapper"])
            
            # Feedback section
            with gr.Row(visible=False, elem_classes=["feedback-container"]) as feedback_row:
                gr.HTML('<span class="feedback-label">Was this helpful?</span>')
                thumbs_up_btn = gr.Button("👍", elem_classes=["feedback-btn", "thumbs-up"])
                thumbs_down_btn = gr.Button("👎", elem_classes=["feedback-btn", "thumbs-down"])
                feedback_status = gr.HTML("")
        
        # Sidebar
        with gr.Column(scale=1, elem_classes=["sidebar"]):
            gr.HTML('<div class="sidebar-title">⚙️ Filters</div>')
            
            grade = gr.Dropdown(
                choices=[""] + [str(i) for i in range(5, 11)],
                label="Grade",
                value="",
                info="Select grade (5-10)",
                elem_classes=["dropdown-container"]
            )
            
            subject = gr.Dropdown(
                choices=["All Subjects", "Mathematics", "Science", "Social Science", "Hindi", "English", "Sanskrit"],
                label="Subject",
                value="All Subjects",
                elem_classes=["dropdown-container"]
            )
            
            language = gr.Dropdown(
                choices=["Auto-detect"] + list(LANGUAGE_NAMES.values()),
                label="Language",
                value="Auto-detect",
                info="Auto-detects from your question",
                elem_classes=["dropdown-container"]
            )
            
            gr.HTML('<div style="margin: 20px 0; border-top: 1px solid var(--border);"></div>')
            
            clear_btn = gr.Button(
                "🗑️ Clear Chat",
                variant="secondary",
                elem_classes=["secondary-btn"]
            )
            
            gr.HTML('<div style="margin: 20px 0; border-top: 1px solid var(--border);"></div>')
            
            with gr.Accordion("📊 Statistics", open=False, elem_classes=["accordion"]):
                stats_display = gr.Markdown("")
                refresh_stats_btn = gr.Button("Refresh", size="sm", elem_classes=["secondary-btn"])
    
    # Footer
    gr.HTML("""
        <div class="footer">
            <p class="footer-text">
                🎓 Intel Unnati Program
                <span>•</span>
                📖 Powered by NCERT
                <span>•</span>
                🤖 Qwen3-4B LLM
                <span>•</span>
                Made with ❤️ in India
            </p>
        </div>
    """)
    
    # Event handlers
    msg.submit(
        chat,
        inputs=[msg, chatbot, grade, subject, language],
        outputs=[msg, chatbot, citations_html, meta_display, feedback_row]
    )
    send_btn.click(
        chat,
        inputs=[msg, chatbot, grade, subject, language],
        outputs=[msg, chatbot, citations_html, meta_display, feedback_row]
    )
    thumbs_up_btn.click(thumbs_up, outputs=[feedback_status])
    thumbs_down_btn.click(thumbs_down, outputs=[feedback_status])
    clear_btn.click(clear_chat, outputs=[chatbot, citations_html, meta_display, feedback_row])
    refresh_stats_btn.click(get_stats, outputs=[stats_display])

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting NCERT Doubt Solver Web App")
    print("="*60 + "\n")
    
    # Initialize backend
    if not initialize_app():
        print("⚠️ Backend initialization failed. UI will still load.")
    
    print("\n🌐 Launching at http://localhost:7860\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
