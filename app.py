"""
NCERT Doubt Solver - Gradio Web Application
Premium UI with streaming responses, citations, and feedback
"""
import gradio as gr
import sys
import uuid
import time
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.doubt_solver import NCERTDoubtSolver
from core.feedback import save_feedback, get_feedback_stats

# Global state
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
        print("✅ Backend Initialized Successfully!")
        return True
    except Exception as e:
        print(f"❌ Error initializing backend: {e}")
        traceback.print_exc()
        return False


def format_citations_html(citations):
    """Format citations as beautiful cards with source info."""
    if not citations:
        return ""

    html = f"""
    <div class="citations-container">
        <div class="citations-header" onclick="this.parentElement.classList.toggle('expanded')">
            <span class="citations-icon">📚</span>
            <span class="citations-title">Sources & Citations</span>
            <span class="citations-count">{len(citations)} sources</span>
            <span class="citations-chevron">▼</span>
        </div>
        <div class="citations-content">
    """

    for cite in citations:
        grade_color = {5: '#10b981', 6: '#06b6d4', 7: '#3b82f6',
                       8: '#8b5cf6', 9: '#ec4899', 10: '#f59e0b'}.get(cite['grade'], '#6b7280')
        source_file = cite.get('source_file', '') or ''
        # Pretty-print source file name
        source_display = source_file.replace('.pdf', '').replace('_', ' ') if source_file else ''
        page = cite.get('page', 'N/A')

        html += f"""
        <div class="citation-card">
            <div class="citation-head">
                <span class="citation-badge" style="background: {grade_color}">Grade {cite['grade']}</span>
                <span class="citation-subject">{cite['subject']}</span>
                {f'<span class="citation-page">Page {page}</span>' if page and page != 'N/A' else ''}
            </div>
            {f'<div class="citation-source">📄 {source_display}</div>' if source_display else ''}
            <div class="citation-text">{cite['text']}</div>
        </div>
        """

    html += "</div></div>"
    return html


def chat(message, history, grade, subject):
    """Streaming chat handler."""
    global last_response, solver
    history = history or []

    if not message.strip():
        yield "", history, "", "", gr.update(visible=False)
        return

    if solver is None:
        yield "", history, "", "⚠️ Backend not initialized", gr.update(visible=False)
        return

    # Append user message + placeholder assistant message
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "▌"},
    ]
    yield "", history, "", "", gr.update(visible=False)

    try:
        citations_html = ""
        meta_html = ""
        final_answer = ""
        final_metadata = {}

        for event in solver.ask_stream(
            question=message,
            grade=int(grade) if grade else None,
            subject=subject if subject and subject != "All Subjects" else None,
            n_context_chunks=5
        ):
            if event["type"] == "citations":
                citations_html = format_citations_html(event["citations"])

            elif event["type"] == "token":
                final_answer += event["text"]
                history[-1] = {"role": "assistant", "content": final_answer + " ▌"}
                yield "", history, citations_html, meta_html, gr.update(visible=False)

            elif event["type"] == "done":
                final_answer = event["answer"]
                final_metadata = event["metadata"]
                history[-1] = {"role": "assistant", "content": final_answer}

                latency = final_metadata.get("latency_seconds", 0)
                latency_color = '#10b981' if latency < 3 else '#f59e0b' if latency < 5 else '#ef4444'
                meta_html = f"""
                <div class="meta-badges">
                    <span class="meta-badge latency" style="--badge-color: {latency_color}">
                        <span class="meta-icon">⚡</span> {latency:.1f}s
                    </span>
                    <span class="meta-badge sources">
                        <span class="meta-icon">📖</span> {final_metadata.get('num_sources', 0)} sources
                    </span>
                </div>
                """

                last_response = {
                    'question': message,
                    'answer': final_answer,
                    'grade': grade,
                    'subject': subject,
                    'latency': latency
                }
                yield "", history, citations_html, meta_html, gr.update(visible=True)

    except Exception as e:
        print(f"Error during chat: {e}")
        traceback.print_exc()
        history[-1] = {"role": "assistant", "content": f"❌ Error: {str(e)}"}
        yield "", history, "", "", gr.update(visible=False)


def submit_feedback(is_helpful):
    global last_response
    if not last_response:
        return "❌ No response to rate yet"

    try:
        save_feedback(
            question=last_response.get('question', ''),
            answer=last_response.get('answer', ''),
            helpful=is_helpful,
            session_id=session_id,
            grade=int(last_response.get('grade')) if last_response.get('grade') else None,
            subject=last_response.get('subject'),
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

**Feedback:**
- Total Ratings: {feedback_stats.get('total_feedback', 0)}
- Helpful Rate: {feedback_stats.get('helpful_rate', 0)}%
- Avg Latency: {feedback_stats.get('average_latency', 'N/A')}s
"""
    except Exception as e:
        return f"Error loading stats: {str(e)}"


def load_example(example_text):
    """Load an example question into the input box."""
    return example_text


# Premium CSS
PREMIUM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

body, .gradio-container, .main, .contain {
    background: linear-gradient(135deg, #f0f4ff 0%, #e8ecff 50%, #f5f7ff 100%) !important;
    min-height: 100vh;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

* {
    --primary: #6366f1;
    --primary-dark: #4f46e5;
    --secondary: #ec4899;
    --success: #10b981;
    --warning: #f59e0b;
    --error: #ef4444;
    --bg-card: #ffffff;
    --bg-hover: #f8fafc;
    --text-primary: #1e293b;
    --text-secondary: #475569;
    --border: #e2e8f0;
    --shadow: 0 4px 20px rgba(99, 102, 241, 0.1);
    --radius: 16px;
    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}

/* Header */
.header-main {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border-radius: 24px;
    padding: 32px;
    margin-bottom: 24px;
    text-align: center;
    animation: fadeInUp 0.6s ease-out;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 40px -10px rgba(99, 102, 241, 0.3);
}

.header-main::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(255, 255, 255, 0.15) 0%, transparent 70%);
    animation: float 6s ease-in-out infinite;
}

.header-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 8px 0;
    position: relative;
    z-index: 1;
    text-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.header-subtitle {
    color: rgba(255, 255, 255, 0.95);
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
    background: rgba(255, 255, 255, 0.2);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    color: #ffffff;
    border: 1px solid rgba(255, 255, 255, 0.25);
    transition: var(--transition);
}

.header-badge:hover {
    background: rgba(255, 255, 255, 0.3);
    transform: translateY(-2px);
}

/* Chatbot */
.chatbot-container {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden;
    box-shadow: var(--shadow);
}

.chatbot { background: transparent !important; }
.message { animation: fadeInUp 0.3s ease-out !important; }

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

/* Input */
.input-row { display: flex; gap: 12px; margin-top: 16px; }

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

textarea::placeholder { color: var(--text-secondary) !important; }

/* Buttons */
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

/* Example chips */
.example-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
}

.example-chip {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 8px 16px;
    font-size: 0.85rem;
    color: var(--text-secondary);
    cursor: pointer;
    transition: var(--transition);
}

.example-chip:hover {
    border-color: var(--primary);
    color: var(--primary);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.15);
}

/* Feedback */
.feedback-container {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px;
    background: var(--bg-card);
    border-radius: 16px;
    border: 1px solid var(--border);
    margin-top: 16px;
}

.feedback-label { color: var(--text-secondary); font-size: 0.9rem; }

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

.feedback-btn:hover { transform: scale(1.15) !important; border-color: var(--primary) !important; }
.feedback-btn.thumbs-up:hover { background: rgba(16, 185, 129, 0.2) !important; border-color: var(--success) !important; }
.feedback-btn.thumbs-down:hover { background: rgba(239, 68, 68, 0.2) !important; border-color: var(--error) !important; }

.feedback-toast {
    padding: 12px 20px;
    border-radius: 12px;
    font-size: 0.9rem;
    animation: fadeInUp 0.3s ease-out;
}

.feedback-toast.success { background: rgba(16, 185, 129, 0.2); color: var(--success); border: 1px solid rgba(16, 185, 129, 0.3); }
.feedback-toast.noted { background: rgba(245, 158, 11, 0.2); color: var(--warning); border: 1px solid rgba(245, 158, 11, 0.3); }

/* Meta badges */
.meta-badges { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 12px; }

.meta-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
    background: var(--bg-card);
    border: 1px solid var(--border);
    color: var(--text-secondary);
    transition: var(--transition);
}

.meta-badge:hover { transform: translateY(-2px); }
.meta-badge.latency { color: var(--badge-color, var(--success)); border-color: var(--badge-color, var(--success)); }
.meta-icon { font-size: 1rem; }

/* Citations */
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

.citations-header:hover { background: var(--bg-hover); }
.citations-icon { font-size: 1.2rem; }
.citations-title { font-weight: 600; color: var(--text-primary); flex: 1; }

.citations-count {
    background: var(--primary);
    color: white;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 500;
}

.citations-chevron { color: var(--text-secondary); transition: transform 0.3s ease; }
.citations-container.expanded .citations-chevron { transform: rotate(180deg); }

.citations-content { max-height: 0; overflow: hidden; transition: max-height 0.4s ease; }
.citations-container.expanded .citations-content { max-height: 600px; overflow-y: auto; }

.citation-card {
    padding: 16px 20px;
    border-top: 1px solid var(--border);
    animation: fadeInUp 0.3s ease-out;
}

.citation-head {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
    flex-wrap: wrap;
}

.citation-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 0.75rem;
    font-weight: 600;
    color: white;
}

.citation-subject {
    font-weight: 600;
    color: var(--text-primary);
    font-size: 0.9rem;
}

.citation-page {
    font-size: 0.75rem;
    color: var(--text-secondary);
    background: var(--bg-hover);
    padding: 2px 8px;
    border-radius: 6px;
}

.citation-source {
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin-bottom: 8px;
    font-style: italic;
}

.citation-text {
    color: var(--text-secondary);
    font-size: 0.9rem;
    line-height: 1.6;
    border-left: 3px solid var(--primary);
    padding-left: 12px;
}

/* Sidebar */
.sidebar {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    animation: fadeInUp 0.4s ease-out 0.2s backwards;
    box-shadow: var(--shadow);
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

/* Footer */
.footer {
    text-align: center;
    padding: 24px;
    margin-top: 24px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    animation: fadeInUp 0.4s ease-out 0.3s backwards;
}

.footer-text { color: var(--text-secondary); font-size: 0.9rem; }
.footer-text span { margin: 0 8px; opacity: 0.5; }

.accordion {
    background: var(--bg-hover) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    overflow: hidden;
}

.accordion button { background: transparent !important; color: var(--text-primary) !important; }

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--bg-hover); border-radius: 4px; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-secondary); }

@media (max-width: 768px) {
    .header-title { font-size: 1.75rem; }
    .header-subtitle { font-size: 0.95rem; }
    .user-message { margin-left: 10% !important; }
    .bot-message { margin-right: 10% !important; }
    .meta-badges { flex-direction: column; }
    .feedback-container { flex-wrap: wrap; justify-content: center; }
}

* { transition: background 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease; }
footer { display: none !important; }
.contain { background: transparent !important; }
.block { background: transparent !important; border: none !important; box-shadow: none !important; }
.wrap { background: transparent !important; }
label { color: var(--text-secondary) !important; }
"""

# Examples
EXAMPLES = [
    "What is photosynthesis?",
    "What is a rational number?",
    "Explain the water cycle.",
    "What were the causes of the First World War?",
    "Define democracy.",
    "What is the difference between a cell and an atom?",
]

# Build interface
with gr.Blocks(title="NCERT Doubt Solver") as demo:

    gr.HTML("""
        <div class="header-main">
            <h1 class="header-title">📚 NCERT Doubt Solver</h1>
            <p class="header-subtitle">Your AI-powered study companion for Grades 5-10</p>
            <div class="header-badges">
                <span class="header-badge">📖 NCERT Textbooks</span>
                <span class="header-badge">⚡ Fast Answers</span>
                <span class="header-badge">🎯 Accurate Citations</span>
                <span class="header-badge">🔍 Hybrid Retrieval</span>
            </div>
        </div>
    """)

    with gr.Row():
        # Main chat column
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="",
                height=500,
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
                    container=False,
                )
                send_btn = gr.Button(
                    "Send ✨",
                    variant="primary",
                    scale=1,
                    elem_classes=["primary-btn"],
                )

            # Example chips
            gr.HTML("""
                <div style="margin-top: 16px; margin-bottom: 8px; color: #475569; font-size: 0.9rem;">
                    <strong>Try these:</strong>
                </div>
            """)
            with gr.Row(elem_classes=["example-chips"]):
                for ex in EXAMPLES[:3]:
                    gr.Button(ex, size="sm", elem_classes=["example-chip"])

            # Meta info
            meta_display = gr.HTML("", elem_id="meta-display")

            # Citations
            citations_html = gr.HTML("", elem_classes=["citations-wrapper"])

            # Feedback row
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
                elem_classes=["dropdown-container"],
            )

            subject = gr.Dropdown(
                choices=["All Subjects", "Mathematics", "Science", "Social Science", "English", "EVS"],
                label="Subject",
                value="All Subjects",
                elem_classes=["dropdown-container"],
            )

            gr.HTML('<div style="margin: 20px 0; border-top: 1px solid var(--border);"></div>')

            clear_btn = gr.Button(
                "🗑️ Clear Chat",
                variant="secondary",
                elem_classes=["secondary-btn"],
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
                🤖 Groq LLM
                <span>•</span>
                Made with ❤️ in India
            </p>
        </div>
    """)

    # Wire up examples
    example_buttons = [c for c in demo.blocks.values() if isinstance(c, gr.Button) and c.elem_classes and "example-chip" in c.elem_classes]
    for i, btn in enumerate(example_buttons):
        if i < len(EXAMPLES):
            btn.click(
                fn=lambda ex=EXAMPLES[i]: ex,
                outputs=[msg],
            )

    # Main handlers
    msg.submit(
        chat,
        inputs=[msg, chatbot, grade, subject],
        outputs=[msg, chatbot, citations_html, meta_display, feedback_row],
    )
    send_btn.click(
        chat,
        inputs=[msg, chatbot, grade, subject],
        outputs=[msg, chatbot, citations_html, meta_display, feedback_row],
    )
    thumbs_up_btn.click(thumbs_up, outputs=[feedback_status])
    thumbs_down_btn.click(thumbs_down, outputs=[feedback_status])
    clear_btn.click(clear_chat, outputs=[chatbot, citations_html, meta_display, feedback_row])
    refresh_stats_btn.click(get_stats, outputs=[stats_display])


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Starting NCERT Doubt Solver Web App")
    print("=" * 60 + "\n")

    if not initialize_app():
        print("⚠️ Backend initialization failed. UI will still load.")

    print("\n🌐 Launching at http://localhost:7860\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )