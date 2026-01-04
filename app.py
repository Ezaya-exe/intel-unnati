"""
NCERT Multilingual Doubt Solver - Gradio Web Application
Features:
- Chat interface with conversation history
- Grade, Subject, Language selection
- Citations display
- Feedback system (thumbs up/down)
- Latency indicator
- Mobile-responsive design
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
    """Format citations as HTML for display"""
    if not citations:
        return ""
    
    html = "<details><summary>📚 <b>Sources & Citations</b> (Click to expand)</summary><br>"
    for cite in citations:
        html += f"""
        <div style='background: #f0f4f8; padding: 10px; margin: 5px 0; border-radius: 8px; border-left: 4px solid #3b82f6;'>
            <b>Source {cite['index']}</b> - Grade {cite['grade']}, {cite['subject']} ({cite['language']})<br>
            <i style='color: #666; font-size: 0.9em;'>{cite['text']}</i>
        </div>
        """
    html += "</details>"
    return html

def chat(message, history, grade, subject, language):
    """Process chat message and return response"""
    global last_response, solver
    
    if not message.strip():
        return "", history, "", ""
    
    if solver is None:
        return "", history or [], "", "⚠️ Backend not initialized. Check server logs."
    
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
        meta_text = f"⏱️ {metadata['latency_seconds']}s | 🌐 {LANGUAGE_NAMES.get(metadata['detected_language'], 'Unknown')} | 📖 {metadata['num_sources']} sources"
        
        history = history or []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})
        
        return "", history, citations_html, meta_text
        
    except Exception as e:
        print(f"Error during chat: {e}")
        traceback.print_exc()
        error_msg = f"❌ Error: {str(e)}"
        history = history or []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return "", history, "", "Error processing request"

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
        return f"{emoji} Feedback recorded! (ID: {feedback_id})"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def thumbs_up(): return submit_feedback(True)
def thumbs_down(): return submit_feedback(False)

def clear_chat():
    global solver, last_response
    if solver:
        solver.clear_history()
    last_response = {}
    return [], "", ""

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
- Total Chunks: {db_stats.get('total_chunks', 0)}
- Grades: {db_stats.get('grades', {})}
- Subjects: {db_stats.get('subjects', {})}

**Feedback:**
- Total Ratings: {feedback_stats.get('total_feedback', 0)}
- Helpful Rate: {feedback_stats.get('helpful_rate', 0)}%
- Avg Latency: {feedback_stats.get('average_latency', 'N/A')}s
"""
    except Exception as e:
        return f"Error loading stats: {str(e)}"

# Define custom CSS for mobile responsiveness
CUSTOM_CSS = """
/* Mobile-first responsive design */
@media (max-width: 768px) {
    .gradio-container {
        padding: 10px !important;
    }
    
    /* Stack columns vertically on mobile */
    .contain > .flex {
        flex-direction: column !important;
    }
    
    /* Make chat take full width */
    .chatbot {
        height: 350px !important;
    }
    
    /* Hide sidebar on very small screens, show as collapsible */
    .sidebar-column {
        order: -1 !important;
    }
    
    /* Larger touch targets */
    button {
        min-height: 44px !important;
        font-size: 16px !important;
    }
    
    /* Bigger input text for mobile */
    textarea, input {
        font-size: 16px !important;
    }
    
    /* Feedback buttons inline */
    .feedback-row {
        flex-wrap: nowrap !important;
    }
    
    .feedback-btn {
        min-width: 60px !important;
    }
}

/* Tablet styles */
@media (min-width: 769px) and (max-width: 1024px) {
    .chatbot {
        height: 380px !important;
    }
}

/* Desktop styles */
@media (min-width: 1025px) {
    .chatbot {
        height: 450px !important;
    }
}

/* Custom styling for all devices */
.citation-box {
    max-height: 200px;
    overflow-y: auto;
}

.feedback-btn {
    padding: 8px 16px !important;
    font-size: 20px !important;
}

/* Header styling */
.header-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
}

.header-container h1 {
    margin: 0 !important;
    color: white !important;
}

.header-container p {
    margin: 5px 0 0 0 !important;
    opacity: 0.9;
}

/* Filter section */
.filter-section {
    background: #f8fafc;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
}

/* Footer styling */
.footer {
    text-align: center;
    margin-top: 20px;
    padding: 15px;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 10px;
}

/* Input styling */
.input-container textarea {
    border-radius: 12px !important;
    border: 2px solid #e2e8f0 !important;
}

.input-container textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
}

/* Send button styling */
.send-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 12px !important;
}
"""

# Define Gradio interface with custom theme
with gr.Blocks(
    title="NCERT Doubt Solver",
    css=CUSTOM_CSS,
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter")
    )
) as demo:
    
    gr.HTML("""
        <div class="header-container" style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 12px; margin-bottom: 20px;">
            <h1 style="margin: 0; color: white;">📚 NCERT Multilingual Doubt Solver</h1>
            <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 0;">Ask doubts from your NCERT textbooks (Grades 5-10) in Hindi or English!</p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Chat",
                height=400,
                avatar_images=(None, "https://em-content.zobj.net/source/twitter/376/woman-teacher-light-skin-tone_1f469-1f3fb-200d-1f3eb.png")
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Your Question",
                    placeholder="Type your doubt here... (e.g., What is photosynthesis?)",
                    scale=4,
                    show_label=False
                )
                send_btn = gr.Button("Send 📤", variant="primary", scale=1)
            
            citations_html = gr.HTML(label="Citations", elem_classes=["citation-box"])
            
            with gr.Row():
                meta_display = gr.Markdown("", elem_id="meta-display")
                with gr.Row(elem_classes=["feedback-row"]):
                    gr.Markdown("**Rate this answer:**")
                    thumbs_up_btn = gr.Button("👍", elem_classes=["feedback-btn"])
                    thumbs_down_btn = gr.Button("👎", elem_classes=["feedback-btn"])
                    feedback_status = gr.Markdown("")
        
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Filters")
            grade = gr.Dropdown(choices=[""] + [str(i) for i in range(5, 11)], label="Grade", value="", info="Select grade level (5-10)")
            subject = gr.Dropdown(choices=["All Subjects", "Mathematics", "Science", "Social Science", "Hindi", "English", "Sanskrit"], label="Subject", value="All Subjects")
            language = gr.Dropdown(choices=["Auto-detect"] + list(LANGUAGE_NAMES.values()), label="Language", value="Auto-detect", info="Auto-detects from your question")
            
            gr.Markdown("---")
            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
            gr.Markdown("---")
            
            with gr.Accordion("📊 Statistics", open=False):
                stats_display = gr.Markdown("")
                refresh_stats_btn = gr.Button("Refresh Stats", size="sm")
    
    gr.HTML("""
        <div style="text-align: center; margin-top: 20px; padding: 10px; background: #f0f4f8; border-radius: 8px;">
            <p style="margin: 0; color: #666;">
                🎓 Built for Intel Unnati Program | 📖 Powered by NCERT Textbooks | 🤖 Using Qwen3-4B LLM
            </p>
        </div>
    """)
    
    msg.submit(chat, inputs=[msg, chatbot, grade, subject, language], outputs=[msg, chatbot, citations_html, meta_display])
    send_btn.click(chat, inputs=[msg, chatbot, grade, subject, language], outputs=[msg, chatbot, citations_html, meta_display])
    thumbs_up_btn.click(thumbs_up, outputs=[feedback_status])
    thumbs_down_btn.click(thumbs_down, outputs=[feedback_status])
    clear_btn.click(clear_chat, outputs=[chatbot, citations_html, meta_display])
    refresh_stats_btn.click(get_stats, outputs=[stats_display])

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting NCERT Doubt Solver Web App")
    print("="*60)
    print("\n⚠️  Make sure the WSL GGUF server is running:")
    print("   wsl -d Ubuntu -- bash -c 'source ~/gguf_env/bin/activate && cd /mnt/d/study/python/intel\\ unnati && python3 wsl_gguf_server.py'")
    print("\n" + "="*60 + "\n")
    
    # Initialize backend first
    if not initialize_app():
        print("⚠️ Failed to initialize backend components. Check errors above.")
        # We don't exit, we let the UI load so user sees error in UI
    
    print("Starting Gradio launch...")
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )
    except Exception as e:
        print(f"FAILED TO LAUNCH: {e}")
        traceback.print_exc()
    
    print(" Application is running. Press CTRL+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping application...")
