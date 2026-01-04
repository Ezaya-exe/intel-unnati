#!/usr/bin/env python3
"""
GGUF Model Server for Qwen3-4B
Runs in WSL with llama-cpp-python
Provides REST API for Windows client
"""
from flask import Flask, request, jsonify
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os

app = Flask(__name__)

# Model configuration
MODEL_REPO = "unsloth/Qwen3-4B-GGUF"
MODEL_FILE = "Qwen3-4B-Q4_K_M.gguf"
MODEL_DIR = "/tmp/models"

# Global model instance
model = None

def load_model():
    global model
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    
    # Download model if not exists
    if not os.path.exists(model_path):
        print(f"🔄 Downloading {MODEL_FILE}...")
        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir=MODEL_DIR
        )
        print(f"✅ Downloaded to: {model_path}")
    else:
        print(f"✅ Model found at: {model_path}")
    
    print("🔄 Loading Qwen3-4B GGUF model...")
    
    # Load model (CPU mode in WSL - GPU passthrough requires extra setup)
    model = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=8,
        verbose=False
    )
    
    print("✅ Qwen3-4B loaded!")

SYSTEM_PROMPT = (
    "You are a helpful NCERT assistant for students in grades 5-10. "
    "Use ONLY the provided NCERT textbook context to answer. "
    "If the answer is not clearly in the context, say: "
    "\"I don't know – please ask an NCERT-related question.\" "
)

def build_qwen_prompt(context: str, question: str) -> str:
    """Build prompt in Qwen3 chat format"""
    prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
Context from NCERT textbooks:
{context}

Question: {question}

Answer in a simple way suitable for a school student.<|im_end|>
<|im_start|>assistant
"""
    return prompt

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model": MODEL_FILE})

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    context = data.get('context', '')
    question = data.get('question', '')
    max_tokens = data.get('max_tokens', 256)
    
    prompt = build_qwen_prompt(context, question)
    
    response = model(
        prompt,
        max_tokens=max_tokens,
        temperature=0.3,
        top_p=0.9,
        stop=["<|im_end|>", "<|im_start|>"],
        echo=False
    )
    
    answer = response["choices"][0]["text"].strip()
    
    return jsonify({
        "answer": answer,
        "model": MODEL_FILE,
        "tokens_used": response.get("usage", {})
    })

if __name__ == "__main__":
    load_model()
    print("\n🚀 Starting GGUF server on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
