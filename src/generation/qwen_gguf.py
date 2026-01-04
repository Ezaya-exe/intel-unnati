"""
Native GGUF Generation Module (WSL/Linux)
Uses llama-cpp-python with GPU acceleration.
"""
import logging
from typing import Dict, Any, Optional
import os

# Try to import llama_cpp, handle if not installed yet (during migration)
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model cache
_model_path = "/mnt/d/study/python/intel unnati/models/Qwen3-4B-Q4_K_M.gguf"
_llm = None

def load_model(model_path: str = None):
    """Load the GGUF model globally"""
    global _llm, _model_path
    
    if model_path:
        _model_path = model_path
        
    if _llm is not None:
        return _llm
    
    if not os.path.exists(_model_path):
        raise FileNotFoundError(f"Model not found at: {_model_path}")

    logger.info(f"Loading GGUF model from {_model_path}...")
    try:
        if Llama is None:
            raise ImportError("llama-cpp-python not installed")
            
        # Initialize Llama (n_gpu_layers=-1 for max GPU offset)
        _llm = Llama(
            model_path=_model_path,
            n_gpu_layers=-1,      # Offload all layers to GPU
            n_ctx=4096,           # Context window
            n_threads=6,          # CPU threads
            verbose=True
        )
        logger.info("✅ GGUF Model loaded successfully!")
        return _llm
    except Exception as e:
        logger.error(f"Failed to load GGUF model: {e}")
        raise e

def generate_answer_gguf(
    context: str, 
    question: str, 
    max_tokens: int = 1024
) -> Dict[str, Any]:
    """
    Generate answer using GGUF model with think parsing.
    """
    global _llm
    if _llm is None:
        load_model()

    # Prompt format for Qwen3 (ChatML format)
    # Using /no_think for faster, non-reasoning mode responses
    
    system_prompt = (
        "You are a helpful NCERT tutor for Grade 5-10 students. "
        "Answer the question based ONLY on the provided context. "
        "If the answer is not in the context, say you don't know. "
        "Explain simply and clearly for school students."
    )
    
    # Add /no_think to disable thinking mode for faster responses
    prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
Context:
{context}

Question: {question} /no_think<|im_end|>
<|im_start|>assistant
"""

    logger.info("Generating response...")
    output = _llm(
        prompt,
        max_tokens=max_tokens,
        stop=["<|im_end|>"],
        echo=False,
        temperature=0.7,
        top_p=0.8,
        top_k=20
    )
    
    full_text = output['choices'][0]['text']
    
    # Parse thinking tags
    thinking_content = ""
    answer_content = full_text
    
    if "<think>" in full_text:
        try:
            parts = full_text.split("</think>")
            if len(parts) > 1:
                thinking_content = parts[0].replace("<think>", "").strip()
                answer_content = parts[1].strip()
            else:
                # Open tag but no close tag?
                thinking_content = full_text.replace("<think>", "").strip()
                answer_content = "" 
        except Exception:
            pass
            
    return {
        "answer": answer_content,
        "thinking": thinking_content
    }
