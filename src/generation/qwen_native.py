"""
Native Qwen3-4B Generation Module
Uses transformers library with GPU support and thinking process parsing.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model and tokenizer cache
_model = None
_tokenizer = None
# Defaulting to 1.5B due to system OOM with 4B model during loading
_model_name = "Qwen/Qwen2.5-1.5B-Instruct" 

def load_model(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """Load the model and tokenizer globally"""
    global _model, _tokenizer, _model_name
    
    if _model is not None:
        return _model, _tokenizer
    
    logger.info(f"Loading native model: {model_name}...")
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        _model_name = model_name
        logger.info(f"✅ Model {model_name} loaded successfully on {_model.device}")
        return _model, _tokenizer
    except Exception as e:
        logger.error(f"Failed to load native model: {e}")
        raise e

def generate_answer_native(
    context: str, 
    question: str, 
    max_new_tokens: int = 2048
) -> Dict[str, Any]:
    """
    Generate answer using native Qwen model with thinking process parsing.
    
    Args:
        context: The retrieved context from textbooks
        question: The student's question
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Dictionary containing 'answer' and 'thinking_content'
    """
    global _model, _tokenizer
    
    if _model is None:
        load_model(_model_name)
        
    # Prepare prompt
    messages = [
        {
            "role": "system", 
            "content": (
                "You are a helpful NCERT tutor for Grade 5-10 students. "
                "Answer the question based ONLY on the provided context. "
                "Explain simply and clearly."
            )
        },
        {
            "role": "user", 
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]
    
    # Apply template
    text = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True 
    )
    
    # Tokenize and move to device
    model_inputs = _tokenizer([text], return_tensors="pt").to(_model.device)
    
    # Generate
    generated_ids = _model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens
    )
    
    # Extract output tokens (exclude input)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    
    # Parse thinking content (looking for </think> token)
    # Note: Qwen think tag handling logic from user request
    thinking_content = ""
    content = ""
    
    try:
        # Try to find the closing tag ID (151668 for Qwen 2.5/3 usually, but safer to decode first or search)
        # User provided specific logic: index = len(output_ids) - output_ids[::-1].index(151668)
        # We will try user's logic first
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
            thinking_content = _tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
            content = _tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
        except ValueError:
            # Fallback if specific ID not found (might vary by tokenizer version)
            full_text = _tokenizer.decode(output_ids, skip_special_tokens=True)
            if "</think>" in full_text:
                parts = full_text.split("</think>")
                thinking_content = parts[0].replace("<think>", "").strip()
                content = parts[1].strip()
            else:
                content = full_text.strip()
                
    except Exception as e:
        logger.warning(f"Error parsing thinking content: {e}")
        content = _tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    
    # Clean up any residual tags in content just in case
    content = content.replace("<think>", "").replace("</think>", "").strip()
    
    return {
        "answer": content,
        "thinking": thinking_content
    }

if __name__ == "__main__":
    # Test block
    print("Testing Qwen Native...")
    try:
        load_model()
        res = generate_answer_native(
            "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar.",
            "What do plants need for photosynthesis?"
        )
        print(f"\nThinking:\n{res['thinking']}")
        print(f"\nAnswer:\n{res['answer']}")
    except Exception as e:
        print(f"Test failed: {e}")
