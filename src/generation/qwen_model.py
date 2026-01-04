"""
Qwen3-4B GGUF Model Client
Calls the WSL GGUF server for inference
"""
import requests
import subprocess
import time
import sys

WSL_SERVER_URL = "http://127.0.0.1:5000"

def check_server():
    """Check if WSL GGUF server is running"""
    try:
        response = requests.get(f"{WSL_SERVER_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_wsl_server():
    """Start the WSL GGUF server in background"""
    print("🔄 Starting WSL GGUF server...")
    
    # Get the Windows path and convert to WSL path
    import os
    win_path = os.path.abspath("wsl_gguf_server.py")
    # Convert to WSL path format
    wsl_path = "/mnt/" + win_path[0].lower() + win_path[2:].replace("\\", "/")
    
    # Start server in background
    cmd = f'wsl -d Ubuntu -- bash -c "source /tmp/gguf_env/bin/activate && python3 {wsl_path}"'
    
    # Note: This starts the server - user should run this in a separate terminal
    print(f"\n⚠️  Please run the following command in a separate terminal:")
    print(f"   {cmd}")
    print("\nThen press Enter when the server is ready...")
    input()

def generate_answer(context: str, question: str, max_new_tokens: int = 256) -> str:
    """Generate answer using WSL GGUF server"""
    
    if not check_server():
        print("❌ WSL GGUF server is not running!")
        print("   Start it with: wsl -d Ubuntu -- bash -c 'source /tmp/gguf_env/bin/activate && python3 /mnt/d/study/python/intel\\ unnati/wsl_gguf_server.py'")
        return "Error: Server not running"
    
    response = requests.post(
        f"{WSL_SERVER_URL}/generate",
        json={
            "context": context,
            "question": question,
            "max_tokens": max_new_tokens
        },
        timeout=120
    )
    
    if response.status_code == 200:
        return response.json()["answer"]
    else:
        return f"Error: {response.text}"


if __name__ == "__main__":
    # Quick test
    if not check_server():
        print("Server not running. Starting...")
        start_wsl_server()
    
    test_context = "Rational numbers are numbers that can be expressed as p/q where p and q are integers and q is not zero. For example, 1/2, 3/4, -5/6 are all rational numbers."
    test_question = "What is a rational number?"
    
    print(f"\nTest Question: {test_question}")
    answer = generate_answer(test_context, test_question)
    print(f"Answer: {answer}")
