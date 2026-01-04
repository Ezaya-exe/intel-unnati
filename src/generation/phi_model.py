from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"  # 3.8B, good for local use[web:169]

print("🔄 Loading Phi-3 Mini 4K Instruct... (first time will download, can take a while)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",   # uses GPU if available, else CPU
)

print("✅ Phi-3 loaded!")

SYSTEM_PROMPT = (
    "You are a helpful NCERT assistant for students in grades 5-10. "
    "Use ONLY the provided NCERT textbook context to answer. "
    "If the answer is not clearly in the context, say: "
    "\"I don't know – please ask an NCERT-related question.\" "
)

def build_phi_chat_prompt(context: str, question: str) -> str:
    # Phi-3 prefers chat-style messages; we format manually[web:169][web:177]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Context from NCERT textbooks:\n"
                f"{context}\n\n"
                f"Question: {question}\n\n"
                "Answer in a simple way suitable for a school student."
            ),
        },
    ]

    # Simple chat template: <|system|>...<|end|><|user|>...<|end|><|assistant|>
    # (Phi-3 docs suggest using role tags like this)[web:177]
    formatted = ""
    for m in messages:
        formatted += f"<|{m['role']}|>\n{m['content']}<|end|>\n"
    formatted += "<|assistant|>\n"
    return formatted

def generate_answer(context: str, question: str, max_new_tokens: int = 256) -> str:
    prompt = build_phi_chat_prompt(context, question)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=3500,  # leave room for generation
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Return text after the last <|assistant|> tag if present
    if "<|assistant|>" in full:
        return full.split("<|assistant|>")[-1].strip()
    return full.strip()
