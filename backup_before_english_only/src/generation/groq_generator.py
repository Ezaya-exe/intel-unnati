"""
Groq Generation Module
Uses Groq's fast inference API via LangChain.
"""
import logging
import os
import re
import time
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_llm = None

_SYSTEM_PROMPT = """You are a friendly NCERT tutor for Grade 5-10 students in India.
Answer ONLY using the provided context from NCERT textbooks.
If the context does not contain the answer, respond: "I cannot answer this from the provided context."
Do NOT use outside knowledge or infer beyond what is stated.
Write in the same language as the student's question (Hindi or English).
Keep answers grade-appropriate: simple, clear, and encouraging.
Cite the source grade and subject when relevant."""

_USER_TEMPLATE = """Context from NCERT textbooks:
{context}

Student's Question: {question}

Answer:"""


def load_model() -> ChatGroq:
    """Load and cache the Groq model globally."""
    global _llm

    if _llm is not None:
        return _llm

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY environment variable not set.")
        raise ValueError("GROQ_API_KEY environment variable is required")

    model_name = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

    logger.info(f"Initializing Groq model: {model_name}")

    llm_kwargs = {
        "model": model_name,
        "groq_api_key": api_key,
        "temperature": 0.1,
        "max_tokens": 500,
    }

    if any(tag in model_name.lower() for tag in ["qwen3", "gpt-oss", "deepseek", "r1"]):
        llm_kwargs["reasoning_effort"] = "low"
        logger.info(f"Reasoning model detected: {model_name}. Set reasoning_effort=low.")

    _llm = ChatGroq(**llm_kwargs)
    logger.info("Groq model initialized successfully")
    return _llm


def generate_answer_groq(
    context: str,
    question: str,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Generate an answer using Groq with retry-with-backoff on rate limits."""
    llm = load_model()

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human", _USER_TEMPLATE),
    ])
    chain = prompt | llm

    last_error = None
    for attempt in range(max_retries):
        try:
            # Diagnostic: input size
            prompt_length = len(context) + len(question)
            logger.info(f"Prompt chars: {prompt_length}, approx tokens: {prompt_length // 4}")

            response = chain.invoke({
                "context": context,
                "question": question,
            })

            # Extract answer text FIRST
            answer_text = response.content if hasattr(response, "content") else str(response)

            # Strip thinking blocks
            answer_text = re.sub(r" thinking.*?", "", answer_text, flags=re.DOTALL).strip()
            answer_text = re.sub(r"^Thinking:.*?\n\n", "", answer_text, flags=re.DOTALL).strip()

            # Diagnostic: output size (AFTER answer_text is defined)
            logger.info(f"Response words: {len(answer_text.split())}")

            return {
                "answer": answer_text,
                "thinking": None,
            }

        except Exception as e:
            last_error = e
            err = str(e).lower()

            if "429" in err or "rate" in err or "timeout" in err or "503" in err:
                wait = 10 * (attempt + 1)
                logger.warning(f"Rate limited (attempt {attempt+1}/{max_retries}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"Non-retryable error: {e}")
                raise

    logger.error(f"Failed after {max_retries} retries: {last_error}")
    raise last_error