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

Your job is to help students understand concepts from their NCERT textbooks.

Rules:
1. Use the provided context from NCERT textbooks as your primary source.
2. If the student asks a Math problem, you may APPLY the concepts from the context to solve it step by step. Show your work.
3. If the context does not contain information to answer a conceptual question, respond: "I cannot answer this from the provided context."
4. Keep answers grade-appropriate: simple, clear, and encouraging.
5. Always respond in English.
6. Cite the source grade and subject when relevant.
7. If the question is vague (like "Why is it important?") and the context does not clarify what "it" refers to, ask the student to rephrase their question with more detail."""

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
def generate_answer_groq_stream(
    context: str,
    question: str,
    max_retries: int = 3,
):
    """
    Streaming version of generate_answer_groq.
    Yields tokens as they arrive from Groq.
    Usage:
        for chunk in generate_answer_groq_stream(context, question):
            print(chunk, end="", flush=True)
    """
    llm = load_model()

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human", _USER_TEMPLATE),
    ])
    chain = prompt | llm

    last_error = None
    for attempt in range(max_retries):
        try:
            for chunk in chain.stream({"context": context, "question": question}):
                # chunk is an AIMessageChunk; .content holds the delta
                text = chunk.content if hasattr(chunk, "content") else str(chunk)
                if text:
                    yield text
            return  # stream finished successfully
        except Exception as e:
            last_error = e
            err = str(e).lower()
            if "429" in err or "rate" in err or "timeout" in err or "503" in err:
                wait = 10 * (attempt + 1)
                logger.warning(f"Rate limited on stream (attempt {attempt+1}/{max_retries}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"Non-retryable error on stream: {e}")
                raise

    logger.error(f"Stream failed after {max_retries} retries: {last_error}")
    raise last_error    