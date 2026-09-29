from __future__ import annotations

import os
from typing import Any, Protocol
from dotenv import load_dotenv

from src.config.logging_config import setup_logger
from src.llm.ollama_client import OllamaClient
from src.llm.gemini_client import GeminiClient

logger = setup_logger()
load_dotenv()


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        ...


def get_llm_client(
    provider: str | None = None,
    model: str | None = None,
) -> LLMClient:
    """
    Factory to retrieve an LLM client instance (Gemini or Ollama).

    Parameters
    ----------
    provider : "gemini" | "ollama" | None
        If None, checks LLM_PROVIDER env var, or defaults to "gemini" if GEMINI_API_KEY/GOOGLE_API_KEY is present.
    model : str | None
        Specific model name. Defaults to "gemini-2.5-flash" (or "gemini-1.5-flash") for Gemini, or "qwen3:8b" for Ollama.
    """
    if provider is None:
        provider = os.getenv("LLM_PROVIDER")

    if not provider:
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            provider = "gemini"
        else:
            provider = "ollama"

    provider = provider.lower().strip()

    if provider == "gemini":
        target_model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        logger.info(f"Using Gemini LLM Backend (Model: {target_model})")
        return GeminiClient(model=target_model)
    elif provider == "ollama":
        target_model = model or os.getenv("OLLAMA_MODEL", "qwen3:8b")
        logger.info(f"Using Ollama LLM Backend (Model: {target_model})")
        return OllamaClient(model=target_model)
    else:
        raise ValueError(f"Unknown LLM provider '{provider}'. Must be 'gemini' or 'ollama'.")
