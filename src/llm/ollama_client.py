from __future__ import annotations

import ollama

from src.config.logging_config import setup_logger

logger = setup_logger()


class OllamaClient:
    """
    Client responsible for communicating with the local Ollama LLM.
    """

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
    ) -> None:

        self.model = model
        self.host = host

        self.client = ollama.Client(
            host=self.host
        )

        logger.info("=" * 50)
        logger.info("OLLAMA CLIENT INITIALIZED")
        logger.info("=" * 50)
        logger.info(f"Model : {self.model}")
        logger.info(f"Host  : {self.host}")

    def generate(
        self,
        prompt: str,
    ) -> str:
        """
        Send a prompt to Ollama and return the generated response.
        """

        if not prompt or not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty."
            )

        logger.info("=" * 50)
        logger.info("LLM GENERATION")
        logger.info("=" * 50)
        logger.info(
            f"Model : {self.model}"
        )
        logger.info(
            f"Prompt Length : {len(prompt)} characters"
        )

        try:

            response = self.client.generate(
                model=self.model,
                prompt=prompt,
            )

            generated_response = response["response"]

            logger.info(
                "LLM GENERATION COMPLETE"
            )

            logger.info(
                f"Response Length : "
                f"{len(generated_response)} characters"
            )

            return generated_response

        except Exception as exc:

            logger.error(
                f"LLM generation failed: {exc}"
            )

            raise