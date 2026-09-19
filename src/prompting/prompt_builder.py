from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from src.config.logging_config import setup_logger
from src.retrieval.retriever import RetrievedDocument

logger = setup_logger()


class PromptBuilder:
    """
    Builds the final prompt for the ARIA LLM using
    a Jinja2 prompt template and retrieved context.

    Responsibilities
    ----------------
    - Load the Jinja2 template.
    - Prepare retrieved documents.
    - Inject query and context into the template.
    - Return the final rendered prompt.

    Does NOT
    --------
    - Retrieve documents.
    - Generate embeddings.
    - Call the LLM.
    - Generate the final answer.
    """

    def __init__(
        self,
        template_name: str = "triage_prompt.j2",
    ) -> None:

        # Project root:
        # RAG_Pipeline/
        project_root = Path(__file__).resolve().parents[2]

        template_directory = project_root / "src" / "templates"

        if not template_directory.exists():
            raise FileNotFoundError(
                f"Template directory not found: "
                f"{template_directory}"
            )

        self.environment = Environment(
            loader=FileSystemLoader(template_directory),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )

        try:
            self.template = self.environment.get_template(
                template_name
            )
        except Exception as exc:
            raise FileNotFoundError(
                f"Template '{template_name}' "
                f"could not be loaded."
            ) from exc

        logger.info("=" * 50)
        logger.info("PROMPT BUILDER INITIALIZED")
        logger.info("=" * 50)
        logger.info(f"Template : {template_name}")
        logger.info(
            f"Template Directory : {template_directory}"
        )
        logger.info("=" * 50)

    def build(
        self,
        query: str,
        retrieved_documents: list[RetrievedDocument],
    ) -> str:
        """
        Build the final LLM prompt.

        Parameters
        ----------
        query
            User's security query.

        retrieved_documents
            Documents retrieved from the knowledge base.

        Returns
        -------
        str
            Fully rendered prompt.
        """

        if query is None:
            raise ValueError(
                "Query cannot be None."
            )

        if not isinstance(query, str):
            raise TypeError(
                "Query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if not isinstance(
            retrieved_documents,
            list,
        ):
            raise TypeError(
                "retrieved_documents must be a list."
            )

        logger.info("=" * 50)
        logger.info("BUILDING TRIAGE PROMPT")
        logger.info("=" * 50)

        logger.info(
            f"Query : {query}"
        )

        logger.info(
            f"Retrieved Documents : "
            f"{len(retrieved_documents)}"
        )

        context: list[dict[str, Any]] = []

        for result in retrieved_documents:

            source = result.metadata.get(
                "source",
                "UNKNOWN",
            )

            document_type = result.metadata.get(
                "type",
                "UNKNOWN",
            )

            name = result.metadata.get(
                "name",
                "UNKNOWN",
            )

            context.append(
                {
                    "source": source,
                    "type": document_type,
                    "name": name,
                    "distance": result.distance,
                    "document": result.document,
                }
            )

        prompt = self.template.render(
            query=query,
            retrieved_documents=context,
        )

        logger.info(
            f"Prompt Length : {len(prompt)} characters"
        )

        logger.info("=" * 50)
        logger.info("TRIAGE PROMPT BUILT")
        logger.info("=" * 50)

        return prompt