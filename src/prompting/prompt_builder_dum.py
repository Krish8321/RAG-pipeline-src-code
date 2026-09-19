from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from src.config.logging_config import setup_logger

logger = setup_logger()

# Fields the template accesses without a `default` filter or `if` guard —
# StrictUndefined will raise if any of these are missing from a document.
REQUIRED_DOCUMENT_FIELDS = (
    "rank",
    "source",
    "type",
    "name",
    "rerank_score",
    "document",
)

# Fields the template guards with `if`/`default`, so they're allowed to
# be absent — listed here only so validation errors can tell the two
# categories apart if you ever want to warn on missing optional fields.
OPTIONAL_DOCUMENT_FIELDS = (
    "matched_behavior",
    "relevance_tier",
)


class PromptBuilder:
    """
    Builds the final ARIA LLM prompt from an alert and
    structured RAG context.

    Responsibilities
    ----------------
    - Load the Jinja2 prompt template.
    - Inject the alert text and optional alert metadata.
    - Inject structured retrieved evidence.
    - Render the final prompt.

    Does NOT
    --------
    - Retrieve documents.
    - Generate queries.
    - Deduplicate documents.
    - Rerank documents.
    - Build or modify security evidence.
    - Call the LLM.
    """

    def __init__(
        self,
        template_name: str = "final_prompt.j2",
    ) -> None:

        project_root = Path(
            __file__
        ).resolve().parents[2]

        template_directory = (
            project_root
            / "src"
            / "templates"
        )

        if not template_directory.exists():
            raise FileNotFoundError(
                "Template directory not found: "
                f"{template_directory}"
            )

        self.environment = Environment(
            loader=FileSystemLoader(
                template_directory
            ),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )

        try:
            self.template = (
                self.environment.get_template(
                    template_name
                )
            )
        except Exception as exc:
            raise FileNotFoundError(
                f"Template '{template_name}' "
                "could not be loaded."
            ) from exc

        logger.info("=" * 60)
        logger.info("PROMPT BUILDER INITIALIZED")
        logger.info("=" * 60)

        logger.info(
            f"Template : {template_name}"
        )

        logger.info(
            f"Template Directory : "
            f"{template_directory}"
        )

        logger.info("=" * 60)

    @staticmethod
    def _validate_documents(
        context: list[dict[str, Any]],
    ) -> None:
        """
        Validate that every document in context has the fields the
        template renders unconditionally. Raising here with a clear
        message is much easier to debug than a StrictUndefined
        traceback from inside Jinja.
        """

        for index, document in enumerate(context):
            if not isinstance(document, dict):
                raise TypeError(
                    f"context[{index}] must be a dict, "
                    f"got {type(document).__name__}."
                )

            missing = [
                field
                for field in REQUIRED_DOCUMENT_FIELDS
                if field not in document
            ]

            if missing:
                raise ValueError(
                    f"context[{index}] (name="
                    f"{document.get('name', '<unknown>')!r}) is "
                    f"missing required field(s): {missing}. "
                    f"Required fields: {list(REQUIRED_DOCUMENT_FIELDS)}."
                )

    @staticmethod
    def _normalize_documents(
        context: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Ensure optional fields exist on every document before render.

        With `StrictUndefined`, a bare `{% if result.matched_behavior %}`
        raises the instant the key is missing, because boolean
        coercion touches the Undefined object directly — unlike the
        `| default(...)` filter, which checks "is this undefined"
        first and never triggers the raise. Rather than relying on
        every template author to remember that distinction, we
        guarantee the keys exist here so either template style is
        safe.

        Returns new dicts; does not mutate the caller's context list.
        """

        normalized = []

        for document in context:
            normalized_document = dict(document)
            normalized_document.setdefault("matched_behavior", None)
            normalized_document.setdefault("relevance_tier", "unrated")
            normalized.append(normalized_document)

        return normalized

    def build(
        self,
        alert_text: str,
        context: list[dict[str, Any]],
        alert_metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Build the final ARIA LLM prompt.

        Parameters
        ----------
        alert_text
            Original security alert text. Rendered into the template's
            `alert_text` variable.

        context
            Structured evidence produced by ContextBuilder. Each item
            must contain: rank, source, type, name, rerank_score,
            document. `matched_behavior` and `relevance_tier` are
            optional and rendered only if present. Rendered into the
            template's `retrieved_documents` variable.

        alert_metadata
            Optional dict of alert-level metadata (e.g. source,
            timestamp, severity). Rendered into the template's
            `alert_metadata` variable; omitted from the prompt
            entirely if None or empty.

        Returns
        -------
        str
            Fully rendered LLM prompt.
        """

        if alert_text is None:
            raise ValueError(
                "alert_text cannot be None."
            )

        if not isinstance(alert_text, str):
            raise TypeError(
                "alert_text must be a string."
            )

        if not alert_text.strip():
            raise ValueError(
                "alert_text cannot be empty."
            )

        if context is None:
            raise ValueError(
                "Context cannot be None."
            )

        if not isinstance(context, list):
            raise TypeError(
                "Context must be a list."
            )

        if alert_metadata is not None and not isinstance(
            alert_metadata, dict
        ):
            raise TypeError(
                "alert_metadata must be a dict or None."
            )

        self._validate_documents(context)
        normalized_context = self._normalize_documents(context)

        logger.info("=" * 60)
        logger.info("BUILDING TRIAGE PROMPT")
        logger.info("=" * 60)

        logger.info(
            f"Alert : {alert_text}"
        )

        logger.info(
            f"Alert Metadata : {alert_metadata or {}}"
        )

        logger.info(
            f"Context Documents : "
            f"{len(context)}"
        )

        prompt = self.template.render(
            alert_text=alert_text,
            alert_metadata=alert_metadata,
            retrieved_documents=normalized_context,
        )

        logger.info(
            f"Prompt Length : "
            f"{len(prompt)} characters"
        )

        logger.info("=" * 60)
        logger.info("TRIAGE PROMPT BUILT")
        logger.info("=" * 60)

        return prompt