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


from src.utils.alert_parser import parse_alert


class PromptBuilder:
    """
    Builds the final ARIA LLM prompt from an alert and
    structured RAG context.
    """

    def __init__(
        self,
        template_name: str = "aria_triage_prompt_v2.j2",
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
        template renders unconditionally.
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
        context: list[dict[str, Any]] | dict[str, Any],
        alert_metadata: dict[str, Any] | None = None,
        technique_consensus: str | None = None,
    ) -> str:
        """
        Build the final ARIA LLM prompt.
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

        parsed_alert = parse_alert(alert_metadata if isinstance(alert_metadata, dict) else alert_text)
        effective_metadata = dict(parsed_alert.metadata_dict)
        if alert_metadata:
            effective_metadata.update(alert_metadata)

        computed_risk_score = 50
        risk_factors = []
        escalation_recommendation = "Escalate to Tier-2 Analyst Review"
        technique_support_info = {}
        risk_breakdown = {}
        primary_severity_summary = "NONE"
        supporting_severity_summary = "NONE"

        if isinstance(context, dict):
            documents = context.get("documents", [])
            technique_consensus = context.get("technique_consensus", technique_consensus)
            technique_support_info = context.get("technique_support_info", {})
            computed_risk_score = context.get("computed_risk_score", 50)
            risk_breakdown = context.get("risk_breakdown", {})
            risk_factors = context.get("risk_factors", [])
            primary_severity_summary = context.get("primary_severity_summary", "NONE")
            supporting_severity_summary = context.get("supporting_severity_summary", "NONE")
            escalation_recommendation = context.get("escalation_recommendation", "Escalate to Tier-2 / Analyst Investigation Required")
        elif isinstance(context, list):
            documents = context
        else:
            raise TypeError(
                "Context must be a list or dict."
            )

        if alert_metadata is not None and not isinstance(
            alert_metadata, dict
        ):
            raise TypeError(
                "alert_metadata must be a dict or None."
            )

        self._validate_documents(documents)
        normalized_context = self._normalize_documents(documents)

        logger.info("=" * 60)
        logger.info("BUILDING TRIAGE PROMPT")
        logger.info("=" * 60)

        logger.info(
            f"Alert : {alert_text}"
        )

        logger.info(
            f"Alert Metadata : {effective_metadata}"
        )

        logger.info(
            f"Context Documents : "
            f"{len(documents)}"
        )

        if technique_consensus:
            logger.info(
                f"Technique Consensus : {technique_consensus}"
            )

        logger.info(
            f"Computed Risk Score : {computed_risk_score}/100"
        )
        logger.info(
            f"Escalation Recommendation : {escalation_recommendation}"
        )

        prompt = self.template.render(
            alert_text=alert_text,
            alert_metadata=effective_metadata,
            parsed_alert=parsed_alert,
            upstream_risk_score=parsed_alert.upstream_risk_score,
            upstream_mitre_techniques=parsed_alert.upstream_mitre_techniques,
            formatted_telemetry=parsed_alert.formatted_telemetry,
            retrieved_documents=normalized_context,
            technique_consensus=technique_consensus,
            technique_support_info=technique_support_info,
            computed_risk_score=computed_risk_score,
            risk_breakdown=risk_breakdown,
            risk_factors=risk_factors,
            primary_severity_summary=primary_severity_summary,
            supporting_severity_summary=supporting_severity_summary,
            escalation_recommendation=escalation_recommendation,
        )

        logger.info(
            f"Prompt Length : "
            f"{len(prompt)} characters"
        )

        logger.info("=" * 60)
        logger.info("TRIAGE PROMPT BUILT")
        logger.info("=" * 60)

        return prompt