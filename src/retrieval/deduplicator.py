from __future__ import annotations

import re
from typing import Any

from src.config.logging_config import setup_logger
from src.retrieval.retriever import RetrievedDocument

logger = setup_logger()


class DocumentDeduplicator:
    """
    Removes duplicate retrieved documents.

    If the same document is retrieved multiple times, the
    version with the lowest distance is retained because a
    lower distance represents a better similarity match.

    Duplicate identity is determined using:
    1. source + metadata document/chunk ID, when available
    2. source + normalized document text, as fallback
    """

    def deduplicate(
        self,
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """
        Deduplicate retrieved documents.

        Parameters
        ----------
        documents
            Retrieved documents from multi-query retrieval.

        Returns
        -------
        list[RetrievedDocument]
            Unique documents containing only the best
            occurrence of each duplicate.
        """

        if not documents:
            return []

        unique_documents: dict[str, RetrievedDocument] = {}

        for document in documents:
            key = self._create_key(document)

            existing = unique_documents.get(key)

            # First occurrence
            if existing is None:
                unique_documents[key] = document
                continue

            # Duplicate found:
            # keep the document with the better distance.
            if document.distance < existing.distance:
                unique_documents[key] = document

        deduplicated_documents = list(unique_documents.values())

        duplicates_removed = (
            len(documents) - len(deduplicated_documents)
        )

        logger.info("=" * 60)
        logger.info("DOCUMENT DEDUPLICATION")
        logger.info("=" * 60)
        logger.info(
            f"Input Documents    : {len(documents)}"
        )
        logger.info(
            f"Unique Documents   : {len(deduplicated_documents)}"
        )
        logger.info(
            f"Duplicates Removed : {duplicates_removed}"
        )
        logger.info("=" * 60)

        return deduplicated_documents

    @staticmethod
    def _create_key(
        document: RetrievedDocument,
    ) -> str:
        """
        Create a stable identity key for a retrieved document.

        Identity is source-specific because every ARIA knowledge
        source has a different identifier structure.
        """

        metadata: dict[str, Any] = document.metadata

        source = str(
            metadata.get("source", "UNKNOWN")
        )

        # --------------------------------------------------
        # MITRE ATT&CK
        # --------------------------------------------------
        if source == "MITRE_ATTACK":
            mitre_id = metadata.get("mitre_id")

            if mitre_id is not None:
                return f"{source}:mitre:{mitre_id}"

        # --------------------------------------------------
        # SIGMA RULES
        # --------------------------------------------------
        if source == "SIGMA_RULES":
            sigma_id = metadata.get("sigma_id")

            if sigma_id is not None:
                return f"{source}:sigma:{sigma_id}"

        # --------------------------------------------------
        # IR PLAYBOOKS
        # --------------------------------------------------
        if source == "IR_PLAYBOOKS":
            playbook = metadata.get("playbook")
            chunk_index = metadata.get("chunk_index")

            if playbook is not None and chunk_index is not None:
                return (
                    f"{source}:playbook:"
                    f"{playbook}:{chunk_index}"
                )

        # --------------------------------------------------
        # Generic fallback
        # --------------------------------------------------
        normalized_text = re.sub(
            r"\s+",
            " ",
            document.document.strip(),
        ).casefold()

        return f"{source}:text:{normalized_text}"