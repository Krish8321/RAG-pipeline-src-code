from __future__ import annotations

import re
from typing import Any

from src.config.logging_config import setup_logger
from src.retrieval.retriever import RetrievedDocument

logger = setup_logger()


# ============================================================
# BEHAVIOR TAGGING
#
# Rule-based, not an LLM call — deterministic, fast, and it's
# exactly this kind of "what action is this document actually
# about" judgment that an 8B model gets wrong when it has to
# infer it purely from raw document text at inference time
# (see: SMB/Admin-Shares doc outranking 7 PsExec-specific docs
# in the analyst's mind despite scoring 0.11 on rerank).
#
# Patterns are checked in order; first match wins. Order goes
# from most specific to most generic so e.g. "PsExec service
# execution" matches PsExec before the generic "service
# execution" pattern gets a chance to.
# ============================================================

BEHAVIOR_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"copy.*admin share|admin share.*copy|sysvol", re.I),
     "file copy to an admin/sysvol share"),
    (re.compile(r"psexec", re.I),
     "PsExec-based remote service execution"),
    (re.compile(r"service (execution|child process)", re.I),
     "remote or local service execution"),
    (re.compile(r"scheduled task", re.I),
     "scheduled task creation or execution"),
    (re.compile(r"\bwmi\b|windows management instrumentation", re.I),
     "WMI-based remote execution"),
    (re.compile(r"powershell", re.I),
     "PowerShell script or command execution"),
    (re.compile(r"dcsync", re.I),
     "credential replication via DCSync"),
    (re.compile(r"credential dump|lsass", re.I),
     "credential dumping"),
    (re.compile(r"ransomware", re.I),
     "ransomware-related activity"),
    (re.compile(r"exfiltrat", re.I),
     "data exfiltration"),
    (re.compile(r"persistence|\bprofile\b|startup", re.I),
     "persistence mechanism"),
    (re.compile(r"defense evasion|renamed?\b|clear.*history", re.I),
     "defense evasion technique"),
    (re.compile(r"admin share|\bsmb\b", re.I),
     "administrative share / SMB access (generic — not necessarily execution)"),
    (re.compile(r"authentication|logon|login", re.I),
     "authentication-related activity"),
    (re.compile(r"lateral movement", re.I),
     "lateral movement"),
    (re.compile(r"phishing|malware|exploit", re.I),
     "malware or exploitation activity"),
]

# How much of the document body to fall back to when the name
# alone doesn't match any pattern. Kept short — this is a tag,
# not a summary.
BEHAVIOR_FALLBACK_SCAN_CHARS = 300


class ContextBuilder:
    """
    Builds a structured evidence context from the documents
    selected by the reranker.

    Responsibilities
    ----------------
    - Accept the final reranked documents.
    - Preserve ranking and retrieval metadata.
    - Normalize document metadata.
    - Tag each document with a plain-language `matched_behavior`
      and a `relevance_tier` derived from its rerank score, so
      the final LLM has an explicit relevance signal instead of
      inferring it from keyword overlap.
    - Organize evidence for downstream prompt construction.

    Does NOT
    --------
    - Retrieve documents.
    - Generate queries.
    - Deduplicate documents.
    - Rerank documents.
    - Call the LLM.
    - Generate security conclusions.
    - Invent or infer missing security information.
    """

    def __init__(
        self,
        primary_threshold: float = 0.6,
        supporting_threshold: float = 0.3,
        noise_floor: float = 0.05,
        min_documents: int = 3,
        reorder_for_recency: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        primary_threshold
            Documents with rerank_score >= this are tagged
            "primary" — direct matches to the observed behavior.

        supporting_threshold
            Documents with rerank_score >= this (but below
            primary_threshold) are tagged "supporting". Below
            this is "contextual".

        noise_floor
            Documents scoring below this are dropped entirely
            rather than sent to the LLM as "contextual" evidence.
            A score like 0.003-0.01 (as seen in production logs)
            isn't weak context, it's noise -- the cross-encoder is
            saying it's essentially unrelated to the query. An 8B
            local model has repeatedly anchored on documents like
            this despite an explicit "contextual" tier label, so
            the more reliable fix is to not show it the noise at
            all rather than trust it to ignore a label.

        min_documents
            If filtering by noise_floor leaves fewer than this many
            documents, backfill with the next-best dropped documents
            (by score) so the LLM always has some minimum evidence
            to reason over, even for a sparse knowledge base.

        reorder_for_recency
            If True, order the final context so the highest-scoring
            document appears LAST (closest to the response
            instructions) instead of first. This trades away the
            intuitive "best evidence first" ordering to exploit
            models' well-documented recency bias in long contexts
            ("lost in the middle") -- empirically, an 8B model
            attends more strongly to content near the end of a long
            prompt. `rank` values still reflect true relevance
            regardless of list position. Treat this as a heuristic:
            A/B test with it off if your model's behavior differs.

        These defaults were chosen against your BAAI/bge-reranker-base
        score distribution (0.98 down to 0.003 in the PsExec case).
        Recalibrate if you switch reranker models -- cross-encoders
        aren't guaranteed to share a score scale.
        """

        if not 0 <= noise_floor <= supporting_threshold <= primary_threshold <= 1:
            raise ValueError(
                "Thresholds must satisfy "
                "0 <= noise_floor <= supporting_threshold "
                "<= primary_threshold <= 1."
            )

        if min_documents < 0:
            raise ValueError("min_documents cannot be negative.")

        self.primary_threshold = primary_threshold
        self.supporting_threshold = supporting_threshold
        self.noise_floor = noise_floor
        self.min_documents = min_documents
        self.reorder_for_recency = reorder_for_recency

        logger.info("=" * 60)
        logger.info("CONTEXT BUILDER INITIALIZED")
        logger.info(f"Primary Tier Threshold    : {self.primary_threshold}")
        logger.info(f"Supporting Tier Threshold : {self.supporting_threshold}")
        logger.info(f"Noise Floor               : {self.noise_floor}")
        logger.info(f"Min Documents             : {self.min_documents}")
        logger.info(f"Reorder For Recency       : {self.reorder_for_recency}")
        logger.info("=" * 60)

    def _relevance_tier(self, rerank_score: float | None) -> str:
        if rerank_score is None:
            return "unscored"
        if rerank_score >= self.primary_threshold:
            return "primary"
        if rerank_score >= self.supporting_threshold:
            return "supporting"
        return "contextual"

    def _matched_behavior(
        self,
        name: str,
        document_text: str,
        source: str,
        document_type: str,
    ) -> str:
        for pattern, label in BEHAVIOR_PATTERNS:
            if pattern.search(name):
                return label

        # Name alone didn't match — check a short window of the
        # body before falling back to a generic label. Kept short
        # on purpose so this stays a tag, not a re-summarization.
        excerpt = document_text[:BEHAVIOR_FALLBACK_SCAN_CHARS]
        for pattern, label in BEHAVIOR_PATTERNS:
            if pattern.search(excerpt):
                return label

        return f"general {document_type or source} reference material"

    def build(
        self,
        documents: list[RetrievedDocument],
    ) -> list[dict[str, Any]]:
        """
        Build structured context from the final reranked documents.

        Parameters
        ----------
        documents
            Final documents returned by the reranker.

        Returns
        -------
        list[dict[str, Any]]
            Structured evidence context. Each item includes
            `matched_behavior` and `relevance_tier` in addition to
            the existing rank/source/type/name/distance/rerank_score/
            document fields.
        """

        if documents is None:
            raise ValueError(
                "Documents cannot be None."
            )

        if not isinstance(documents, list):
            raise TypeError(
                "Documents must be a list."
            )

        if not documents:
            logger.warning(
                "No documents provided to ContextBuilder."
            )
            return []

        logger.info("=" * 60)
        logger.info("BUILDING RAG CONTEXT")
        logger.info("=" * 60)

        logger.info(
            f"Documents received : {len(documents)}"
        )

        context: list[dict[str, Any]] = []
        tier_counts = {"primary": 0, "supporting": 0, "contextual": 0, "unscored": 0}

        for rank, result in enumerate(documents, start=1):

            if not isinstance(
                result,
                RetrievedDocument,
            ):
                raise TypeError(
                    "All documents must be "
                    "RetrievedDocument instances."
                )

            metadata = result.metadata or {}

            source = metadata.get(
                "source",
                "UNKNOWN",
            )

            document_type = metadata.get(
                "type",
                "UNKNOWN",
            )

            name = metadata.get(
                "name",
                "UNKNOWN",
            )

            rerank_score = getattr(
                result,
                "rerank_score",
                None,
            )

            relevance_tier = self._relevance_tier(rerank_score)
            matched_behavior = self._matched_behavior(
                name=name,
                document_text=result.document,
                source=source,
                document_type=document_type,
            )

            tier_counts[relevance_tier] = tier_counts.get(relevance_tier, 0) + 1

            evidence: dict[str, Any] = {
                "rank": rank,
                "source": source,
                "type": document_type,
                "name": name,
                "distance": result.distance,
                "rerank_score": rerank_score,
                "relevance_tier": relevance_tier,
                "matched_behavior": matched_behavior,
                "document": result.document,
            }

            context.append(evidence)

            logger.debug(
                f"Context #{rank}: "
                f"{name} | "
                f"{source} | "
                f"Rerank Score: {rerank_score} | "
                f"Tier: {relevance_tier} | "
                f"Behavior: {matched_behavior}"
            )

        logger.info(
            f"Structured Context Items : "
            f"{len(context)}"
        )

        logger.info(
            f"Tier Distribution : "
            f"primary={tier_counts['primary']}, "
            f"supporting={tier_counts['supporting']}, "
            f"contextual={tier_counts['contextual']}, "
            f"unscored={tier_counts['unscored']}"
        )

        # ---------------------------------------------------------
        # DROP NOISE-FLOOR DOCUMENTS
        #
        # `context` is still sorted best-first (rank 1 = highest
        # score) at this point. Documents scoring below noise_floor
        # are dropped outright rather than sent to the LLM tagged
        # "contextual" -- a tier label is advice the model can
        # ignore, but a document that was never in the prompt can't
        # be hallucinated into "Observed Evidence."
        # ---------------------------------------------------------

        def _above_noise_floor(item: dict[str, Any]) -> bool:
            score = item["rerank_score"]
            return score is not None and score >= self.noise_floor

        kept = [item for item in context if _above_noise_floor(item)]
        dropped = [item for item in context if not _above_noise_floor(item)]

        if len(kept) < self.min_documents and dropped:
            backfill_needed = self.min_documents - len(kept)
            dropped.sort(
                key=lambda item: item["rerank_score"]
                if item["rerank_score"] is not None
                else float("-inf"),
                reverse=True,
            )
            kept.extend(dropped[:backfill_needed])
            kept.sort(
                key=lambda item: item["rerank_score"]
                if item["rerank_score"] is not None
                else float("-inf"),
                reverse=True,
            )

        logger.info(
            f"Noise Filter : kept {len(kept)}, "
            f"dropped {len(context) - len(kept)} "
            f"below floor {self.noise_floor}"
        )

        final_context = kept

        # ---------------------------------------------------------
        # REORDER FOR RECENCY (see reorder_for_recency docstring)
        #
        # `rank` is left untouched so it still communicates true
        # relevance order regardless of position in the list.
        # ---------------------------------------------------------

        if self.reorder_for_recency:
            final_context = list(reversed(final_context))
            logger.info(
                "Context reordered: highest-relevance document "
                "placed last (closest to generation instructions)."
            )

        logger.info("=" * 60)
        logger.info("RAG CONTEXT BUILT")
        logger.info("=" * 60)

        return final_context