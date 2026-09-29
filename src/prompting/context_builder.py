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
    (re.compile(r"authentication|logon|login|oauth|iam", re.I),
     "authentication or identity-related activity"),
    (re.compile(r"lateral movement", re.I),
     "lateral movement"),
    (re.compile(r"\bsqli\b|sql injection|\bxss\b|\bssrf\b|\brce\b|web attack|\bweb exploit", re.I),
     "web application exploitation or vulnerability abuse"),
    (re.compile(r"aws|azure|gcp|cloudtrail|cloud", re.I),
     "cloud environment activity"),
    (re.compile(r"kubernetes|k8s|docker|container", re.I),
     "container environment activity"),
    (re.compile(r"phishing|malware", re.I),
     "malware or phishing activity"),
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
    """

    def __init__(
        self,
        primary_threshold: float = 0.6,
        supporting_threshold: float = 0.3,
        primary_relative_fraction: float = 0.75,
        supporting_relative_fraction: float = 0.40,
        noise_floor: float = 0.05,
        min_documents: int = 3,
        min_primary_for_exclusion: int = 3,
        reorder_for_recency: bool = False,
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

        min_documents
            If filtering by noise_floor leaves fewer than this many
            documents, backfill with the next-best dropped documents
            (by score).

        min_primary_for_exclusion
            If there are at least this many primary tier documents,
            drop supporting and contextual tier documents from the
            prompt entirely.

        reorder_for_recency
            If True, order the final context so the highest-scoring
            document appears LAST.
        """

        if not 0 <= noise_floor <= supporting_threshold <= primary_threshold <= 1:
            raise ValueError(
                "Thresholds must satisfy "
                "0 <= noise_floor <= supporting_threshold "
                "<= primary_threshold <= 1."
            )

        if min_documents < 0:
            raise ValueError("min_documents cannot be negative.")

        if min_primary_for_exclusion < 0:
            raise ValueError("min_primary_for_exclusion cannot be negative.")

        self.primary_threshold = primary_threshold
        self.supporting_threshold = supporting_threshold
        self.primary_relative_fraction = primary_relative_fraction
        self.supporting_relative_fraction = supporting_relative_fraction
        self.noise_floor = noise_floor
        self.min_documents = min_documents
        self.min_primary_for_exclusion = min_primary_for_exclusion
        self.reorder_for_recency = reorder_for_recency

        logger.info("=" * 60)
        logger.info("CONTEXT BUILDER INITIALIZED")
        logger.info(f"Primary Tier Threshold          : {self.primary_threshold}")
        logger.info(f"Supporting Tier Threshold       : {self.supporting_threshold}")
        logger.info(f"Primary Relative Fraction       : {self.primary_relative_fraction}")
        logger.info(f"Supporting Relative Fraction    : {self.supporting_relative_fraction}")
        logger.info(f"Noise Floor                     : {self.noise_floor}")
        logger.info(f"Min Documents                   : {self.min_documents}")
        logger.info(f"Min Primary For Exclusion       : {self.min_primary_for_exclusion}")
        logger.info(f"Reorder For Recency             : {self.reorder_for_recency}")
        logger.info("=" * 60)

    def _relevance_tier(
        self,
        rerank_score: float | None,
        top_score: float | None = None,
    ) -> str:
        if rerank_score is None:
            return "unscored"

        # Absolute threshold check
        is_primary = rerank_score >= self.primary_threshold
        is_supporting = rerank_score >= self.supporting_threshold

        # Relative threshold check against batch top score
        if top_score is not None and top_score > 0:
            if rerank_score >= top_score * self.primary_relative_fraction:
                is_primary = True
            elif rerank_score >= top_score * self.supporting_relative_fraction:
                is_supporting = True

        if is_primary:
            return "primary"
        if is_supporting:
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

    def _extract_technique_consensus(
        self,
        primary_docs: list[dict[str, Any]],
    ) -> str | None:
        if not primary_docs:
            return None

        pattern = re.compile(r"(?:attack\.t|\bT)(\d{4}(?:\.\d{3})?)\b", re.IGNORECASE)
        tech_counts: dict[str, int] = {}

        for doc in primary_docs:
            doc_text = doc.get("document", "")
            doc_name = doc.get("name", "")
            combined_text = f"{doc_name}\n{doc_text}"

            found_in_doc = set()
            for match in pattern.finditer(combined_text):
                tech_id = f"T{match.group(1).upper()}"
                found_in_doc.add(tech_id)

            for tech_id in found_in_doc:
                tech_counts[tech_id] = tech_counts.get(tech_id, 0) + 1

        if not tech_counts:
            return None

        top_tech, max_count = max(tech_counts.items(), key=lambda item: item[1])
        total_primary = len(primary_docs)

        if max_count / total_primary >= 0.5:
            consensus_str = f"{top_tech} (cited in {max_count}/{total_primary} primary documents)"
            logger.info(f"Technique Consensus Extracted: {consensus_str}")
            return consensus_str

        return None

    def build(
        self,
        documents: list[RetrievedDocument],
    ) -> dict[str, Any]:
        """
        Build structured context from the final reranked documents.

        Parameters
        ----------
        documents
            Final documents returned by the reranker.

        Returns
        -------
        dict[str, Any]
            Structured evidence context containing:
            - `documents`: list of structured evidence dicts
            - `technique_consensus`: optional MITRE consensus string
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

        top_score = max(
            (
                getattr(doc, "rerank_score", None)
                for doc in documents
                if getattr(doc, "rerank_score", None) is not None
            ),
            default=None,
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

            relevance_tier = self._relevance_tier(rerank_score, top_score=top_score)
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
            if item.get("source") == "IR_PLAYBOOKS":
                return score is not None and score >= 0.0001
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

        final_context = kept

        # ---------------------------------------------------------
        # PRIMARY TIER EXCLUSION
        #
        # If there are enough PRIMARY tier documents (>= min_primary_for_exclusion),
        # drop SUPPORTING and CONTEXTUAL tier documents entirely to prevent
        # 8B local models from anchoring on lower-tier evidence.
        # ---------------------------------------------------------
        primary_docs = [doc for doc in final_context if doc.get("relevance_tier") == "primary"]
        if self.min_primary_for_exclusion > 0 and len(primary_docs) >= self.min_primary_for_exclusion:
            logger.info(
                f"Primary Exclusion Filter : {len(primary_docs)} primary docs >= threshold "
                f"{self.min_primary_for_exclusion}. Dropping {len(final_context) - len(primary_docs)} "
                f"supporting/contextual docs from prompt context."
            )
            final_context = primary_docs

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

        technique_consensus = self._extract_technique_consensus(primary_docs)

        computed_risk_score, risk_factors, escalation_recommendation = self._compute_risk_and_escalation(
            documents=final_context,
            technique_consensus=technique_consensus,
        )

        logger.info(f"Computed Deterministic Risk Score : {computed_risk_score}/100")
        logger.info(f"Escalation Recommendation         : {escalation_recommendation}")

        logger.info("=" * 60)
        logger.info("RAG CONTEXT BUILT")
        logger.info("=" * 60)

        return {
            "documents": final_context,
            "technique_consensus": technique_consensus,
            "computed_risk_score": computed_risk_score,
            "risk_factors": risk_factors,
            "escalation_recommendation": escalation_recommendation,
        }

    def _compute_risk_score(
        self,
        documents: list[dict[str, Any]],
        technique_consensus: str | None,
    ) -> int:
        score, _, _ = self._compute_risk_and_escalation(documents, technique_consensus)
        return score

    def _compute_risk_and_escalation(
        self,
        documents: list[dict[str, Any]],
        technique_consensus: str | None,
    ) -> tuple[int, list[str], str]:
        """
        Compute a transparent, deterministic 0-100 risk score, explicit risk factors,
        and escalation recommendation based on evidence severity, primary coverage,
        technique consensus, and alert signals.
        """
        severity_map = {
            "critical": 100,
            "high": 75,
            "medium": 50,
            "low": 25,
        }
        base_score = 50
        risk_factors = []

        found_severities = []
        for doc in documents:
            text = f"{doc.get('name', '')}\n{doc.get('document', '')}".lower()
            for sev_name, score in severity_map.items():
                if f"severity level: {sev_name}" in text or f"severity: {sev_name}" in text:
                    found_severities.append((score, sev_name.upper(), doc.get("name", "Unknown Rule")))

        if found_severities:
            max_sev_score, max_sev_name, rule_name = max(found_severities, key=lambda x: x[0])
            base_score = max_sev_score
            risk_factors.append(f"Retrieved detection rule severity: {max_sev_name} ({rule_name})")
        else:
            risk_factors.append("No explicit detection severity in evidence (default base score 50)")

        score = base_score

        if technique_consensus:
            score += 15
            risk_factors.append(f"MITRE ATT&CK technique consensus confirmed ({technique_consensus})")

        primary_count = sum(1 for d in documents if d.get("relevance_tier") == "primary")
        if primary_count > 1:
            boost = min(10, (primary_count - 1) * 5)
            score += boost
            risk_factors.append(f"Multiple primary evidence rules matched ({primary_count} rules)")

        for doc in documents:
            text = doc.get("document", "").lower()
            if "exfiltration" in text or "data leak" in text or "upload" in text:
                score += 10
                risk_factors.append("Retrieved evidence indicates potential data exfiltration signal")
                break

        if primary_count == 0:
            score -= 10
            risk_factors.append("No primary tier direct matches in retrieval")

        final_score = min(100, max(10, score))

        if final_score >= 90:
            escalation = "Escalate to Tier-2 / Analyst Investigation Required"
        elif final_score >= 70:
            escalation = "Escalate to Tier-2 / Review Required"
        elif final_score >= 40:
            escalation = "Tier-1 Investigation / Monitor Host"
        else:
            escalation = "Monitor — Low Risk / No Action Required"

        return final_score, risk_factors, escalation