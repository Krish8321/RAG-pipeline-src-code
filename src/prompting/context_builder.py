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
        min_top_for_relative_primary = self.primary_threshold * self.primary_relative_fraction
        min_top_for_relative_supporting = self.supporting_threshold * self.supporting_relative_fraction

        if top_score is not None and top_score > 0:
            if top_score >= min_top_for_relative_primary and rerank_score >= top_score * self.primary_relative_fraction:
                is_primary = True
            elif top_score >= min_top_for_relative_supporting and rerank_score >= top_score * self.supporting_relative_fraction:
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

    def _extract_technique_support(
        self,
        primary_docs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Extract MITRE ATT&CK technique support across primary tier documents.

        Classifies technique support as:
        - UNANIMOUS (100% of primary docs) -> +15 boost
        - MAJORITY (>50% of primary docs)  -> +15 boost
        - PARTIAL  (=50% of primary docs)  -> +5 boost
        - MINORITY (<50% of primary docs)  -> +0 boost
        - NONE     (0 primary docs or no technique IDs found) -> +0 boost
        """
        if not primary_docs:
            return {
                "status": "NONE",
                "top_technique": None,
                "counts": {},
                "fraction": 0.0,
                "total_primary": 0,
                "boost": 0,
                "description": "No primary evidence documents available",
                "techniques_by_doc": [],
            }

        pattern = re.compile(r"(?:attack\.t|\bT)(\d{4}(?:\.\d{3})?)\b", re.IGNORECASE)
        tech_counts: dict[str, int] = {}
        techniques_by_doc: list[set[str]] = []

        for doc in primary_docs:
            doc_text = doc.get("document", "")
            doc_name = doc.get("name", "")
            combined_text = f"{doc_name}\n{doc_text}"

            found_in_doc = set()
            for match in pattern.finditer(combined_text):
                tech_id = f"T{match.group(1).upper()}"
                found_in_doc.add(tech_id)

            techniques_by_doc.append(found_in_doc)
            for tech_id in found_in_doc:
                tech_counts[tech_id] = tech_counts.get(tech_id, 0) + 1

        total_primary = len(primary_docs)
        if not tech_counts:
            return {
                "status": "NONE",
                "top_technique": None,
                "counts": {},
                "fraction": 0.0,
                "total_primary": total_primary,
                "boost": 0,
                "description": "No explicit MITRE technique IDs found in primary evidence",
                "techniques_by_doc": techniques_by_doc,
            }

        top_tech, max_count = max(tech_counts.items(), key=lambda item: item[1])
        fraction = max_count / total_primary

        if fraction == 1.0:
            status = "UNANIMOUS"
            boost = 15
        elif fraction > 0.5:
            status = "MAJORITY"
            boost = 15
        elif fraction == 0.5:
            status = "PARTIAL"
            boost = 5
        else:
            status = "MINORITY"
            boost = 0

        description = f"{top_tech} ({status} support: cited in {max_count}/{total_primary} primary documents)"
        logger.info(f"Technique Support Extracted: {description}")

        return {
            "status": status,
            "top_technique": top_tech,
            "counts": tech_counts,
            "fraction": fraction,
            "total_primary": total_primary,
            "boost": boost,
            "description": description,
            "techniques_by_doc": techniques_by_doc,
        }

    def _extract_technique_consensus(
        self,
        primary_docs: list[dict[str, Any]],
    ) -> str | None:
        info = self._extract_technique_support(primary_docs)
        if info["status"] in ("UNANIMOUS", "MAJORITY", "PARTIAL"):
            return info["description"]
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
            - `computed_risk_score`: deterministic risk score (0-100)
            - `risk_breakdown`: structured score breakdown
            - `risk_factors`: human-readable list of risk factors
            - `escalation_recommendation`: escalation level string
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

        def _above_noise_floor(item: dict[str, Any]) -> bool:
            score = item["rerank_score"]
            if item.get("source") == "IR_PLAYBOOKS":
                return score is not None and score >= 0.10
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

        primary_docs = [doc for doc in final_context if doc.get("relevance_tier") == "primary"]
        if self.min_primary_for_exclusion > 0 and len(primary_docs) >= self.min_primary_for_exclusion:
            logger.info(
                f"Primary Exclusion Filter : {len(primary_docs)} primary docs >= threshold "
                f"{self.min_primary_for_exclusion}. Dropping {len(final_context) - len(primary_docs)} "
                f"supporting/contextual docs from prompt context."
            )
            final_context = primary_docs

        if self.reorder_for_recency:
            final_context = list(reversed(final_context))
            logger.info(
                "Context reordered: highest-relevance document "
                "placed last (closest to generation instructions)."
            )

        technique_support_info = self._extract_technique_support(primary_docs)
        technique_consensus = (
            technique_support_info.get("description")
            if technique_support_info.get("status") in ("UNANIMOUS", "MAJORITY", "PARTIAL")
            else None
        )

        (
            computed_risk_score,
            risk_breakdown,
            risk_factors,
            escalation_recommendation,
            primary_severity_summary,
            supporting_severity_summary,
        ) = self._compute_risk_and_escalation(
            documents=final_context,
            technique_support_info=technique_support_info,
        )

        logger.info(f"Computed Deterministic Risk Score : {computed_risk_score}/100")
        logger.info(f"Escalation Recommendation         : {escalation_recommendation}")

        logger.info("=" * 60)
        logger.info("RAG CONTEXT BUILT")
        logger.info("=" * 60)

        return {
            "documents": final_context,
            "technique_consensus": technique_consensus,
            "technique_support_info": technique_support_info,
            "computed_risk_score": computed_risk_score,
            "risk_breakdown": risk_breakdown,
            "risk_factors": risk_factors,
            "primary_severity_summary": primary_severity_summary,
            "supporting_severity_summary": supporting_severity_summary,
            "escalation_recommendation": escalation_recommendation,
        }

    def _compute_risk_score(
        self,
        documents: list[dict[str, Any]],
        technique_consensus: str | None = None,
    ) -> int:
        score, _, _, _, _, _ = self._compute_risk_and_escalation(documents)
        return score

    def _compute_risk_and_escalation(
        self,
        documents: list[dict[str, Any]],
        technique_support_info: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any], list[str], str, str, str]:
        """
        Compute a transparent, deterministic 0-100 risk score, explicit risk factors,
        and escalation recommendation based on evidence severity, primary coverage,
        technique support classification, and alert signals.
        """
        if technique_support_info is None:
            primary_docs = [d for d in documents if d.get("relevance_tier") == "primary" and d.get("source") != "IR_PLAYBOOKS"]
            technique_support_info = self._extract_technique_support(primary_docs)

        severity_map = {
            "critical": 100,
            "high": 75,
            "medium": 50,
            "low": 25,
        }

        # 1. Tiered Evidence Severity
        primary_sevs = []
        supporting_sevs = []

        for doc in documents:
            if doc.get("source") == "IR_PLAYBOOKS":
                continue
            tier = doc.get("relevance_tier", "contextual")
            text = f"{doc.get('name', '')}\n{doc.get('document', '')}".lower()
            for sev_name, score in severity_map.items():
                if f"severity level: {sev_name}" in text or f"severity: {sev_name}" in text or f"severity {sev_name}" in text:
                    item = (score, sev_name.upper(), doc.get("name", "Unknown Rule"))
                    if tier == "primary":
                        primary_sevs.append(item)
                    elif tier == "supporting":
                        supporting_sevs.append(item)

        primary_max_item = max(primary_sevs, key=lambda x: x[0]) if primary_sevs else None
        supporting_max_item = max(supporting_sevs, key=lambda x: x[0]) if supporting_sevs else None

        primary_max_name = primary_max_item[1] if primary_max_item else None
        supporting_max_name = supporting_max_item[1] if supporting_max_item else None

        # Base score is strictly driven by primary tier max severity
        if primary_max_item:
            base_score = primary_max_item[0]
        else:
            base_score = 50

        # Supporting tier evidence cannot set base_score, but supporting HIGH/CRITICAL gives +5 context boost
        supporting_boost = 0
        if supporting_max_name in ("CRITICAL", "HIGH"):
            supporting_boost = 5

        # 2. Technique Support Boost
        technique_boost = technique_support_info.get("boost", 0)
        technique_status = technique_support_info.get("status", "NONE")

        # 3. Multi-Primary Evidence Boost
        primary_docs = [d for d in documents if d.get("relevance_tier") == "primary" and d.get("source") != "IR_PLAYBOOKS"]
        primary_count = len(primary_docs)
        multi_primary_boost = 0

        if primary_count > 1:
            tech_sets = technique_support_info.get("techniques_by_doc", [])
            non_empty_tech_sets = [ts for ts in tech_sets if ts]
            if len(non_empty_tech_sets) > 1 and len(set.intersection(*non_empty_tech_sets)) > 0:
                multi_primary_boost = 5  # Overlapping rules for same MITRE technique
            elif len(non_empty_tech_sets) > 1:
                multi_primary_boost = 10 # Distinct rules across different MITRE techniques
            else:
                multi_primary_boost = 5  # Default multi-primary rule boost

        # 4. Signal Modifiers
        exfil_signal = False
        for doc in documents:
            text = doc.get("document", "").lower()
            if "exfiltration" in text or "data leak" in text or "upload" in text:
                exfil_signal = True
                break

        no_primary_signal = (primary_count == 0)

        signal_modifiers = 0
        if exfil_signal:
            signal_modifiers += 10
        if no_primary_signal:
            signal_modifiers -= 10

        calculated_total = base_score + supporting_boost + technique_boost + multi_primary_boost + signal_modifiers
        final_score = min(100, max(10, calculated_total))

        risk_breakdown = {
            "base_score": base_score,
            "primary_severity": primary_max_name or "NONE",
            "supporting_severity": supporting_max_name or "NONE",
            "supporting_boost": supporting_boost,
            "technique_support_status": technique_status,
            "technique_boost": technique_boost,
            "multi_primary_boost": multi_primary_boost,
            "signal_modifiers": signal_modifiers,
            "final_score": final_score,
        }

        risk_factors = []
        if primary_max_item:
            risk_factors.append(f"Primary detection rule severity: {primary_max_name} ({primary_max_item[2]})")
        else:
            risk_factors.append("No explicit primary detection severity in evidence (default base score 50)")

        if supporting_boost > 0:
            risk_factors.append(f"Supporting evidence context boost ({supporting_max_name}): +{supporting_boost}")

        if technique_status != "NONE":
            desc = technique_support_info.get("description", "")
            risk_factors.append(f"MITRE ATT&CK technique support ({desc}): +{technique_boost}")

        if multi_primary_boost == 10:
            risk_factors.append(f"Multiple primary evidence rules across distinct MITRE techniques ({primary_count} rules): +10")
        elif multi_primary_boost == 5:
            risk_factors.append(f"Multiple primary evidence rules for same MITRE technique ({primary_count} rules): +5")

        if exfil_signal:
            risk_factors.append("Retrieved evidence indicates potential data exfiltration signal: +10")

        if no_primary_signal:
            risk_factors.append("No primary tier direct matches in retrieval: -10")

        if final_score >= 90:
            escalation = "Escalate to Tier-2 / Analyst Investigation Required"
        elif final_score >= 70:
            escalation = "Escalate to Tier-2 / Review Required"
        elif final_score >= 40:
            escalation = "Tier-1 Investigation / Monitor Host"
        else:
            escalation = "Monitor — Low Risk / No Action Required"

        return final_score, risk_breakdown, risk_factors, escalation, primary_max_name or "NONE", supporting_max_name or "NONE"