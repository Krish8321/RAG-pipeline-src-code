from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.config.logging_config import setup_logger

logger = setup_logger()


# ============================================================
# KNOWN MITRE TECHNIQUE ID -> NAME PAIRS
#
# Small, deliberately incomplete safety-net list covering
# techniques your knowledge base actually surfaces for
# execution/lateral-movement alerts. Extend as you see new
# techniques appear in retrieved documents. This is NOT meant
# to replace pulling IDs from retrieved metadata directly --
# if your MITRE documents carry a structured technique_id field,
# prefer wiring that through ContextBuilder instead of relying
# on this list.
# ============================================================

KNOWN_TECHNIQUES: dict[str, str] = {
    "T1021": "Remote Services",
    "T1021.002": "SMB/Windows Admin Shares",
    "T1569": "System Services",
    "T1569.002": "Service Execution",
    "T1059": "Command and Scripting Interpreter",
    "T1059.001": "PowerShell",
    "T1071": "Application Layer Protocol",
    "T1053": "Scheduled Task/Job",
    "T1047": "Windows Management Instrumentation",
    "T1003": "OS Credential Dumping",
    "T1003.006": "DCSync",
    "T1546": "Event Triggered Execution",
    "T1070": "Indicator Removal",
    "T1070.003": "Clear Command History",
}

TECHNIQUE_ID_PATTERN = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")

# Tokens that look like specific technical artifacts (filenames,
# CLI flags, IPs, Windows log-field fabrications).
ARTIFACT_PATTERN = re.compile(
    r"""
    \b[\w\-]+\.(exe|dll|ps1|bat|sh|py)\b           # filenames
    | (?<!\w)-{1,2}[a-zA-Z][\w-]*                  # CLI flags like -u, --password
    | \b\d{1,3}(?:\.\d{1,3}){3}\b                  # IPv4 addresses
    | \b(?:EventID|ShareName|SubjectUserName|TargetUserName|WorkstationName|LogonType|LogonGuid|ProcessName|ParentProcessName|CommandLine|ServiceName|RelativeTargetName|AccessMask|ObjectType|IpAddress|IpPort)\b(?:\s*[:=]\s*[\w\-\.\$\\]+)?  # Windows log / Sigma field tokens
    """,
    re.IGNORECASE | re.VERBOSE,
)

THREAT_ACTOR_MALWARE_PATTERN = re.compile(
    r"""
    \b(?:
        APT\d+
        | Cobalt\s+Strike
        | Mimikatz
        | Emotet
        | LockBit
        | Qakbot|Qbot
        | TrickBot
        | Ryuk
        | BlackCat
        | Wizard\s+Spider
        | Cozy\s+Bear
        | Fancy\s+Bear
        | Lazarus\s+Group
        | [A-Z][a-z]+\s+(?:Spider|Bear|Panda|Kitten|Tiger|Dragon|Viper|Chollima|APT\d*|Group)
    )\b
    """,
    re.VERBOSE,
)

SECTION_HEADER_PATTERN = re.compile(
    r"^(?:\#+\s*)?(Observed Evidence|IOCs|IOCs Extracted|AI Reasoning|RAG Threat Intel Context|Escalation Recommendation|Risk Score|Risk Assessment|Threat Assessment|MITRE ATT&CK Techniques|Alert-Associated MITRE Techniques|Alert Facts|Upstream Model Risk|Observed Telemetry|Evidence Interpretation|ARIA Risk Assessment|Recommended Response|Recommended Investigation|Investigation Steps|Evidence Sources):",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class ValidationResult:
    warnings: list[str] = field(default_factory=list)

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)


class ResponseValidator:
    """
    Post-generation safety net for the final SOC triage response.
    """

    def validate(
        self,
        response: str,
        alert_text: str,
        context: list[dict[str, Any]] | dict[str, Any],
    ) -> ValidationResult:
        result = ValidationResult()

        self._check_ungrounded_artifacts(response, alert_text, result)
        self._check_mitre_pairs(response, alert_text, context, result)
        self._check_threat_actors_and_malware(response, alert_text, context, result)
        self._check_fabricated_severity(response, alert_text, result)
        self._check_playbook_tracing(response, context, result)
        self._check_risk_score_format(response, result)

        if result.has_warnings:
            logger.warning("=" * 60)
            logger.warning("RESPONSE VALIDATION WARNINGS")
            logger.warning("=" * 60)
            for warning in result.warnings:
                logger.warning(warning)
            logger.warning("=" * 60)
        else:
            logger.info("Response validation: no issues flagged.")

        return result

    def _check_ungrounded_artifacts(
        self,
        response: str,
        alert_text: str,
        result: ValidationResult,
    ) -> None:
        """
        Flag technical-looking tokens (filenames, CLI flags, IPs, log fields)
        that appear in the evidence/reasoning/IOC sections but not
        in the alert text itself.
        """

        alert_lower = alert_text.lower()
        matches = list(SECTION_HEADER_PATTERN.finditer(response))

        for i, match in enumerate(matches):
            section_name = match.group(1)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(response)
            section_text = response[start:end]

            if not any(kw in section_name.lower() for kw in ("observed evidence", "iocs", "iocs extracted", "ai reasoning", "rag threat intel")):
                continue

            for art_match in ARTIFACT_PATTERN.finditer(section_text):
                token = art_match.group(0)
                if token.lower() not in alert_lower:
                    result.warnings.append(
                        f"[{section_name}] Contains '{token}', which "
                        f"does not appear in the alert text. Verify "
                        f"this wasn't pulled from a retrieved document."
                    )

    def _check_mitre_pairs(
        self,
        response: str,
        alert_text: str,
        context: list[dict[str, Any]] | dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """
        Flag technique IDs whose ID doesn't appear in either the alert text or
        retrieved documents' text/metadata for this specific run.
        """

        context_parts = [alert_text]
        if isinstance(context, dict):
            context_docs = context.get("documents", [])
        else:
            context_docs = context

        for doc in context_docs:
            if isinstance(doc, dict):
                for v in doc.values():
                    if isinstance(v, (str, int, float)):
                        context_parts.append(str(v))
            else:
                context_parts.append(str(doc))
        full_context_str = "\n".join(context_parts).lower()

        for match in TECHNIQUE_ID_PATTERN.finditer(response):
            technique_id = match.group(0)
            tech_id_lower = technique_id.lower()

            if tech_id_lower not in full_context_str:
                result.warnings.append(
                    f"MITRE technique '{technique_id}' was cited in the response "
                    f"but does not appear anywhere in the alert text or retrieved context for this run."
                )
                continue

            canonical_name = KNOWN_TECHNIQUES.get(technique_id)
            if canonical_name:
                alt_name = re.sub(r"\band\b", "&", canonical_name, flags=re.IGNORECASE)
                resp_lower = response.lower()
                if canonical_name.lower() not in resp_lower and alt_name.lower() not in resp_lower:
                    result.warnings.append(
                        f"MITRE technique '{technique_id}' is known as '{canonical_name}', "
                        f"but that name was not found anywhere in the response."
                    )

    def _check_threat_actors_and_malware(
        self,
        response: str,
        alert_text: str,
        context: list[dict[str, Any]] | dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """
        Flag named threat actors or malware tools cited in the response if they do not
        appear anywhere in the alert text or retrieved context for this run.
        """

        context_parts = [alert_text]
        if isinstance(context, dict):
            context_docs = context.get("documents", [])
        else:
            context_docs = context

        for doc in context_docs:
            if isinstance(doc, dict):
                for v in doc.values():
                    if isinstance(v, (str, int, float)):
                        context_parts.append(str(v))
            else:
                context_parts.append(str(doc))
        full_context_str = "\n".join(context_parts).lower()

        for match in THREAT_ACTOR_MALWARE_PATTERN.finditer(response):
            entity = match.group(0)
            if entity.lower() not in full_context_str:
                result.warnings.append(
                    f"Threat actor / malware '{entity}' was cited in the response "
                    f"but does not appear anywhere in the alert text or retrieved context for this run."
                )

    def _check_fabricated_severity(
        self,
        response: str,
        alert_text: str,
        result: ValidationResult,
    ) -> None:
        """
        Flag affirmative claims that the alert has an explicit severity when the
        alert text contains no such indicator. Does NOT match template-compliant
        disclaimers like "No alert severity was provided".
        """

        has_severity_in_alert = bool(
            re.search(
                r"\bsever(?:e|ity)\b|\b(low|medium|high|critical)\s+severity\b",
                alert_text,
                re.IGNORECASE,
            )
        )

        if has_severity_in_alert:
            return

        # Match severity claim phrases
        claim_pattern = re.compile(
            r"(?:alert'?s?\s+severity|severity\s+(?:level\s+)?is\s+(?:explicitly\s+)?(?:marked|labeled|stated|provided))",
            re.IGNORECASE,
        )

        for match in claim_pattern.finditer(response):
            # Check prefix window before match for negation terms
            start = max(0, match.start() - 30)
            prefix = response[start:match.start()].lower()

            # Ignore compliant negated phrasing (e.g. "No alert severity was provided", "without explicit severity")
            if any(negation in prefix for negation in ("no ", "no\n", "not ", "without ")):
                continue

            result.warnings.append(
                "Response claims the alert has an explicit severity "
                "level, but no severity indicator was found in the "
                "alert text or metadata. Likely fabricated."
            )
            break

    def _check_playbook_tracing(
        self,
        response: str,
        context: list[dict[str, Any]] | dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """
        Flag when IR_PLAYBOOKS documents were provided in context, but the response
        contains recommendations that do not trace back to any retrieved playbook.
        """
        context_docs = context.get("documents", []) if isinstance(context, dict) else context

        playbook_docs = [
            doc for doc in context_docs
            if isinstance(doc, dict) and doc.get("source") == "IR_PLAYBOOKS"
        ]

        if not playbook_docs:
            return

        playbook_text = "\n".join(
            str(doc.get("document", "")) for doc in playbook_docs
        ).lower()

        matches = list(SECTION_HEADER_PATTERN.finditer(response))
        rec_section_text = ""
        for i, match in enumerate(matches):
            section_name = match.group(1).lower()
            if any(k in section_name for k in ("recommended response", "escalation recommendation", "investigation steps")):
                start = match.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(response)
                rec_section_text += response[start:end] + "\n"

        if not rec_section_text.strip():
            return

        action_keywords = [
            word for word in re.findall(r"\b[a-z]{4,}\b", rec_section_text.lower())
            if word not in ("with", "from", "that", "this", "have", "been", "were", "should", "would", "could", "recommendation", "escalation", "response", "immediately")
        ]

        overlap = any(kw in playbook_text for kw in action_keywords)

        if not overlap and action_keywords:
            result.warnings.append(
                "Response recommendation guidance was generated when IR_PLAYBOOKS documents were available in context, "
                "but recommendation details do not trace back to any retrieved playbook document."
            )

    def _check_risk_score_format(
        self,
        response: str,
        result: ValidationResult,
    ) -> None:
        """
        Validate that a numeric Risk Score (0-100) is present and formatted cleanly.
        Flexibly accepts N/100, N%, or integer N between 0 and 100.
        """
        score_match = re.search(r"Risk\s+Score:\s*(\d{1,3})(?:\s*/\s*100|\s*%)?", response, re.IGNORECASE)
        if not score_match:
            result.warnings.append(
                "Risk Score section is missing or does not contain a valid numeric score (0-100)."
            )
            return

        score_val = int(score_match.group(1))
        if not (0 <= score_val <= 100):
            result.warnings.append(
                f"Risk Score value '{score_val}' is out of range (must be 0-100)."
            )