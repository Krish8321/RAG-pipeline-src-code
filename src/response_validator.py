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
# CLI flags, etc). Deliberately broad -- false positives here just
# mean an extra warning line, which is a cheap cost compared to an
# unflagged fabrication reaching an analyst.
ARTIFACT_PATTERN = re.compile(
    r"""
    \b[\w\-]+\.(exe|dll|ps1|bat|sh|py)\b   # filenames
    | (?<!\w)-{1,2}[a-zA-Z][\w-]*          # CLI flags like -u, --password
    | \b\d{1,3}(?:\.\d{1,3}){3}\b          # IPv4-looking strings
    """,
    re.IGNORECASE | re.VERBOSE,
)

SECTION_PATTERN = re.compile(
    r"^(Observed Evidence|IOCs):\s*$", re.IGNORECASE | re.MULTILINE
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

    Does NOT rewrite or "fix" the response -- an automated fix could
    itself introduce errors. It only flags likely hallucinations so
    a human reviewer knows to double-check specific lines before
    trusting them. This exists because prompt instructions alone
    have repeatedly failed to prevent these exact failure modes with
    an 8B local model; a deterministic check doesn't depend on the
    model choosing to comply.

    Does NOT
    --------
    - Modify the response text.
    - Call the LLM again.
    - Replace careful prompt design -- this catches what slips
      through, it doesn't replace fixing the prompt.
    """

    def validate(
        self,
        response: str,
        alert_text: str,
        context: list[dict[str, Any]],
    ) -> ValidationResult:
        result = ValidationResult()

        self._check_ungrounded_artifacts(response, alert_text, result)
        self._check_mitre_pairs(response, result)
        self._check_fabricated_severity(response, alert_text, result)

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
        Flag technical-looking tokens (filenames, CLI flags, IPs)
        that appear in the Observed Evidence / IOCs sections but not
        in the alert text itself.
        """

        alert_lower = alert_text.lower()

        for section_match in SECTION_PATTERN.finditer(response):
            section_name = section_match.group(1)
            start = section_match.end()
            # Section runs until the next blank-line-separated header
            # or end of string. Simple heuristic: up to next line
            # starting with a capitalized word followed by ":".
            end_match = re.search(
                r"\n[A-Z][\w /]+:\s*\n", response[start:]
            )
            end = start + end_match.start() if end_match else len(response)
            section_text = response[start:end]

            for match in ARTIFACT_PATTERN.finditer(section_text):
                token = match.group(0)
                if token.lower() not in alert_lower:
                    result.warnings.append(
                        f"[{section_name}] Contains '{token}', which "
                        f"does not appear in the alert text. Verify "
                        f"this wasn't pulled from a retrieved document."
                    )

    def _check_mitre_pairs(
        self,
        response: str,
        result: ValidationResult,
    ) -> None:
        """
        Flag technique IDs whose paired name doesn't match the known
        canonical name, or IDs not recognized at all.
        """

        for match in TECHNIQUE_ID_PATTERN.finditer(response):
            technique_id = match.group(0)

            # Look at a short window after the ID for "– Name" or "- Name"
            window = response[match.end():match.end() + 80]
            name_match = re.match(r"\s*[–—-]\s*([^\n(]+)", window)
            claimed_name = name_match.group(1).strip() if name_match else None

            canonical_name = KNOWN_TECHNIQUES.get(technique_id)

            if canonical_name is None:
                result.warnings.append(
                    f"MITRE technique '{technique_id}' is not in the "
                    f"known-technique list — verify this ID actually "
                    f"appears in the retrieved evidence, not from "
                    f"model memory."
                )
            elif claimed_name and canonical_name.lower() not in claimed_name.lower():
                result.warnings.append(
                    f"MITRE technique '{technique_id}' was paired with "
                    f"'{claimed_name}', but the known name is "
                    f"'{canonical_name}' — likely a mismatched ID/name pair."
                )

    def _check_fabricated_severity(
        self,
        response: str,
        alert_text: str,
        result: ValidationResult,
    ) -> None:
        """
        Flag claims that the alert has an explicit severity when the
        alert text contains no such indicator.
        """

        severity_claim = re.search(
            r"(alert'?s?\s+severity|severity\s+(?:level\s+)?is\s+(?:explicitly\s+)?"
            r"(?:marked|labeled|stated))",
            response,
            re.IGNORECASE,
        )

        has_severity_word = re.search(
            r"\bsever(?:e|ity)\b|\b(low|medium|high|critical)\s+severity\b",
            alert_text,
            re.IGNORECASE,
        )

        if severity_claim and not has_severity_word:
            result.warnings.append(
                "Response claims the alert has an explicit severity "
                "level, but no severity indicator was found in the "
                "alert text or metadata. Likely fabricated."
            )