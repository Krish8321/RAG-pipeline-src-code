from __future__ import annotations

from dataclasses import dataclass

from src.config.logging_config import setup_logger

logger = setup_logger()


@dataclass(slots=True)
class RetrievalPlan:
    """
    Defines which ARIA knowledge sources should be searched
    for a user query.

    Attributes
    ----------
    sources
        Knowledge-base sources that should be searched.

    top_k_per_source
        Number of relevant chunks to retrieve from each source.
    """

    sources: list[str]
    top_k_per_source: int = 5

    def __post_init__(self) -> None:
        """
        Validate the retrieval plan.
        """

        if not self.sources:
            raise ValueError(
                "Retrieval plan must contain at least one source."
            )

        if self.top_k_per_source <= 0:
            raise ValueError(
                "top_k_per_source must be greater than 0."
            )


class RetrievalPlanner:
    """
    Creates a retrieval plan based on the intent expressed
    in the user's query.

    Responsibilities
    ----------------
    - Analyze the query wording.
    - Identify relevant knowledge sources.
    - Decide how many chunks to retrieve per source.
    - Return a structured RetrievalPlan.

    Does NOT
    --------
    - Generate embeddings.
    - Query ChromaDB.
    - Retrieve documents.
    - Build prompts.
    - Call the LLM.
    """

    def __init__(
        self,
        top_k_per_source: int = 5,
    ) -> None:

        if top_k_per_source <= 0:
            raise ValueError(
                "top_k_per_source must be greater than 0."
            )

        self.top_k_per_source = top_k_per_source

        logger.info("=" * 50)
        logger.info("RETRIEVAL PLANNER INITIALIZED")
        logger.info("=" * 50)
        logger.info(
            f"Top-K Per Source : {self.top_k_per_source}"
        )
        logger.info("=" * 50)

    def create_plan(
        self,
        query: str,
    ) -> RetrievalPlan:
        """
        Analyze a user query and create a retrieval plan.
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

        query_lower = query.lower().strip()

        logger.info("=" * 50)
        logger.info("CREATING RETRIEVAL PLAN")
        logger.info("=" * 50)
        logger.info(f"Query : {query}")

        sources: set[str] = set()

        # --------------------------------------------------
        # RESPONSE / MITIGATION
        # --------------------------------------------------

        response_keywords = [
            "how should i handle",
            "how should i respond",
            "how do i handle",
            "how do i respond",
            "how can i handle",
            "how can i respond",
            "how can i mitigate",
            "how to mitigate",
            "mitigate",
            "mitigation",
            "contain",
            "containment",
            "respond",
            "response",
            "recover",
            "recovery",
            "remediation",
            "remediate",
            "what should i do",
            "what should the analyst do",
            "analyst action",
            "recommended action",
            "next steps",
        ]

        if self._contains_keyword(
            query_lower,
            response_keywords,
        ):
            sources.update(
                {
                    "IR_PLAYBOOKS",
                    "MITRE_ATTACK",
                    "SIGMA_RULES",
                }
            )

            logger.debug(
                "Response/Mitigation intent detected."
            )

        # --------------------------------------------------
        # DETECTION
        # --------------------------------------------------

        detection_keywords = [
            "detect",
            "detection",
            "how can i detect",
            "how to detect",
            "identify malicious activity",
            "identify attack",
            "identify an attack",
            "monitor",
            "monitoring",
            "alert",
            "alerts",
            "sigma",
            "detection rule",
            "detection rules",
            "indicator",
            "indicators",
            "ioc",
            "iocs",
        ]

        if self._contains_keyword(
            query_lower,
            detection_keywords,
        ):
            sources.update(
                {
                    "SIGMA_RULES",
                    "MITRE_ATTACK",
                }
            )

            logger.debug(
                "Detection intent detected."
            )

        # --------------------------------------------------
        # MITRE / ATT&CK TECHNIQUE
        # --------------------------------------------------

        technique_keywords = [
            "mitre",
            "mitre attack",
            "technique",
            "techniques",
            "tactic",
            "tactics",
            "ttp",
            "ttps",
            "attacker technique",
            "attacker techniques",
            "attacker behavior",
            "attacker behavior",
            "attack behavior",
        ]

        if self._contains_keyword(
            query_lower,
            technique_keywords,
        ):
            sources.add("MITRE_ATTACK")

            logger.debug(
                "MITRE/Technique intent detected."
            )
            
        # --------------------------------------------------
        # ATTACK BEHAVIOR / SECURITY DOMAIN
        # --------------------------------------------------

        attack_behavior_keywords = [
            "lateral movement",
            "privilege escalation",
            "persistence",
            "credential access",
            "execution",
            "defense evasion",
            "discovery",
            "command and control",
            "command-and-control",
            "exfiltration",
            "initial access",
            "credential dumping",
            "remote access",
            "remote service",
        ]

        if self._contains_keyword(
            query_lower,
            attack_behavior_keywords,
        ):
            sources.update(
                {
                    "MITRE_ATTACK",
                    "SIGMA_RULES",
                    "IR_PLAYBOOKS",
                }
            )

            logger.debug(
                "Attack behavior/security domain detected."
            )

        # --------------------------------------------------
        # POWERSHELL / SCRIPTING
        # --------------------------------------------------

        powershell_keywords = [
            "powershell",
            "power shell",
            "pwsh",
            "powershell script",
            "powershell command",
            "powershell activity",
        ]

        if self._contains_keyword(
            query_lower,
            powershell_keywords,
        ):
            sources.update(
                {
                    "MITRE_ATTACK",
                    "SIGMA_RULES",
                    "IR_PLAYBOOKS",
                }
            )

            logger.debug(
                "PowerShell activity detected."
            )


        # --------------------------------------------------
        # VULNERABILITY / NVD
        # --------------------------------------------------

        vulnerability_keywords = [
            "cve",
            "vulnerability",
            "vulnerabilities",
            "severity",
            "cvss",
            "affected product",
            "affected products",
            "exploitability",
            "exploitable",
            "vulnerable",
            "vulnerability score",
            "base score",
        ]

        if self._contains_keyword(
            query_lower,
            vulnerability_keywords,
        ):
            sources.add("NVD_CVE")
            sources.add("CISA_KEV")

            logger.debug(
                "Vulnerability/NVD intent detected."
            )

        # --------------------------------------------------
        # FALLBACK
        # --------------------------------------------------

        if not sources:
            logger.info(
                "No specific retrieval intent detected."
            )

            logger.info(
                "Using all ARIA knowledge sources as fallback."
            )

            sources.update(
                {
                    "IR_PLAYBOOKS",
                    "MITRE_ATTACK",
                    "SIGMA_RULES",
                    "NVD_CVE",
                    "CISA_KEV",
                }
            )

        # --------------------------------------------------
        # CREATE PLAN
        # --------------------------------------------------

        plan = RetrievalPlan(
            sources=sorted(sources),
            top_k_per_source=self.top_k_per_source,
        )

        logger.info("-" * 50)
        logger.info("RETRIEVAL PLAN")
        logger.info("-" * 50)
        logger.info(
            f"Sources : {plan.sources}"
        )
        logger.info(
            f"Top-K Per Source : "
            f"{plan.top_k_per_source}"
        )
        logger.info("=" * 50)

        return plan

    @staticmethod
    def _contains_keyword(
        query: str,
        keywords: list[str],
    ) -> bool:
        """
        Check whether any configured keyword or phrase
        appears in the normalized query.
        """

        return any(
            keyword in query
            for keyword in keywords
        )