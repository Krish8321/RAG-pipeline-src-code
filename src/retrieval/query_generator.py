from __future__ import annotations

from src.config.logging_config import setup_logger
from src.llm.ollama_client import OllamaClient

logger = setup_logger()


class QueryGenerator:
    """
    Generates multiple search-oriented queries from an original query.

    Responsibilities
    ----------------
    - Accept an original security query.
    - Generate multiple semantically diverse search queries using the LLM.
    - Parse and validate the generated queries.

    Does NOT
    --------
    - Generate embeddings.
    - Search the vector database.
    - Retrieve documents.
    - Filter metadata.
    - Re-rank documents.
    - Build the final LLM prompt.
    """

    def __init__(
        self,
        llm_client: OllamaClient,
        num_queries: int = 4,
    ) -> None:

        if num_queries < 1:
            raise ValueError(
                "num_queries must be at least 1."
            )

        self.llm_client = llm_client
        self.num_queries = num_queries

        logger.info("=" * 50)
        logger.info("QUERY GENERATOR INITIALIZED")
        logger.info("=" * 50)
        logger.info(
            f"Number of Queries : {self.num_queries}"
        )

    def generate_queries(
        self,
        query: str,
    ) -> list[str]:
        """
        Generate multiple search queries from the original query.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        logger.info("=" * 50)
        logger.info("MULTI-QUERY GENERATION")
        logger.info("=" * 50)

        logger.info(
            f"Original Query : {query}"
        )

        prompt = self._build_prompt(query)

        response = self.llm_client.generate(prompt)

        queries = self._parse_response(response)

        logger.info(
            f"Generated Queries : {len(queries)}"
        )

        for index, generated_query in enumerate(
            queries,
            start=1,
        ):
            logger.info(
                f"Query {index} : {generated_query}"
            )

        return queries

    def _build_prompt(
        self,
        query: str,
    ) -> str:
        """
        Build the prompt used for query generation.
        """

        return f"""
You are a cybersecurity query generator for a RAG retrieval system.

Generate exactly {self.num_queries} diverse search queries based ONLY on
what is explicitly stated in the alert below. The alert is your only
source of truth — do not add entities, tools, techniques, IOCs, or
attack stages that are not mentioned in it (e.g. if it says
"PowerShell execution", do not assume credential theft, persistence,
or C2 unless those words also appear).

Vary queries across these angles, using only alert-supported terms,
skipping any angle that would require inventing detail:
- direct behavior described in the alert
- technical mechanism of that behavior
- detection/investigation of that exact behavior
- mitigation/response for that exact behavior

Rules:
- Every entity in every query must trace back to a word or clear
  paraphrase in the alert.
- Rephrase and expand — never introduce new facts.
- Do not answer the alert or explain anything.
- Do not number the queries.
- Return exactly {self.num_queries} queries, one per line, nothing else.

Alert:
{query}
""".strip()

    def _parse_response(
        self,
        response: str,
    ) -> list[str]:
        """
        Parse the LLM response into individual queries.
        """

        if not response or not response.strip():
            raise ValueError(
                "LLM returned an empty response."
            )

        queries = []

        for line in response.splitlines():

            query = line.strip()

            if not query:
                continue

            # Remove accidental numbering.
            if query[0].isdigit() and "." in query:
                query = query.split(".", 1)[1].strip()

            if query:
                queries.append(query)

        # Remove duplicate queries while preserving order.
        queries = list(dict.fromkeys(queries))

        if not queries:
            raise ValueError(
                "No valid queries were generated."
            )

        return queries