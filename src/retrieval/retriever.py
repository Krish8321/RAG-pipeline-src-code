from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config.logging_config import setup_logger
from src.embeddings.embedder import DocumentEmbedder
from src.retrieval.query_processor import ProcessedQuery
from src.retrieval.retrieval_planner import RetrievalPlan
from src.vector_store.chroma_store import ChromaStore

logger = setup_logger()


@dataclass(slots=True)
class RetrievedDocument:
    """
    Represents a document retrieved from the ARIA knowledge base.
    """

    document: str
    distance: float
    metadata: dict[str, Any]
    query: str
    rerank_score: float | None = None


class Retriever:
    """
    Retrieves relevant documents from the ARIA knowledge base.

    Supports both:
    - Traditional single-query retrieval.
    - Multi-query retrieval.

    Responsibilities
    ----------------
    - Accept retrieval queries.
    - Embed queries.
    - Search selected knowledge sources independently.
    - Retrieve top-K documents from each source.
    - Convert raw ChromaDB results into structured objects.
    - Build a candidate document pool.

    Does NOT
    --------
    - Generate queries.
    - Decide which sources should be searched.
    - Filter metadata.
    - Re-rank documents.
    - Deduplicate documents.
    - Build prompts.
    - Call the LLM.
    """

    def __init__(
        self,
        vector_store: ChromaStore,
        embedder: DocumentEmbedder,
    ) -> None:

        self.vector_store = vector_store
        self.embedder = embedder

        logger.info("=" * 50)
        logger.info("RETRIEVER INITIALIZED")
        logger.info("=" * 50)

    # ======================================================
    # SINGLE QUERY RETRIEVAL
    # ======================================================

    def retrieve(
        self,
        processed_query: ProcessedQuery,
        retrieval_plan: RetrievalPlan,
    ) -> list[RetrievedDocument]:
        """
        Retrieve documents using a single processed query.

        This method is kept for comparison with the
        multi-query retrieval approach.
        """

        logger.info("=" * 50)
        logger.info("SINGLE-QUERY DOCUMENT RETRIEVAL")
        logger.info("=" * 50)

        if not isinstance(processed_query, ProcessedQuery):
            raise TypeError(
                "Expected a ProcessedQuery object."
            )

        if not isinstance(retrieval_plan, RetrievalPlan):
            raise TypeError(
                "Expected a RetrievalPlan object."
            )

        logger.info(
            f"Query : {processed_query.cleaned_query}"
        )

        logger.info(
            f"Sources : {retrieval_plan.sources}"
        )

        logger.info(
            f"Top-K Per Source : "
            f"{retrieval_plan.top_k_per_source}"
        )

        retrieved_documents: list[RetrievedDocument] = []

        for source in retrieval_plan.sources:

            logger.info("-" * 50)
            logger.info(
                f"RETRIEVING FROM SOURCE : {source}"
            )
            logger.info("-" * 50)

            results = self.vector_store.similarity_search(
                query_embedding=processed_query.embedding,
                n_results=retrieval_plan.top_k_per_source,
                source=source,
            )

            source_documents = self._parse_results(
                results,
                query=processed_query.cleaned_query,
            )

            logger.info(
                f"Retrieved from {source} : "
                f"{len(source_documents)}"
            )

            retrieved_documents.extend(
                source_documents
            )

        logger.info("=" * 50)
        logger.info("SINGLE-QUERY RETRIEVAL COMPLETE")
        logger.info("=" * 50)

        logger.info(
            f"Total Documents Retrieved : "
            f"{len(retrieved_documents)}"
        )

        return retrieved_documents

    # ======================================================
    # MULTI QUERY RETRIEVAL
    # ======================================================

    def retrieve_multi_query(
        self,
        queries: list[str],
        retrieval_plan: RetrievalPlan,
    ) -> list[RetrievedDocument]:
        """
        Retrieve documents using multiple semantic queries.

        Each query is embedded independently and searched
        against every source selected by the RetrievalPlan.

        The returned documents form a broad candidate pool.

        No:
        - metadata filtering
        - re-ranking
        - deduplication

        is performed here.
        """

        logger.info("=" * 50)
        logger.info("MULTI-QUERY DOCUMENT RETRIEVAL")
        logger.info("=" * 50)

        # --------------------------------------------------
        # VALIDATE QUERIES
        # --------------------------------------------------

        if not isinstance(queries, list):
            raise TypeError(
                "queries must be a list of strings."
            )

        if not queries:
            raise ValueError(
                "queries cannot be empty."
            )

        for query in queries:

            if not isinstance(query, str):
                raise TypeError(
                    "Every query must be a string."
                )

            if not query.strip():
                raise ValueError(
                    "Queries cannot contain empty strings."
                )

        # --------------------------------------------------
        # VALIDATE RETRIEVAL PLAN
        # --------------------------------------------------

        if not isinstance(retrieval_plan, RetrievalPlan):
            raise TypeError(
                "Expected a RetrievalPlan object."
            )

        logger.info(
            f"Number of Queries : {len(queries)}"
        )

        logger.info(
            f"Sources : {retrieval_plan.sources}"
        )

        logger.info(
            f"Top-K Per Source : "
            f"{retrieval_plan.top_k_per_source}"
        )

        # --------------------------------------------------
        # CANDIDATE POOL
        # --------------------------------------------------

        retrieved_documents: list[RetrievedDocument] = []

        # --------------------------------------------------
        # PROCESS EACH QUERY
        # --------------------------------------------------

        for query_index, query in enumerate(
            queries,
            start=1,
        ):

            query = query.strip()

            logger.info("-" * 50)
            logger.info(
                f"PROCESSING QUERY {query_index}"
            )
            logger.info("-" * 50)

            logger.info(
                f"Query : {query}"
            )

            # --------------------------------------------------
            # EMBED QUERY
            # --------------------------------------------------

            query_embedding = self.embedder.embed_query(
                query
            )

            logger.debug(
                f"Query embedding generated | "
                f"Dimension : {query_embedding.shape[0]}"
            )

            # --------------------------------------------------
            # SEARCH EACH SOURCE
            # --------------------------------------------------

            for source in retrieval_plan.sources:

                logger.info(
                    f"Searching source : {source}"
                )

                results = self.vector_store.similarity_search(
                    query_embedding=query_embedding,
                    n_results=retrieval_plan.top_k_per_source,
                    source=source,
                )

                source_documents = self._parse_results(
                    results,
                    query=query,
                )

                logger.info(
                    f"Retrieved from {source} : "
                    f"{len(source_documents)}"
                )

                # --------------------------------------------------
                # ADD TO CANDIDATE POOL
                # --------------------------------------------------

                retrieved_documents.extend(
                    source_documents
                )

        # --------------------------------------------------
        # RETRIEVAL SUMMARY
        # --------------------------------------------------

        logger.info("=" * 50)
        logger.info("MULTI-QUERY RETRIEVAL COMPLETE")
        logger.info("=" * 50)

        logger.info(
            f"Queries Processed : {len(queries)}"
        )

        logger.info(
            f"Sources Searched : "
            f"{len(retrieval_plan.sources)}"
        )

        logger.info(
            f"Total Candidate Documents : "
            f"{len(retrieved_documents)}"
        )

        logger.info(
            "No deduplication or re-ranking performed."
        )

        return retrieved_documents

    # ======================================================
    # PARSE CHROMADB RESULTS
    # ======================================================

    def _parse_results(
        self,
        results: dict[str, Any],
        query: str,
    ) -> list[RetrievedDocument]:
        """
        Convert raw ChromaDB results into structured
        RetrievedDocument objects.
        """

        documents = results.get("documents")
        distances = results.get("distances")
        metadatas = results.get("metadatas")

        if not documents:
            logger.warning(
                "No documents retrieved from source."
            )
            return []

        documents = documents[0]

        if distances:
            distances = distances[0]
        else:
            distances = [0.0] * len(documents)

        if metadatas:
            metadatas = metadatas[0]
        else:
            metadatas = [{} for _ in documents]

        retrieved_documents: list[RetrievedDocument] = []

        for document, distance, metadata in zip(
            documents,
            distances,
            metadatas,
        ):

            retrieved_documents.append(
                RetrievedDocument(
                    document=document,
                    distance=float(distance),
                    metadata=metadata or {},
                    query=query,
                )
            )

        return retrieved_documents