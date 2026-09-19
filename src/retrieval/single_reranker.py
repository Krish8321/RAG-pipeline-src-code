from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.config.logging_config import setup_logger
from src.retrieval.retriever import RetrievedDocument


logger = setup_logger()


class DocumentReranker:
    """
    Re-ranks retrieved documents using a CrossEncoder.

    This version performs SINGLE-QUERY reranking.

    Responsibilities
    ----------------
    - Score query-document pairs.
    - Sort documents by relevance.
    - Return the top-K documents.

    Does NOT
    --------
    - Retrieve documents.
    - Generate queries.
    - Deduplicate documents.
    - Aggregate scores across multiple queries.
    - Filter metadata.
    - Call the LLM.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
    ) -> None:

        self.model_name = model_name

        logger.info("=" * 60)
        logger.info("INITIALIZING DOCUMENT RERANKER")
        logger.info("=" * 60)

        logger.info(
            f"Model : {self.model_name}"
        )

        self.model = CrossEncoder(
            self.model_name
        )

        # ---------------------------------------------------------
        # RERANKER TOKENIZER DIAGNOSTICS
        # ---------------------------------------------------------

        logger.info(
            f"Reranker Max Length : {self.model.max_length}"
        )

        logger.info(
            f"Tokenizer Max Length : "
            f"{self.model.tokenizer.model_max_length}"
        )

        logger.info(
            "DOCUMENT RERANKER INITIALIZED"
        )

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int = 10,
    ) -> list[RetrievedDocument]:
        """
        Re-rank retrieved documents using a single query.

        Each document is scored exactly once against
        the supplied query.

        Parameters
        ----------
        query:
            Single query used for reranking.

        documents:
            Deduplicated documents retrieved from
            the retrieval stage.

        top_k:
            Number of highest-scoring documents to return.

        Returns
        -------
        list[RetrievedDocument]
            Documents sorted by CrossEncoder relevance score.
        """

        # ---------------------------------------------------------
        # INPUT VALIDATION
        # ---------------------------------------------------------

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        if not isinstance(documents, list):
            raise TypeError(
                "documents must be a list."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if not documents:
            logger.warning(
                "No documents available for re-ranking."
            )
            return []

        top_k = min(
            top_k,
            len(documents),
        )

        # ---------------------------------------------------------
        # RERANKING INFORMATION
        # ---------------------------------------------------------

        logger.info("=" * 60)
        logger.info("DOCUMENT RE-RANKING")
        logger.info("=" * 60)

        logger.info(
            f"Query           : {query}"
        )

        logger.info(
            f"Input Documents : {len(documents)}"
        )

        logger.info(
            f"Requested Top-K : {top_k}"
        )

        # ---------------------------------------------------------
        # DOCUMENT TOKEN LENGTH DIAGNOSTICS
        # ---------------------------------------------------------

        logger.info("-" * 60)
        logger.info("DOCUMENT TOKEN LENGTHS")
        logger.info("-" * 60)

        tokenizer = self.model.tokenizer

        for document in documents:

            name = document.metadata.get(
                "name",
                "UNKNOWN",
            )

            text = document.document

            # Do NOT truncate here.
            # We are measuring the actual document size.
            tokenized = tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
            )

            token_count = len(
                tokenized["input_ids"]
            )

            logger.info(
                f"Name: {name} | "
                f"Characters: {len(text)} | "
                f"Tokens: {token_count}"
            )

        # ---------------------------------------------------------
        # CREATE QUERY-DOCUMENT PAIRS
        # ---------------------------------------------------------

        pairs = []

        for document in documents:

            name = document.metadata.get(
                "name",
                "",
            )

            rerank_text = (
                f"Document Name: {name}\n"
                f"{document.document}"
            )

            pairs.append(
                [
                    query,
                    rerank_text,
                ]
            )

        logger.info(
            f"Total Query-Document Pairs : "
            f"{len(pairs)}"
        )

        # ---------------------------------------------------------
        # CROSS-ENCODER PREDICTION
        # ---------------------------------------------------------

        logger.info("-" * 60)
        logger.info("CROSS-ENCODER PREDICTION")
        logger.info("-" * 60)

        scores = self.model.predict(
            pairs
        )

        logger.info(
            "Cross-Encoder prediction completed."
        )

        # ---------------------------------------------------------
        # ASSIGN SCORES TO DOCUMENTS
        # ---------------------------------------------------------

        logger.info("-" * 60)
        logger.info("ASSIGNING RERANKER SCORES")
        logger.info("-" * 60)

        scored_documents: list[
            RetrievedDocument
        ] = []

        for document, score in zip(
            documents,
            scores,
        ):

            document.rerank_score = float(
                score
            )

            scored_documents.append(
                document
            )

            name = document.metadata.get(
                "name",
                "UNKNOWN",
            )

            logger.debug(
                f"Name: {name} | "
                f"Score: {float(score):.4f}"
            )

        # ---------------------------------------------------------
        # RERANKER INPUT PREVIEW
        # ---------------------------------------------------------

        logger.info("-" * 60)
        logger.info("RERANKER INPUT PREVIEW")
        logger.info("-" * 60)

        for document in documents:

            name = document.metadata.get(
                "name",
                "UNKNOWN",
            )

            logger.info(
                f"Name: {name}"
            )

        # ---------------------------------------------------------
        # SORT BY RERANKER SCORE
        # ---------------------------------------------------------

        scored_documents.sort(
            key=lambda document:
                document.rerank_score
                if document.rerank_score is not None
                else float("-inf"),
            reverse=True,
        )

        # ---------------------------------------------------------
        # SELECT TOP-K
        # ---------------------------------------------------------

        reranked_documents = scored_documents[
            :top_k
        ]

        # ---------------------------------------------------------
        # LOG TOP RESULTS
        # ---------------------------------------------------------

        logger.info("-" * 60)
        logger.info("TOP RE-RANKED DOCUMENTS")
        logger.info("-" * 60)

        for rank, document in enumerate(
            reranked_documents,
            start=1,
        ):

            source = document.metadata.get(
                "source",
                "UNKNOWN",
            )

            name = document.metadata.get(
                "name",
                "UNKNOWN",
            )

            logger.info(
                f"Rank {rank:02d} | "
                f"Score: {document.rerank_score:.4f} | "
                f"Source: {source} | "
                f"Name: {name}"
            )

        logger.info("=" * 60)
        logger.info(
            "DOCUMENT RE-RANKING COMPLETE"
        )
        logger.info("=" * 60)

        logger.info(
            f"Documents Before : "
            f"{len(documents)}"
        )

        logger.info(
            f"Documents After  : "
            f"{len(reranked_documents)}"
        )

        return reranked_documents