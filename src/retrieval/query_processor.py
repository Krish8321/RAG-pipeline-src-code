from dataclasses import dataclass

import numpy as np

from src.config.logging_config import setup_logger
from src.embeddings.embedder import DocumentEmbedder

logger = setup_logger()


@dataclass(slots=True)
class ProcessedQuery:
    """
    Represents a processed user query ready for retrieval.
    """

    original_query: str
    cleaned_query: str
    embedding: np.ndarray


class QueryProcessor:
    """
    Processes user queries before semantic retrieval.

    Responsibilities
    ----------------
    - Validate the incoming query
    - Normalize the query
    - Generate its embedding
    - Return a structured ProcessedQuery object

    Does NOT
    --------
    - Retrieve documents
    - Talk to ChromaDB
    - Call the LLM
    """

    def __init__(self, embedder: DocumentEmbedder) -> None:
        self.embedder = embedder

    def process(self, query: str) -> ProcessedQuery:
        """
        Complete query processing pipeline.
        """

        logger.info("=" * 50)
        logger.info("PROCESSING USER QUERY")
        logger.info("=" * 50)

        validated_query = self._validate_query(query)
        cleaned_query = self._normalize_query(validated_query)
        embedding = self._generate_embedding(cleaned_query)

        logger.info("Query processed successfully.")
        logger.info("=" * 50)

        return ProcessedQuery(
            original_query=query,
            cleaned_query=cleaned_query,
            embedding=embedding,
        )

    def _validate_query(self, query: str) -> str:
        """
        Validate the input query.
        """

        if query is None:
            raise ValueError("Query cannot be None.")

        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        return query

    def _normalize_query(self, query: str) -> str:
        """
        Normalize whitespace.

        Example
        -------
        Input:
            '   How     do   I    stop ransomware?   '

        Output:
            'How do I stop ransomware?'
        """

        cleaned = " ".join(query.split())

        logger.debug(f"Normalized Query : {cleaned}")

        return cleaned

    def _generate_embedding(self, query: str) -> np.ndarray:
        """
        Generate a dense embedding for the query.
        """

        embedding = self.embedder.embed_query(query)
        
        if not isinstance(embedding, np.ndarray):
            raise TypeError(
                "Query embedding must be a NumPy array."
            )

        if embedding.ndim != 1:
            raise ValueError(
                f"Query embedding must be 1-dimensional, "
                f"got shape {embedding.shape}."
            )

        if embedding.size == 0:
            raise ValueError(
                "Query embedding cannot be empty."
            )

        logger.debug(
            f"Generated query embedding with shape {embedding.shape}"
        )

        return embedding