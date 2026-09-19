"""
Persistent ChromaDB Vector Store for ARIA.

Responsibilities
----------------
- Initialize a persistent ChromaDB client.
- Create or load the collection.
- Store embedded documents.
- Perform similarity search.
- Retrieve collection statistics.

Does NOT
--------
- Load documents
- Generate embeddings
- Perform chunking
"""

from __future__ import annotations

from typing import Any

import chromadb
import numpy as np
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings

from src.config.logging_config import setup_logger
from src.models.document import Document
from src.vector_store.config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
)

logger = setup_logger()

# ARIA embedding model:
# BAAI/bge-base-en-v1.5
EXPECTED_EMBEDDING_DIMENSION = 768


class ChromaStore:
    """
    Wrapper around ChromaDB for persistent vector storage.
    """

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
    ) -> None:

        CHROMA_DB_PATH.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_PATH),
            settings=Settings(
                anonymized_telemetry=False,
            ),
        )

        self.collection: Collection = (
            self.client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "description": "ARIA Knowledge Base",
                    "hnsw:space": "cosine",
                },
            )
        )

        logger.info("=" * 50)
        logger.info("CHROMADB INITIALIZED")
        logger.info("=" * 50)
        logger.info(f"Collection : {collection_name}")
        logger.info(f"Database   : {CHROMA_DB_PATH}")
        logger.info("=" * 50)

    # STORE DOCUMENTS

    def store_documents(
        self,
        documents: list[Document],
        embeddings: np.ndarray,
    ) -> None:
        """
        Store Documents inside ChromaDB.

        Parameters
        ----------
        documents
            Documents produced by the ingestion pipeline.

        embeddings
            Embeddings returned by DocumentEmbedder.
        """

        if not documents:
            logger.warning("No documents received for storage.")
            return

        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents and embeddings must match."
            )

        ids = []
        texts = []
        metadatas = []
        vectors = []

        seen_ids = set()

        for document, embedding in zip(documents, embeddings):

            if not document.text_content.strip():
                logger.warning(
                    f"Skipping empty document: {document.document_id}"
                )
                continue
            
            # Embedding dimension validation
            if embedding.shape[0] != EXPECTED_EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Invalid embedding dimension for "
                    f"{document.document_id}. "
                    f"Expected {EXPECTED_EMBEDDING_DIMENSION}, "
                    f"got {embedding.shape[0]}"
                )

            if document.document_id in seen_ids:
                raise ValueError(
                    f"Duplicate document_id detected: "
                    f"{document.document_id}"
                )

            seen_ids.add(document.document_id)

            ids.append(document.document_id)
            texts.append(document.text_content)
            metadatas.append(document.metadata or {})
            vectors.append(embedding.tolist())

        if not ids:
            logger.warning(
                "No valid documents available for storage."
            )
            return
        
        BATCH_SIZE = 4000
        
        for start in range(0,len(ids), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(ids))

            self.collection.upsert(
                ids=ids[start:end],
                documents=texts[start:end],
                embeddings=vectors[start:end],
                metadatas=metadatas[start:end],
            )   
            
            logger.info(
                f"Stored Batch {start // BATCH_SIZE + 1} "
                f"({end}/{len(ids)})"
            )

        logger.info("=" * 50)
        logger.info("DOCUMENT STORAGE SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Stored Documents : {len(ids)}")
        logger.info(
            f"Collection Size  : {self.collection.count()}"
        )
        logger.info("=" * 50)



    # SEARCH
    def similarity_search(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        source: str | None = None,
    ) -> dict[str, Any]:
        """
        Perform semantic similarity search in ChromaDB.

        Parameters
        ----------
        query_embedding
            Dense embedding of the user query.

        n_results
            Number of documents to retrieve.

        source
            Optional knowledge-base source filter.
            If provided, retrieval is restricted to that source.
        """

        if query_embedding.shape[0] != EXPECTED_EMBEDDING_DIMENSION:
            raise ValueError(
                f"Invalid query embedding dimension. "
                f"Expected {EXPECTED_EMBEDDING_DIMENSION}, "
                f"got {query_embedding.shape[0]}"
            )

        if n_results <= 0:
            raise ValueError(
                "n_results must be greater than 0."
            )

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": n_results,
        }

        if source is not None:
            query_kwargs["where"] = {
                "source": source
            }

        logger.debug(
            f"Similarity search | "
            f"Source: {source or 'ALL'} | "
            f"Top-K: {n_results}"
        )

        return self.collection.query(**query_kwargs)




    # GET DOCUMENTS

    def get_documents(
        self,
        ids: list[str] | None = None,
    ) -> dict[str, Any]:

        if ids is None:
            return self.collection.get()

        return self.collection.get(ids=ids)

    def peek(
        self,
        limit: int = 5,
    ) -> dict[str, Any]:

        return self.collection.peek(limit)

    # DELETE

    def delete_documents(
        self,
        ids: list[str],
    ) -> None:

        self.collection.delete(ids=ids)

    def reset_collection(self) -> None:
        """
        Delete all stored vectors while preserving the collection.
        """

        self.client.delete_collection(
            self.collection.name
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=self.collection.name,
                metadata={
                    "description": "ARIA Knowledge Base",
                },
            )
        )

        logger.info("Collection reset successfully.")

    def delete_collection(self) -> None:
        """
        Permanently delete the collection.
        """

        self.client.delete_collection(
            self.collection.name
        )

        logger.info("Collection deleted successfully.")

    # COLLECTION INFO

    def count(self) -> int:
        """
        Number of vectors stored.
        """

        return self.collection.count()

    # MAGIC METHODS

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        return (
            f"ChromaStore("
            f"collection='{self.collection.name}', "
            f"documents={self.count()})"
        )