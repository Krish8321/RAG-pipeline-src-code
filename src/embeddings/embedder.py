from time import perf_counter

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config.logging_config import setup_logger
from src.config.settings import Settings
from src.models.document import Document

logger = setup_logger()


class DocumentEmbedder:
    """
    Generates dense vector embeddings for documents.

    Responsibilities
    ----------------
    - Load the embedding model once.
    - Generate normalized embeddings in batches.
    - Return embeddings in the same order as the input documents.

    Does NOT
    --------
    - Store embeddings
    - Communicate with ChromaDB
    - Perform retrieval
    """

    def __init__(self) -> None:

        self.model_name = Settings.EMBEDDING_MODEL
        self.batch_size = Settings.BATCH_SIZE
        self.normalize_embeddings = True

        # logger.info("=" * 50)
        # logger.info("LOADING EMBEDDING MODEL")
        # logger.info("=" * 50)
        # logger.info(f"Model : {self.model_name}")

        start = perf_counter()

        self.model = SentenceTransformer(
            self.model_name,
            trust_remote_code=False,
        )

        self.embedding_dimension = (
            self.model.get_embedding_dimension()
        )

        elapsed = perf_counter() - start

        # logger.info(
        #     f"Embedding Dimension : {self.embedding_dimension}"
        # )
        # logger.info(
        #     f"Model loaded in {elapsed:.2f} seconds."
        # )
        # logger.info("=" * 50)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate an embedding for a single query.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        
        return self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
    )

    def embed_documents(
        self,
        documents: list[Document],
    ) -> np.ndarray:
        """
        Generate embeddings for a list of documents.

        Parameters
        ----------
        documents : list[Document]

        Returns
        -------
        np.ndarray
            Shape -> (num_documents, embedding_dimension)
        """

        if not documents:

            # logger.warning("No documents received for embedding.")

            return np.empty((0, self.embedding_dimension))

        start = perf_counter()

        texts = [
            document.text_content
            for document in documents
        ]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
        )

        elapsed = perf_counter() - start

        # logger.info("=" * 50)
        # logger.info("DOCUMENT EMBEDDING SUMMARY")
        # logger.info("=" * 50)
        # logger.info(f"Input Documents     : {len(documents)}")
        # logger.info(f"Embedding Model     : {self.model_name}")
        # logger.info(
        #     f"Embedding Dimension : {self.embedding_dimension}"
        # )
        # logger.info(f"Batch Size          : {self.batch_size}")
        # logger.info(
        #     f"Normalize           : {self.normalize_embeddings}"
        # )
        # logger.info(f"Vectors Generated   : {len(embeddings)}")
        # logger.info(f"Time Taken          : {elapsed:.2f} sec")
        # logger.info("=" * 50)

        return embeddings