from __future__ import annotations

from src.config.logging_config import setup_logger

from src.embeddings.embedder import DocumentEmbedder
from src.llm.ollama_client import OllamaClient
from src.retrieval.query_generator import QueryGenerator
from src.retrieval.retrieval_planner import RetrievalPlanner
from src.retrieval.retriever import Retriever
from src.vector_store.chroma_store import ChromaStore
from src.retrieval.deduplicator import DocumentDeduplicator
from src.retrieval.single_reranker import DocumentReranker


logger = setup_logger()


def main() -> None:

    # ==========================================================
    # USER QUERY
    # ==========================================================

    user_query = (
        "PsExec execution detected from a workstation to a domain controller using ADMIN$ share and remote service execution"
    )

    # ==========================================================
    # INITIALIZE COMPONENTS
    # ==========================================================

    planner = RetrievalPlanner(
        top_k_per_source=5
    )

    ollama_client = OllamaClient(
        model="qwen3:8b"
    )

    query_generator = QueryGenerator(
        llm_client=ollama_client
    )

    embedder = DocumentEmbedder()

    vector_store = ChromaStore()

    retriever = Retriever(
        vector_store=vector_store,
        embedder=embedder
    )

    deduplicator = DocumentDeduplicator()

    reranker = DocumentReranker()

    # ==========================================================
    # 1. CREATE RETRIEVAL PLAN
    # ==========================================================

    retrieval_plan = planner.create_plan(
        user_query
    )

    print("\n" + "=" * 80)
    print("RETRIEVAL PLAN")
    print("=" * 80)

    print(
        f"Sources          : "
        f"{retrieval_plan.sources}"
    )

    print(
        f"Top-K Per Source : "
        f"{retrieval_plan.top_k_per_source}"
    )

    # ==========================================================
    # 2. GENERATE MULTI-QUERIES
    # ==========================================================

    generated_queries = query_generator.generate_queries(
        user_query
    )

    all_queries = [
        user_query
    ] + generated_queries

    print("\n" + "=" * 80)
    print("RETRIEVAL QUERIES")
    print("=" * 80)

    for index, query in enumerate(
        all_queries,
        start=1,
    ):
        print(
            f"{index}. {query}"
        )

    # ==========================================================
    # 3. RETRIEVE DOCUMENTS
    # ==========================================================

    retrieved_documents = retriever.retrieve_multi_query(
        queries=all_queries,
        retrieval_plan=retrieval_plan,
    )

    print("\n" + "=" * 80)
    print("RAW RETRIEVAL")
    print("=" * 80)

    print(
        f"Total Retrieved Documents : "
        f"{len(retrieved_documents)}"
    )

    # ==========================================================
    # 4. DEDUPLICATE
    # ==========================================================

    deduplicated_documents = deduplicator.deduplicate(
        retrieved_documents
    )

    duplicates_removed = (
        len(retrieved_documents)
        - len(deduplicated_documents)
    )

    print("\n" + "=" * 80)
    print("DEDUPLICATION")
    print("=" * 80)

    print(
        f"Retrieved Documents : "
        f"{len(retrieved_documents)}"
    )

    print(
        f"Unique Documents    : "
        f"{len(deduplicated_documents)}"
    )

    print(
        f"Duplicates Removed  : "
        f"{duplicates_removed}"
    )

    # ==========================================================
    # 5. SINGLE-QUERY RERANKING
    # ==========================================================
    #
    # IMPORTANT:
    #
    # We deliberately pass ONLY the original user query
    # to the reranker.
    #
    # The generated queries were used ONLY to build the
    # candidate pool during retrieval.
    #
    # This gives us Experiment #2:
    #
    # Multi-query retrieval
    #        ↓
    # Deduplication
    #        ↓
    # SINGLE QUERY RERANKER
    #
    # ==========================================================

    print("\n" + "=" * 80)
    print("SINGLE QUERY RERANKING")
    print("=" * 80)

    print(
        f"Reranker Query : "
        f"{user_query}"
    )

    reranked_documents = reranker.rerank(
        query=user_query,
        documents=deduplicated_documents,
        top_k=10,
    )

    # ==========================================================
    # 6. PRINT TOP 10 RESULTS
    # ==========================================================

    print("\n" + "=" * 80)
    print("TOP 10 RERANKED DOCUMENTS")
    print("=" * 80)

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

        print("\n" + "=" * 80)
        print(
            f"RANK #{rank}"
        )
        print("=" * 80)

        print(
            f"Source          : "
            f"{source}"
        )

        print(
            f"Name            : "
            f"{name}"
        )

        print(
            f"Chroma Distance : "
            f"{document.distance:.4f}"
        )

        print(
            f"Rerank Score    : "
            f"{document.rerank_score:.4f}"
        )

        print("\nDOCUMENT CHUNK:")
        print("-" * 80)

        print(
            document.document
        )

    # ==========================================================
    # 7. RETRIEVAL SUMMARY
    # ==========================================================

    print("\n" + "=" * 80)
    print("RETRIEVAL SUMMARY")
    print("=" * 80)

    print(
        f"Total Retrieved : "
        f"{len(retrieved_documents)}"
    )

    print(
        f"Unique Chunks   : "
        f"{len(deduplicated_documents)}"
    )

    print(
        f"Duplicates      : "
        f"{duplicates_removed}"
    )

    print(
        f"Reranked Top-K  : "
        f"{len(reranked_documents)}"
    )

    # ==========================================================
    # 8. FINAL RANKING TABLE
    # ==========================================================

    print("\n" + "=" * 80)
    print("FINAL RERANKING")
    print("=" * 80)

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

        print(
            f"{rank:02d}. "
            f"{name} | "
            f"{source} | "
            f"Score: {document.rerank_score:.4f}"
        )

    print("\n" + "=" * 80)
    print("SINGLE QUERY RERANKER TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()