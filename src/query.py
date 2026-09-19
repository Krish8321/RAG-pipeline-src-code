from __future__ import annotations

import argparse
import sys

from src.config.logging_config import setup_logger

from src.embeddings.embedder import DocumentEmbedder
from src.llm.ollama_client import OllamaClient

from src.retrieval.query_generator import QueryGenerator
from src.retrieval.retrieval_planner import RetrievalPlanner
from src.retrieval.retriever import Retriever
from src.retrieval.deduplicator import DocumentDeduplicator
from src.retrieval.single_reranker import DocumentReranker

from src.vector_store.chroma_store import ChromaStore

from src.prompting.context_builder import ContextBuilder
from src.prompting.prompt_builder_dum import PromptBuilder


logger = setup_logger()

# ============================================================
# PIPELINE CONFIG
#
# Centralized here instead of scattered as inline literals so
# tuning retrieval/rerank behavior doesn't require hunting
# through main() for magic numbers.
# ============================================================

OLLAMA_MODEL = "qwen3:8b"
TOP_K_PER_SOURCE = 5
FINAL_RERANK_TOP_K = 10

DEFAULT_QUERY = (
    "PsExec execution detected from a workstation to a domain "
    "controller using ADMIN$ share and remote service execution"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ARIA SOC triage RAG pipeline on a security alert."
    )
    parser.add_argument(
        "--alert",
        type=str,
        default=DEFAULT_QUERY,
        help="Security alert text to triage. Defaults to a sample PsExec alert.",
    )
    return parser.parse_args()


def run_pipeline(user_query: str) -> str:
    """
    Run the complete ARIA RAG pipeline for a single alert and
    return the final rendered SOC response.

    Pipeline
    --------
    User Query
        -> Retrieval Planner
        -> Query Generator
        -> Multi-Query Retriever
        -> Deduplicator
        -> Single-Query Reranker
        -> Context Builder
        -> Prompt Builder
        -> Ollama LLM
        -> Final SOC Response

    Raises
    ------
    ValueError
        If user_query is empty or blank.
    Exception
        Re-raises any component failure after logging which
        pipeline stage it occurred in, so failures are
        diagnosable without a bare traceback.
    """

    if not user_query or not user_query.strip():
        raise ValueError("Alert query cannot be empty.")

    # ============================================================
    # INITIALIZE COMPONENTS
    # ============================================================

    planner = RetrievalPlanner(top_k_per_source=TOP_K_PER_SOURCE)
    ollama_client = OllamaClient(model=OLLAMA_MODEL)
    query_generator = QueryGenerator(llm_client=ollama_client)
    embedder = DocumentEmbedder()
    vector_store = ChromaStore()
    retriever = Retriever(vector_store=vector_store, embedder=embedder)
    deduplicator = DocumentDeduplicator()
    reranker = DocumentReranker()
    context_builder = ContextBuilder()
    prompt_builder = PromptBuilder()

    logger.info("=" * 60)
    logger.info("ARIA PIPELINE START")
    logger.info(f"Alert : {user_query}")
    logger.info("=" * 60)

    # ============================================================
    # RETRIEVAL PLANNING
    # ============================================================

    try:
        retrieval_plan = planner.create_plan(user_query)
    except Exception:
        logger.exception("Pipeline failed at stage: Retrieval Planning")
        raise

    # ============================================================
    # MULTI-QUERY GENERATION
    # ============================================================

    try:
        generated_queries = query_generator.generate_queries(user_query)
        all_queries = [user_query] + generated_queries
        logger.info(f"Generated {len(generated_queries)} additional queries")
    except Exception:
        logger.exception("Pipeline failed at stage: Query Generation")
        raise

    # ============================================================
    # MULTI-QUERY RETRIEVAL
    # ============================================================

    try:
        retrieved_documents = retriever.retrieve_multi_query(
            queries=all_queries,
            retrieval_plan=retrieval_plan,
        )
        logger.info(f"Retrieved {len(retrieved_documents)} raw documents")
    except Exception:
        logger.exception("Pipeline failed at stage: Multi-Query Retrieval")
        raise

    # ============================================================
    # DOCUMENT DEDUPLICATION
    # ============================================================

    try:
        deduplicated_documents = deduplicator.deduplicate(retrieved_documents)
        logger.info(
            f"Deduplicated to {len(deduplicated_documents)} documents "
            f"(removed {len(retrieved_documents) - len(deduplicated_documents)})"
        )
    except Exception:
        logger.exception("Pipeline failed at stage: Deduplication")
        raise

    # ============================================================
    # FINAL SINGLE-QUERY RERANKING
    #
    # All candidates from multi-query retrieval are passed to the
    # reranker. The reranker uses ONLY the original security query
    # to determine the final Top-K documents.
    # ============================================================

    try:
        reranked_documents = reranker.rerank(
            query=user_query,
            documents=deduplicated_documents,
            top_k=FINAL_RERANK_TOP_K,
        )
        logger.info(f"Reranked to top {len(reranked_documents)} documents")
    except Exception:
        logger.exception("Pipeline failed at stage: Reranking")
        raise

    if not reranked_documents:
        logger.warning(
            "No documents survived retrieval/reranking — proceeding "
            "with an alert-only prompt. The response may state "
            "insufficient evidence."
        )

    # ============================================================
    # BUILD CONTEXT FROM FINAL RERANKED DOCUMENTS
    # ============================================================

    try:
        context = context_builder.build(documents=reranked_documents)
    except Exception:
        logger.exception("Pipeline failed at stage: Context Building")
        raise

    # ============================================================
    # BUILD FINAL LLM PROMPT
    # ============================================================

    try:
        prompt = prompt_builder.build(
            alert_text=user_query,
            context=context,
        )
    except Exception:
        logger.exception("Pipeline failed at stage: Prompt Building")
        raise

    # ============================================================
    # GENERATE FINAL SOC RESPONSE
    # ============================================================

    try:
        response = ollama_client.generate(prompt)
    except Exception:
        logger.exception("Pipeline failed at stage: LLM Generation")
        raise

    logger.info("=" * 60)
    logger.info("ARIA PIPELINE COMPLETE")
    logger.info("=" * 60)

    return response


def main() -> None:
    args = parse_args()

    try:
        response = run_pipeline(args.alert)
    except Exception:
        logger.error("ARIA pipeline terminated due to an unhandled error.")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("Final ARIA Response : ")
    print("=" * 80)
    print(response)


if __name__ == "__main__":
    main()