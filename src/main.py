from __future__ import annotations

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

from src.response_validator import ResponseValidator


logger = setup_logger()


def main() -> None:
    """
    Run the complete ARIA RAG pipeline.

    Pipeline
    --------
    User Query
        ↓
    Retrieval Planner
        ↓
    Query Generator
        ↓
    Multi-Query Retriever
        ↓
    Deduplicator
        ↓
    Single-Query Reranker
        ↓
    Context Builder
        ↓
    Prompt Builder
        ↓
    Ollama LLM
        ↓
    Final SOC Response
    """

    # ============================================================
    # 1. USER SECURITY QUERY
    # ============================================================

    user_query = (
        "PsExec execution detected from a workstation to a domain controller using ADMIN$ share and remote service execution"
    )

    # ============================================================
    # 2. INITIALIZE COMPONENTS
    # ============================================================

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

    context_builder = ContextBuilder()

    prompt_builder = PromptBuilder()

    # ============================================================
    # 3. RETRIEVAL PLANNING
    # ============================================================

    retrieval_plan = planner.create_plan(
        user_query
    )

    # ============================================================
    # 4. MULTI-QUERY GENERATION
    # ============================================================

    generated_queries = query_generator.generate_queries(
        user_query
    )

    all_queries = [
        user_query
    ] + generated_queries

    # ============================================================
    # 5. MULTI-QUERY RETRIEVAL
    # ============================================================

    retrieved_documents = retriever.retrieve_multi_query(
        queries=all_queries,
        retrieval_plan=retrieval_plan,
    )

    # ============================================================
    # 6. DOCUMENT DEDUPLICATION
    # ============================================================

    deduplicated_documents = deduplicator.deduplicate(
        retrieved_documents
    )

    # ============================================================
    # 7. FINAL SINGLE-QUERY RERANKING
    #
    # All candidates from multi-query retrieval are passed
    # to the reranker.
    #
    # The reranker uses ONLY the original security query
    # to determine the final Top 10 documents.
    # ============================================================

    reranked_documents = reranker.rerank(
        query=user_query,
        documents=deduplicated_documents,
        top_k=10,
    )

    # ============================================================
    # 8. BUILD CONTEXT FROM FINAL RERANKED DOCUMENTS
    # ============================================================

    context = context_builder.build(
        documents=reranked_documents
    )

    # ============================================================
    # 9. BUILD FINAL LLM PROMPT
    # ============================================================

    prompt = prompt_builder.build(
        alert_text=user_query,
        context=context,
    )

    # ============================================================
    # 10. GENERATE FINAL SOC RESPONSE
    # ============================================================

    response = ollama_client.generate(
        prompt
    )
    
    # ============================================================
    # FINAL RESPONSE VALIDATION
    # ============================================================

    logger.info("=" * 60)
    logger.info("VALIDATING FINAL ARIA RESPONSE")
    logger.info("=" * 60)

    response_validator = ResponseValidator()

    validation_result = response_validator.validate(
        response=response,
        alert_text=user_query,
        context=context,
    )

    if validation_result.has_warnings:
        logger.warning("=" * 60)
        logger.warning("RESPONSE VALIDATION FAILED")
        logger.warning("=" * 60)

        for warning in validation_result.warnings:
            logger.warning(warning)

        logger.warning("=" * 60)

    else:
        logger.info("=" * 60)
        logger.info("RESPONSE VALIDATION PASSED")
        logger.info("=" * 60)

    # ============================================================
    # 11. DISPLAY ONLY LLM RESPONSE
    # ============================================================

    print("\n" + "=" * 80)
    print("Final ARIA Response : ")
    print("=" * 80)

    print(response)


if __name__ == "__main__":
    main()