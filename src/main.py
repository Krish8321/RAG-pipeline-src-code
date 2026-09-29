import os
import sys
from dotenv import load_dotenv

from src.config.logging_config import setup_logger

from src.embeddings.embedder import DocumentEmbedder
from src.llm import OllamaClient, GeminiClient

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
load_dotenv()


def main() -> None:
    """
    Run the complete ARIA RAG pipeline.
    """

    # ============================================================
    # 1. USER SECURITY QUERY
    # ============================================================
    alert = {
        "timestamp": "2026-09-29T07:51:33.085198+00:00",
        "prediction_class": 1,
        "triage_action": "ESCALATE — automated model flagged anomalous behaviour",
        "risk_score": 0.85,
        "mitre_techniques": [
            "T1059 — Command & Scripting Interpreter",
            "T1071 — Application Layer Protocol"
        ],
        "iocs_extracted": {
            "agent_name": "Victus_host",
            "agent_ip": "127.0.0.1",
            "feature_snapshot": {
                "Network_I_ActiveNIC_TCP_APS": 9430.0,
                "Process_Pool_Paged Bytes": 127588240.0,
                "Process_Handle Count": 183315.0,
                "Memory Free System Page Table Entries": 4288201244.0,
                "Process_Virtual_Bytes": 609522551930880.0,
                "Memory System Cache Resident Bytes": 168275968.0,
                "Process_Virtual_Bytes Peak": 611769608167424.0,
                "Process_Thread Count": 5831.0,
                "Process_Working Set": 15218671616.0,
                "Process_Working_Set_Peak": 39099113472.0,
                "Network_I_ActiveNIC_ TCP Active RSC Connections": 0.0,
                "Process_Page_File Bytes": 20492898304.0,
                "Process_Working_Set_ Private": 7479660544.0,
                "Memory Standby Cache Normal Priority Bytes": 2349830144.0,
                "Network_I_ActiveNIC_ Bytes Sent sec": 0.0
            }
        },
        "ai_reasoning": "Random Forest predicted class 1 (non-zero -> anomalous). Top contributing features: Network_I_ActiveNIC_TCP_APS=9430.00, Process_Thread Count=5831.00, Process_Working Set=15218671616.00.",
        "source_event_id": "1790668292.222204"
    }

    user_query = f"Anomalous host activity on Victus_host (127.0.0.1) - Event ID {alert['source_event_id']} - T1059 T1071"

    # ============================================================
    # 2. INITIALIZE COMPONENTS
    # ============================================================
    
    planner = RetrievalPlanner(
        top_k_per_source=5
    )

    llm_backend = os.getenv("LLM_BACKEND", os.getenv("LLM_PROVIDER", "ollama")).lower().strip()

    if llm_backend == "gemini":
        model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
        llm_client = GeminiClient(model=model_name)
    else:
        model_name = os.getenv("OLLAMA_MODEL", "qwen3:8b")
        llm_client = OllamaClient(model=model_name)

    query_generator = QueryGenerator(
        llm_client=llm_client
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
        alert_metadata=alert,
    )

    # ============================================================
    # 10. GENERATE FINAL SOC RESPONSE
    # ============================================================

    response = llm_client.generate(
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
    # 11. DISPLAY RENDERED EVIDENCE SECTION & FINAL LLM RESPONSE
    # ============================================================

    print("\n" + "=" * 80)
    print("RENDERED PROMPT EVIDENCE SECTION:")
    print("=" * 80)

    # Extract and display the RETRIEVED SECURITY KNOWLEDGE block from rendered prompt
    if "RETRIEVED SECURITY KNOWLEDGE:" in prompt:
        evidence_block = prompt.split("RETRIEVED SECURITY KNOWLEDGE:")[1].split("============================================================")[0].strip()
        print("RETRIEVED SECURITY KNOWLEDGE:")
        print(evidence_block.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8'))
    else: 
        print(prompt.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8'))

    print("\n" + "=" * 80)
    print("Final ARIA Response : ")
    print("=" * 80)

    print(response.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8'))


if __name__ == "__main__":
    main()