from src.config.logging_config import setup_logger
from src.llm.ollama_client import OllamaClient
from src.retrieval.query_generator import QueryGenerator


logger = setup_logger()


def main():

    # ---------------------------------------------------------
    # Initialize LLM client
    # ---------------------------------------------------------

    ollama_client = OllamaClient(
        model="qwen3:8b"
    )

    # ---------------------------------------------------------
    # Initialize your existing MultiQuery / QueryGenerator
    # ---------------------------------------------------------

    multi_query = QueryGenerator(
        llm_client=ollama_client
    )

    # ---------------------------------------------------------
    # Test query
    # ---------------------------------------------------------

    user_query = (
        "PsExec execution detected from a workstation to a domain controller using ADMIN$ share and remote service execution"
    )

    print("=" * 80)
    print("MULTI-QUERY GENERATION TEST")
    print("=" * 80)

    print("\nOriginal Query:")
    print(user_query)

    print("\nGenerating queries...")
    print("-" * 80)

    # YOUR EXISTING MULTI-QUERY FUNCTION
    generated_queries = multi_query.generate_queries(user_query)

    print("\nGenerated Queries:")
    print("-" * 80)

    for index, query in enumerate(generated_queries, start=1):
        print(f"{index}. {query}")

    print("\n" + "=" * 80)
    print("TOTAL GENERATED QUERIES:", len(generated_queries))
    print("=" * 80)


if __name__ == "__main__":
    main()