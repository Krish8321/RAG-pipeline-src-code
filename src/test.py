from __future__ import annotations

from src.embeddings.embedder import DocumentEmbedder
from src.vector_store.chroma_store import ChromaStore


def main() -> None:

    # ============================================================
    # TEST QUERY
    # ============================================================

    query = (
        "PsExec execution detected from a workstation to a domain "
        "controller using ADMIN$ share and remote service execution"
    )

    # Number of chunks to inspect from each source
    TOP_K = 5

    # ============================================================
    # INITIALIZE
    # ============================================================

    embedder = DocumentEmbedder()
    vector_store = ChromaStore()

    # ============================================================
    # GENERATE QUERY EMBEDDING
    # ============================================================

    query_embedding = embedder.embed_query(query)

    print("\n")
    print("=" * 100)
    print("CHUNK INSPECTION")
    print("=" * 100)

    print(f"\nQuery:")
    print(query)

    # ============================================================
    # CHECK EACH SOURCE
    # ============================================================

    sources = [
        "MITRE_ATTACK",
        "SIGMA_RULES",
        "IR_PLAYBOOKS",
        "CISA_KEV",
        "NVD_CVE",
        "MITRE_CAPEC"
    ]

    for source in sources:

        print("\n")
        print("=" * 100)
        print(f"SOURCE: {source}")
        print("=" * 100)

        # --------------------------------------------------------
        # Query ChromaDB for this source only
        # --------------------------------------------------------

        results = vector_store.collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K,
            where={
                "source": source
            },
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not documents:
            print("\nNo chunks found for this source.")
            continue

        print(
            f"\nRetrieved {len(documents)} chunks"
        )

        # --------------------------------------------------------
        # PRINT CHUNKS
        # --------------------------------------------------------

        for rank, (
            document,
            metadata,
            distance,
        ) in enumerate(
            zip(
                documents,
                metadatas,
                distances,
            ),
            start=1,
        ):

            metadata = metadata or {}

            print("\n")
            print("-" * 100)
            print(f"RANK #{rank}")
            print("-" * 100)

            print(
                f"Name           : "
                f"{metadata.get('name', 'UNKNOWN')}"
            )

            print(
                f"Type           : "
                f"{metadata.get('type', 'UNKNOWN')}"
            )

            print(
                f"Source         : "
                f"{metadata.get('source', 'UNKNOWN')}"
            )

            print(
                f"Chroma Distance: "
                f"{distance}"
            )

            print("\nDOCUMENT:")
            print(document)

    print("\n")
    print("=" * 100)
    print("CHUNK INSPECTION COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()