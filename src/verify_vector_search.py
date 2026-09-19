from src.vector_store.chroma_store import ChromaStore
from src.embeddings.embedder import DocumentEmbedder
from src.config.logging_config import setup_logger
from pprint import pprint

logger = setup_logger()

def main():
    
    embedder = DocumentEmbedder()
    
    vector_store = ChromaStore()
    
    # results = vector_store.collection.get(
    #     where={"source": "IR_PLAYBOOKS"}
    # )   
    # print(len(results["ids"]))
    
    query = "How should I respond to ransomware?"
    # query = input("Enter the query : ")

    query_embedding = embedder.embed_query(query)

    results = vector_store.collection.query(
        query_embeddings=[query_embedding],
        where={"source": "IR_PLAYBOOKS"},
        n_results=5,
        include=["documents", "metadatas", "distances"],
    )

    from pprint import pprint
    pprint(results)
    
if __name__ == "__main__":
    main()