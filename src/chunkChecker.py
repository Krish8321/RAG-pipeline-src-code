from pprint import pprint

from src.config.logging_config import setup_logger
# from src.embeddings.embedder import DocumentEmbedder
# from src.vector_store.chroma_store import ChromaStore
from src.ingestion.chunker import DocumentChunker
from src.ingestion.loader import DocumentLoader

logger = setup_logger()

def main() -> None:
    import re
    from collections import defaultdict
    from src.ingestion.cleaner import DocumentCleaner

    loader = DocumentLoader()
    cleaner = DocumentCleaner()
    chunker = DocumentChunker()
    
    # Load original documents
    original_docs = loader.load_documents()
    
    # Keep track of original playbook text content
    original_playbooks = {
        doc.metadata.get("name"): doc.text_content
        for doc in original_docs
        if doc.metadata.get("source") == "IR_PLAYBOOKS"
    }
    
    # 2. Clean documents
    cleaned_docs = cleaner.clean_documents(original_docs)
    
    # Keep track of cleaned playbook text content
    cleaned_playbooks = {
        doc.metadata.get("name"): doc.text_content
        for doc in cleaned_docs
        if doc.metadata.get("source") == "IR_PLAYBOOKS"
    }
    
    # 3. Chunk playbooks
    chunked_docs = chunker.chunk_documents(cleaned_docs)
    
    # Group chunked documents by playbook name
    playbook_chunks = defaultdict(list)
    for doc in chunked_docs:
        if doc.metadata.get("source") == "IR_PLAYBOOKS":
            pb_name = doc.metadata.get("playbook", "UNKNOWN")
            playbook_chunks[pb_name].append(doc)
            
    # Verify Data Integrity & Cleaning Reduction
    print("\n" + "=" * 90)
    print("DATA INTEGRITY & CLEANER REDUCTION CHECK (ORIGINAL -> CLEANED -> CHUNKED)")
    print("=" * 90)
    
    all_valid = True
    for pb_name, original_text in original_playbooks.items():
        clean_text = cleaned_playbooks.get(pb_name, "")
        chunks = playbook_chunks.get(pb_name, [])
        combined_text = "\n".join(chunk.text_content for chunk in chunks)
        
        orig_clean = re.sub(r"\s+", "", original_text)
        clean_clean = re.sub(r"\s+", "", clean_text)
        comb_clean = re.sub(r"\s+", "", combined_text)
        
        # Characters removed by cleaner (TLP tags, page numbers, metadata headers, etc.)
        chars_removed = len(orig_clean) - len(clean_clean)
        # Characters lost during chunking (must be 0)
        chunk_loss = abs(len(clean_clean) - len(comb_clean))
        
        status = "OK" if chunk_loss == 0 else "WARNING!"
        if chunk_loss != 0:
            all_valid = False
            
        print(f"Playbook: {pb_name:42} | Stripped: {chars_removed:4d} chars | Chunk Loss: {chunk_loss} ({status})")
            
    if all_valid:
        print("\nSUCCESS: All playbooks passed the chunking integrity check. 0 characters lost after cleaning!")
    else:
        print("\nWARNING: Some character mismatches were found between cleaned text and chunks.")
        
    print("\n" + "=" * 90)
    print("PLAYBOOK CHUNKING SUMMARY & HEADERS")
    print("=" * 90)
    
    for pb_name, chunks in sorted(playbook_chunks.items()):
        print(f"\nPlaybook: {pb_name} ({len(chunks)} chunks)")
        print("-" * 50)
        for idx, doc in enumerate(chunks):
            header_str = doc.metadata.get("hierarchy")
            print(f"  Chunk {idx:02d}: {header_str} (Length: {len(doc.text_content)} chars)")
            
    # Display full example of one playbook's chunks
    example_pb = "IRM-1-WormInfection"
    if example_pb in playbook_chunks:
        print("\n" + "=" * 90)
        print(f"FULL EXAMPLE DETAILS (AFTER CLEANING + CHUNKING): {example_pb}")
        print("=" * 90)
        
        for idx, doc in enumerate(playbook_chunks[example_pb]):
            print(f"\n--- CHUNK {idx} ---")
            print(f"Chunk ID:   {doc.document_id}")
            print(f"Hierarchy:  {doc.metadata.get('hierarchy')}")
            print(f"Length:     {len(doc.text_content)} characters")
            print("-" * 30)
            safe_text = doc.text_content.encode('cp1252', errors='replace').decode('cp1252')
            print(safe_text)
            print("-" * 50)


if __name__ == "__main__":
    main()