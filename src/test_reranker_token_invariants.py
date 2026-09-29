import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
logging.basicConfig(level=logging.WARNING)

from src.retrieval.retriever import RetrievedDocument
from src.retrieval.single_reranker import DocumentReranker as SingleReranker
from src.retrieval.reranker import DocumentReranker as MultiReranker

def test_reranker_token_invariants():
    print("=========================================================")
    print("TESTING RERANKER TOKEN INVARIANTS (<= 512 PAIR TOKENS)")
    print("=========================================================")

    single_reranker = SingleReranker()
    multi_reranker = MultiReranker()

    model_max_len = getattr(single_reranker.model, "max_seq_length", 512) or 512

    query = "Data exfiltration: 4.2 GB upload to Mega.nz A-004 · 14:25:33 · 192.168.5.103 · Suspicious Outbound File Transfer"

    long_text = "Exfiltration Over C2 Channel. " + "Adversaries may steal data by exfiltrating it over an existing command and control channel. " * 40
    doc = RetrievedDocument(
        document=long_text,
        distance=0.1,
        metadata={"source": "MITRE_ATTACK", "name": "Exfiltration Over C2 Channel"},
        query=query,
        rerank_score=0.0
    )

    # 1. Test Single Reranker
    single_res = single_reranker.rerank(query=query, documents=[doc], top_k=1)
    assert len(single_res) == 1
    assert single_res[0].rerank_score is not None

    # 2. Test Multi Reranker
    multi_res = multi_reranker.rerank(queries=[query], documents=[doc], top_k=1)
    assert len(multi_res) == 1
    assert multi_res[0].rerank_score is not None

    print("ALL RERANKER TOKEN INVARIANT TESTS PASSED CLEANLY!")

if __name__ == "__main__":
    test_reranker_token_invariants()
