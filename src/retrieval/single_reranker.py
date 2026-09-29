from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.config.logging_config import setup_logger
from src.retrieval.retriever import RetrievedDocument


logger = setup_logger()


class DocumentReranker:
    """
    Re-ranks retrieved documents using a CrossEncoder.

    This version performs SINGLE-QUERY reranking.

    Responsibilities
    ----------------
    - Score query-document pairs.
    - Sort documents by relevance.
    - Return the top-K documents.

    Does NOT
    --------
    - Retrieve documents.
    - Generate queries.
    - Deduplicate documents.
    - Aggregate scores across multiple queries.
    - Filter metadata.
    - Call the LLM.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
    ) -> None:

        self.model_name = model_name

        logger.info("=" * 60)
        logger.info("INITIALIZING DOCUMENT RERANKER")
        logger.info("=" * 60)

        logger.info(
            f"Model : {self.model_name}"
        )

        self.model = CrossEncoder(
            self.model_name
        )

        # ---------------------------------------------------------
        # RERANKER TOKENIZER DIAGNOSTICS
        # ---------------------------------------------------------

        logger.info(
            f"Reranker Max Length : {self.model.max_length}"
        )

        logger.info(
            f"Tokenizer Max Length : "
            f"{self.model.tokenizer.model_max_length}"
        )

        logger.info(
            "DOCUMENT RERANKER INITIALIZED"
        )

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int = 10,
        min_playbook_slots: int = 1,
        playbook_relevance_floor: float = 0.010,
    ) -> list[RetrievedDocument]:
        """
        Re-rank retrieved documents using a single query.

        Each document is scored exactly once against
        the supplied query.

        Parameters
        ----------
        query:
            Single query used for reranking.

        documents:
            Deduplicated documents retrieved from
            the retrieval stage.

        top_k:
            Number of highest-scoring documents to return.

        Returns
        -------
        list[RetrievedDocument]
            Documents sorted by CrossEncoder relevance score.
        """

        # ---------------------------------------------------------
        # INPUT VALIDATION
        # ---------------------------------------------------------

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        if not isinstance(documents, list):
            raise TypeError(
                "documents must be a list."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if not documents:
            logger.warning(
                "No documents available for re-ranking."
            )
            return []

        top_k = min(
            top_k,
            len(documents),
        )

        # ---------------------------------------------------------
        # RERANKING INFORMATION
        # ---------------------------------------------------------

        logger.info("=" * 60)
        logger.info("DOCUMENT RE-RANKING")
        logger.info("=" * 60)

        logger.info(
            f"Query           : {query}"
        )

        logger.info(
            f"Input Documents : {len(documents)}"
        )

        logger.info(
            f"Requested Top-K : {top_k}"
        )

        # ---------------------------------------------------------
        # DOCUMENT TOKEN LENGTH DIAGNOSTICS
        # ---------------------------------------------------------

        logger.info("-" * 60)
        logger.info("DOCUMENT TOKEN LENGTHS")
        logger.info("-" * 60)

        tokenizer = self.model.tokenizer

        for document in documents:

            name = document.metadata.get(
                "name",
                "UNKNOWN",
            )

            text = document.document

            # Do NOT truncate here.
            # We are measuring the actual document size.
            tokenized = tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=512,
            )

            token_count = len(
                tokenized["input_ids"]
            )

            logger.info(
                f"Name: {name} | "
                f"Characters: {len(text)} | "
                f"Tokens: {token_count}"
            )

        # ---------------------------------------------------------
        # CREATE QUERY-DOCUMENT PAIRS
        # ---------------------------------------------------------

        pairs = []

        # ---------------------------------------------------------
        # CROSS-ENCODER PREDICTION WITH DYNAMIC SLIDING-WINDOW MAX-POOLING
        # ---------------------------------------------------------

        logger.info("-" * 60)
        logger.info("CROSS-ENCODER PREDICTION")
        logger.info("-" * 60)

        model_max_len = getattr(self.model, "max_seq_length", 512) or 512
        query_tokens = len(self.model.tokenizer.encode(query, add_special_tokens=False, truncation=True, max_length=model_max_len))
        special_tokens = 4  # [CLS], [SEP], [SEP] + 1 token safety margin

        batch_pairs: list[list[str]] = []
        doc_window_counts: list[int] = []
        max_pair_tokens_observed = 0

        for document in documents:
            name = document.metadata.get("name", "")
            prefix_header = f"Document Name: {name}\n"
            header_tokens = len(self.model.tokenizer.encode(prefix_header, add_special_tokens=False, truncation=True, max_length=model_max_len))

            max_doc_tokens = max(50, model_max_len - query_tokens - header_tokens - special_tokens)
            overlap = min(50, max_doc_tokens // 4)
            step = max(10, max_doc_tokens - overlap)

            doc_text = document.document
            doc_tokens = self.model.tokenizer.encode(doc_text, add_special_tokens=False, truncation=True, max_length=4096)

            if len(doc_tokens) <= max_doc_tokens:
                full_text = f"{prefix_header}{doc_text}"
                batch_pairs.append([query, full_text])
                doc_window_counts.append(1)
                pair_len = len(self.model.tokenizer.encode(query, full_text, add_special_tokens=True, truncation=True, max_length=model_max_len))
                max_pair_tokens_observed = max(max_pair_tokens_observed, pair_len)
            else:
                windows = []
                for i in range(0, len(doc_tokens), step):
                    w_toks = doc_tokens[i : i + max_doc_tokens]
                    w_text = self.model.tokenizer.decode(w_toks, skip_special_tokens=True)
                    w_full_text = f"{prefix_header}{w_text}"
                    windows.append(w_full_text)
                    pair_len = len(self.model.tokenizer.encode(query, w_full_text, add_special_tokens=True, truncation=True, max_length=model_max_len))
                    max_pair_tokens_observed = max(max_pair_tokens_observed, pair_len)

                for w in windows:
                    batch_pairs.append([query, w])
                doc_window_counts.append(len(windows))

        logger.info(f"Reranker Model Max Seq Length : {model_max_len}")
        logger.info(f"Query Token Count            : {query_tokens}")
        logger.info(f"Total Sub-Windows Scored     : {len(batch_pairs)}")
        logger.info(f"Maximum Actual Pair Tokens    : {max_pair_tokens_observed} / {model_max_len}")
        logger.info(f"Tokenizer Overflow Detected   : {'NONE (0 pairs > ' + str(model_max_len) + ')' if max_pair_tokens_observed <= model_max_len else 'WARNING: OVERFLOW DETECTED'}")

        raw_scores = self.model.predict(batch_pairs)

        logger.info("Cross-Encoder prediction completed.")

        # ---------------------------------------------------------
        # ASSIGN SCORES TO DOCUMENTS (MAX-POOLED)
        # ---------------------------------------------------------

        logger.info("-" * 60)
        logger.info("ASSIGNING RERANKER SCORES")
        logger.info("-" * 60)

        scored_documents: list[RetrievedDocument] = []
        score_idx = 0

        for document, w_count in zip(documents, doc_window_counts):
            doc_scores = raw_scores[score_idx : score_idx + w_count]
            score_idx += w_count

            best_score = max(doc_scores) if len(doc_scores) > 0 else float("-inf")
            document.rerank_score = float(best_score)
            scored_documents.append(document)

            name = document.metadata.get(
                "name",
                "UNKNOWN",
            )

            logger.debug(
                f"Name: {name} | "
                f"Score: {float(best_score):.4f}"
            )

        # ---------------------------------------------------------
        # RERANKER INPUT PREVIEW
        # ---------------------------------------------------------

        logger.info("-" * 60)
        logger.info("RERANKER INPUT PREVIEW")
        logger.info("-" * 60)

        for document in documents:

            name = document.metadata.get(
                "name",
                "UNKNOWN",
            )

            logger.info(
                f"Name: {name}"
            )

        # ---------------------------------------------------------
        # SORT BY RERANKER SCORE
        # ---------------------------------------------------------

        scored_documents.sort(
            key=lambda document:
                document.rerank_score
                if document.rerank_score is not None
                else float("-inf"),
            reverse=True,
        )

        # ---------------------------------------------------------
        # SELECT TOP-K WITH GATED PLAYBOOK SLOT RESERVATION
        # ---------------------------------------------------------

        all_playbook_candidates = [
            doc for doc in scored_documents
            if doc.metadata.get("source") == "IR_PLAYBOOKS" and doc.rerank_score is not None
        ]
        if all_playbook_candidates:
            pb_scores_str = ", ".join(
                f"{doc.metadata.get('name', 'UNKNOWN')}={doc.rerank_score:.4f}"
                for doc in all_playbook_candidates
            )
            logger.info(f"Playbook candidates considered ({len(all_playbook_candidates)} total): {pb_scores_str}")
        else:
            logger.info("Playbook candidates considered: None retrieved")

        playbook_docs = [
            doc for doc in all_playbook_candidates
            if doc.rerank_score >= playbook_relevance_floor
        ]

        if min_playbook_slots > 0 and playbook_docs:
            selected_playbook_docs = playbook_docs[:min_playbook_slots]
            selected_ids = {id(doc) for doc in selected_playbook_docs}

            remaining_slots = max(0, top_k - len(selected_playbook_docs))
            remaining_docs = [
                doc for doc in scored_documents
                if id(doc) not in selected_ids
            ]

            reranked_documents = selected_playbook_docs + remaining_docs[:remaining_slots]
            reranked_documents.sort(
                key=lambda doc: doc.rerank_score if doc.rerank_score is not None else float("-inf"),
                reverse=True,
            )
            logger.info(
                f"Gated Playbook Allocation: Reserved {len(selected_playbook_docs)} IR_PLAYBOOKS "
                f"doc(s) (score >= {playbook_relevance_floor})."
            )
        else:
            reranked_documents = scored_documents[:top_k]

        # ---------------------------------------------------------
        # LOG TOP RESULTS
        # ---------------------------------------------------------

        logger.info("-" * 60)
        logger.info("TOP RE-RANKED DOCUMENTS")
        logger.info("-" * 60)

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

            logger.info(
                f"Rank {rank:02d} | "
                f"Score: {document.rerank_score:.4f} | "
                f"Source: {source} | "
                f"Name: {name}"
            )

        logger.info("=" * 60)
        logger.info(
            "DOCUMENT RE-RANKING COMPLETE"
        )
        logger.info("=" * 60)

        logger.info(
            f"Documents Before : "
            f"{len(documents)}"
        )

        logger.info(
            f"Documents After  : "
            f"{len(reranked_documents)}"
        )

        return reranked_documents