import re

from src.config.logging_config import setup_logger
from src.models.document import Document

logger = setup_logger()


class DocumentChunker:
    """
    Semantic document chunker.

    Strategy
    --------
    MITRE/CAPEC/CISA:
        Returned unchanged because every document is already atomic.

    Playbooks:
        Split by semantic sections.
        If a section becomes too large, split by subsection.
        Recursive splitting can later be added as a final fallback.
    """

    SECTION_HEADERS = [
        "ABSTRACT",
        "PREPARATION",
        "IDENTIFICATION",
        "CONTAINMENT",
        "REMEDIATION",
        "RECOVERY",
        "LESSONS LEARNED",
    ]

    # If a section exceeds this, we'll try subsection splitting.
    MAX_SECTION_LENGTH = 1800
    # Preferred chunk size, not a strict limit.
    # Semantic integrity takes priority over exact length.
    MIN_CHUNK_LENGTH = 200
    

    def chunk_documents(
        self,
        documents: list[Document],
    ) -> list[Document]:

        chunked_documents: list[Document] = []

        for document in documents:

            source = document.metadata.get("source")

            if source != "IR_PLAYBOOKS":
                chunked_documents.append(document)
                continue

            chunked_documents.extend(
                self._chunk_playbook(document)
            )

        logger.info(
            f"Chunking complete. Generated {len(chunked_documents)} documents."
        )

        return chunked_documents

    # ==========================================================
    # PLAYBOOK CHUNKING
    # ==========================================================

    def _chunk_playbook(
        self,
        document: Document,
    ) -> list[Document]:

        text = document.text_content

        # Match headers case-insensitively, allowing optional TLP watermark prefix and spaces inside header words
        pattern = (
            r"(?mi)^(?:TLP\s*:\s*[A-Z]+)?\s*("
            + "|".join(h.replace(" ", r"\s*") for h in self.SECTION_HEADERS)
            + r")"
        )

        matches = list(re.finditer(pattern, text))

        if not matches:
            logger.warning(
                f"No section headers found in playbook {document.metadata.get('name')}"
            )
            return [document]

        chunks = []

        playbook = document.metadata.get("name", "UNKNOWN")

        for section_index, match in enumerate(matches):

            # Start at index 0 for the first chunk to include cover/metadata pages
            start = 0 if section_index == 0 else match.start()

            end = (
                matches[section_index + 1].start()
                if section_index + 1 < len(matches)
                else len(text)
            )

            raw_section = match.group(1)
            # Normalize the matched section header to its canonical uppercase name (with space if any)
            section = raw_section.upper()
            raw_section_clean = re.sub(r"\s+", "", raw_section.upper())
            for header in self.SECTION_HEADERS:
                if re.sub(r"\s+", "", header) == raw_section_clean:
                    section = header
                    break

            section_text = text[start:end].strip()
            section_text = re.sub(r"\n{3,}", "\n\n", section_text)

            # ---------------------------------------
            # Try subsection chunking
            # ---------------------------------------

            if len(section_text) > self.MAX_SECTION_LENGTH:

                subsection_chunks = self._split_subsections(
                    section_text,
                    section,
                )

                if len(subsection_chunks) > 1:

                    total = len(subsection_chunks)

                    for idx, chunk in enumerate(subsection_chunks):
                        
                        

                        # A subsection title that just repeats the section
                        # name (e.g. "Abstract" inside the ABSTRACT section)
                        # shouldn't be shown as its own hierarchy level.
                        is_section_echo = (
                            chunk["title"].strip().upper() == section.upper()
                        )
                        subsection_label = (
                            None if is_section_echo else chunk["title"]
                        )
                        hierarchy = (
                            f"PLAYBOOK > {section}"
                            if is_section_echo
                            else f"PLAYBOOK > {section} > {chunk['title']}"
                        )

                        metadata = document.metadata.copy()

                        metadata.update(
                            {
                                "playbook": playbook,
                                "section": section,
                                "subsection": subsection_label,
                                "chunk_index": idx,
                                "total_chunks": total,
                                "hierarchy": hierarchy,
                            }
                        )

                        chunks.append(
                            Document(
                                document_id=(
                                    f"{document.document_id}"
                                    f"_S{section_index:02d}"
                                    f"_{section.replace(' ', '_')}"
                                    f"_C{idx:02d}"
                                ),
                                metadata=metadata,
                                text_content=chunk["text"],
                            )
                        )

                    continue

                # No natural subsections were found, but the section is
                # still too large - fall back to a recursive split so we
                # never emit an oversized chunk.
                pieces = self._recursive_split(section_text, self.MAX_SECTION_LENGTH)

                if len(pieces) > 1:

                    total = len(pieces)

                    for idx, piece in enumerate(pieces):

                        part_label = f"part {idx + 1}/{total}"

                        metadata = document.metadata.copy()

                        metadata.update(
                            {
                                "playbook": playbook,
                                "section": section,
                                "subsection": part_label,
                                "chunk_index": idx,
                                "total_chunks": total,
                                "hierarchy": f"PLAYBOOK > {section} > {part_label}",
                            }
                        )

                        chunks.append(
                            Document(
                                document_id=(
                                    f"{document.document_id}"
                                    f"_S{section_index:02d}"
                                    f"_{section.replace(' ', '_')}"
                                    f"_P{idx:02d}"
                                ),
                                metadata=metadata,
                                text_content=piece,
                            )
                        )

                    continue

            # ---------------------------------------
            # Default semantic section
            # ---------------------------------------

            metadata = document.metadata.copy()

            metadata.update(
                {
                    "playbook": playbook,
                    "section": section,
                    "subsection": None,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "hierarchy": f"PLAYBOOK > {section}",
                }
            )

            chunks.append(
                Document(
                    document_id=(
                        f"{document.document_id}"
                        f"_S{section_index:02d}"
                        f"_{section.replace(' ', '_')}"
                    ),
                    metadata=metadata,
                    text_content=section_text,
                )
            )

        return chunks

    # ==========================================================
    # SUBSECTION SPLITTING
    # ==========================================================

    def _split_subsections(
        self,
        text: str,
        section: str,
    ) -> list[dict]:

        """
        Split a section using natural subsection titles.

        Example:

        IDENTIFICATION

        Detect the infection

        Identify the infection

        Assess the perimeter

        ...
        """

        lines = text.splitlines()

        chunks = []

        current_title = section

        current_text = []

        for line in lines:

            stripped = line.strip()
            
            # temporary
            if stripped.upper() == section.upper():
                current_text.append(line)
                continue

            is_sub_header = False
            if (
                stripped
                and stripped[0].isupper()
                and len(stripped.split()) <= 6
                and stripped not in self.SECTION_HEADERS
            ):
                # Exclude punctuation so we do not split mid-sentence
                if not stripped.endswith((".", ":", "?", "!", ",", ";")):
                    # Exclude watermarks and metadata lines
                    has_tlp = "TLP:" in stripped.upper()
                    has_irm_num = bool(re.search(r"IRM\s*#\d+", stripped, re.I))
                    has_vmm_num = bool(re.search(r"VMM\s*#\d+", stripped, re.I))
                    is_author = "author:" in stripped.lower() or "contributor:" in stripped.lower()
                    is_version = "version:" in stripped.lower()
                    
                    if not (has_tlp or has_irm_num or has_vmm_num or is_author or is_version):
                        is_sub_header = True

            if is_sub_header:

                if current_text:

                    chunks.append(
                        {
                            "title": current_title,
                            "text": "\n".join(current_text).strip(),
                        }
                    )

                # current_title = stripped
                current_title = stripped.title()
                current_text = [line]

            else:

                current_text.append(line)

        if current_text:

            chunks.append(
                {
                    "title": current_title,
                    "text": "\n".join(current_text).strip(),
                }
            )

        # Merge chunks that are too small (e.g. less than 120 chars) into the previous chunk to reduce noise
        merged_chunks = []
        for chunk in chunks:
            clean_text = chunk["text"]
            if len(clean_text) < 120 and merged_chunks:
                merged_chunks[-1]["text"] += "\n" + chunk["text"]
            else:
                merged_chunks.append(chunk)

        # Deduplicate subsection titles within this section (e.g. two
        # "Detection" subsections) so retrieval metadata stays unambiguous.
        seen_titles: dict[str, int] = {}
        for chunk in merged_chunks:
            title = chunk["title"]
            seen_titles[title] = seen_titles.get(title, 0) + 1
            if seen_titles[title] > 1:
                chunk["title"] = f"{title} ({seen_titles[title]})"

        # Recursive fallback: a single natural subsection can itself still be
        # larger than MAX_SECTION_LENGTH. Split those further instead of
        # emitting an oversized chunk.
        final_chunks = []
        for chunk in merged_chunks:

            if len(chunk["text"]) <= self.MAX_SECTION_LENGTH:
                final_chunks.append(chunk)
                continue

            pieces = self._recursive_split(chunk["text"], self.MAX_SECTION_LENGTH)
            total_parts = len(pieces)

            for part_idx, piece in enumerate(pieces, start=1):
                title = (
                    chunk["title"]
                    if total_parts == 1
                    else f"{chunk['title']} (part {part_idx}/{total_parts})"
                )
                final_chunks.append({"title": title, "text": piece})

        return final_chunks

    # ==========================================================
    # RECURSIVE FALLBACK SPLITTING
    # ==========================================================

    def _recursive_split(
        self,
        text: str,
        max_length: int,
    ) -> list[str]:
        """
        Final fallback for text that is still too large after section and
        subsection splitting. Tries, in order: paragraph breaks, sentence
        breaks, then a hard character-length cut. Each strategy is only
        used if it actually produces more than one piece, so we never
        fragment text that doesn't need it.
        """

        text = text.strip()

        if len(text) <= max_length:
            return [text]

        paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(paragraphs) > 1:
            return self._merge_pieces(paragraphs, max_length, joiner="\n\n")

        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(sentences) > 1:
            return self._merge_pieces(sentences, max_length, joiner=" ")

        # Nothing left to split on semantically - hard cut as a last resort.
        
        pieces = [
            text[i:i + max_length].strip()
            for i in range(0, len(text), max_length)
        ]
        
        if (
            len(pieces) > 1
            and len(pieces[-1]) < self.MIN_CHUNK_LENGTH
        ):
            pieces[-2] += "\n" + pieces[-1]
            pieces.pop()
            
        return pieces
        
        # return [
        #     text[i:i + max_length].strip()
        #     for i in range(0, len(text), max_length)
        # ]

    def _merge_pieces(
        self,
        pieces: list[str],
        max_length: int,
        joiner: str,
    ) -> list[str]:
        """
        Greedily re-merge small pieces (paragraphs/sentences) back up to
        max_length, so we don't over-fragment into tiny chunks. Any single
        piece that is itself still too large is recursively split.
        """

        result = []
        current = ""

        for piece in pieces:

            candidate = f"{current}{joiner}{piece}" if current else piece

            if len(candidate) <= max_length:
                current = candidate
                continue

            if current:
                result.append(current)
                current = ""

            if len(piece) > max_length:
                result.extend(self._recursive_split(piece, max_length))
            else:
                current = piece

        if current:
            result.append(current)
            
        # ----------------------------------------------------------
        # Merge tiny final chunk with previous one.
        # ----------------------------------------------------------
        if (
            len(result) > 1
            and len(result[-1]) < self.MIN_CHUNK_LENGTH
        ):
            result[-2] += joiner + result[-1]
            result.pop()

        return result