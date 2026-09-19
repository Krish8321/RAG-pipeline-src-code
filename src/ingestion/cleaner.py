import re

from src.config.logging_config import setup_logger
from src.models.document import Document

logger = setup_logger()


class DocumentCleaner:
    """
    Cleans and validates documents before chunking.
    """

    # Public API

    def clean_documents(self, documents: list[Document]) -> list[Document]:
        """
        Clean and validate all documents.

        Args:
            documents (list[Document]): Documents from the loader.

        Returns:
            list[Document]: Cleaned documents.
        """

        cleaned_documents: list[Document] = []
        skipped_documents = 0

        for document in documents:

            try:

                if not self._validate_document(document):
                    skipped_documents += 1
                    continue

                cleaned_documents.append(
                    self._clean_document(document)
                )

            except Exception as e:

                skipped_documents += 1

                logger.error(f"Error cleaning document: {e}")

        self._log_summary(
            total_documents=len(documents),
            cleaned_documents=len(cleaned_documents),
            skipped_documents=skipped_documents,
        )

        return cleaned_documents

    # Validation

    def _validate_document(self, document: Document) -> bool:
        """
        Validate a Document object.
        """

        if not isinstance(document, Document):

            logger.warning("Invalid Document object.")

            return False

        if not isinstance(document.metadata, dict):

            logger.warning("Invalid metadata.")

            return False

        if not isinstance(document.text_content, str):

            logger.warning("Invalid text content.")

            return False

        if not document.text_content.strip():

            logger.warning("Empty document.")

            return False

        return True

    # Cleaning

    def _clean_document(self, document: Document) -> Document:
        """
        Clean a single document.
        """

        text = document.text_content

        text = self._strip_text(text)
        text = self._normalize_whitespace(text)
        text = self._normalize_newlines(text)
        
        source = document.metadata.get("source")
        
        if source == "IR_PLAYBOOKS":
            text = self.clean_playbook(text)

        return Document(
            document_id=document.document_id,
            metadata=document.metadata,
            text_content=text,
        )

    # Text Cleaning Helpers

    def _strip_text(self, text: str) -> str:
        """
        Remove leading and trailing whitespace.
        """

        return text.strip()

    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace.
        """

        text = text.replace("\r", "\n")

        text = re.sub(r"[ \t]+", " ", text)

        text = re.sub(r" +\n", "\n", text)

        text = re.sub(r"\n +", "\n", text)

        return text

    def _normalize_newlines(self, text: str) -> str:
        """
        Collapse multiple blank lines into two.
        """

        return re.sub(r"\n{3,}", "\n\n", text)
    
    # Playbook Cleaning 
    
    def clean_playbook(self, text: str) -> str:
        """
        Remove PDF extraction artifacts from Incident Response playbooks.
        """

        # Remove repeated TLP labels
        text = re.sub(r"TLP:CLEAR", "", text)

        # Remove standalone page numbers
        text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

        # Remove metadata lines
        text = re.sub(r"IRM Author:.*", "", text)
        text = re.sub(r"Contributor:.*", "", text)
        text = re.sub(r"IRM version:.*", "", text)
        text = re.sub(r"E-Mail:.*", "", text)
        text = re.sub(r"Web:.*", "", text)
        text = re.sub(r"Twitter:.*", "", text)

        # Collapse blank lines again
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
    

    # Logging

    def _log_summary(
        self,
        total_documents: int,
        cleaned_documents: int,
        skipped_documents: int,
    ) -> None:
        """
        Log cleaning statistics.
        """

        logger.info("=" * 50)
        logger.info("DOCUMENT CLEANING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Input Documents   : {total_documents}")
        logger.info(f"Cleaned Documents : {cleaned_documents}")
        logger.info(f"Skipped Documents : {skipped_documents}")
        logger.info("=" * 50)