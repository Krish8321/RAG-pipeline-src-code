from pathlib import Path
import json

from src.config.settings import Settings
from src.config.logging_config import setup_logger
from src.models.document import Document

logger = setup_logger()


class DocumentLoader:
    """
    Loads cybersecurity documents from the raw data directory.
    """

    SUPPORTED_EXTENSIONS = {
        ".json",
    }
    
    ID_FIELDS = (
        "mitre_id",
        "capec_id",
        "cve_id",
        "sigma_id",
    )

    def __init__(self):
        self.raw_data_dir = Settings.RAW_DATA_DIR

    # File Discovery

    def discover_files(self) -> list[Path]:
        """
        Recursively discover all supported files.

        Returns:
            list[Path]: List of discovered file paths.
        """

        files = []

        for file_path in self.raw_data_dir.rglob("*"):

            if (
                file_path.is_file()
                and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ):
                files.append(file_path)

        logger.info(f"Discovered {len(files)} supported files.")

        return files

    # Document Loading

    def load_documents(self) -> list[Document]:
        """
        Load and validate all documents from discovered JSON files.

        Returns:
            list[Document]: List of validated Document objects.
        """

        files = self.discover_files()

        documents: list[Document] = []

        total_files = len(files)
        loaded_files = 0
        failed_files = 0

        valid_documents = 0
        invalid_documents = 0

        for file_path in files:

            logger.info(f"Loading {file_path.name}")

            try:

                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)

                loaded_files += 1

                if isinstance(data, list):

                    file_valid = 0

                    for document in data:

                        if self._validate_document(document):

                            documents.append(
                                self._create_document(document)
                            )

                            valid_documents += 1
                            file_valid += 1

                        else:

                            invalid_documents += 1

                            logger.warning(
                                f"Skipping invalid document in {file_path.name}"
                            )

                    logger.info(
                        f"Loaded {file_valid} valid records from {file_path.name}"
                    )

                elif isinstance(data, dict):

                    if self._validate_document(data):

                        documents.append(
                            self._create_document(data)
                        )

                        valid_documents += 1

                        logger.info(
                            f"Loaded 1 valid record from {file_path.name}"
                        )

                    else:

                        invalid_documents += 1

                        logger.warning(
                            f"Invalid document structure in {file_path.name}"
                        )

                else:

                    logger.warning(
                        f"Unsupported JSON structure in {file_path.name}"
                    )

            except json.JSONDecodeError:

                failed_files += 1

                logger.error(
                    f"Invalid JSON file: {file_path.name}"
                )

            except Exception as e:

                failed_files += 1

                logger.error(
                    f"Error loading {file_path.name}: {e}"
                )

        self._log_summary(
            total_files,
            loaded_files,
            failed_files,
            valid_documents,
            invalid_documents,
        )

        return documents

    # Validation

    def _validate_document(self, document: dict) -> bool:
        """
        Validate the document structure.
        """
        
        if not isinstance(document,dict):
            return False
        
        if "metadata" not in document:
            return False
        
        if "text_content" not in document:
            return False
        
        if not isinstance(document["metadata"],dict):
            return False
        
        if not document["text_content"].strip():
            return False
        
        return True

        # return (
        #     isinstance(document, dict)
        #     and "metadata" in document
        #     and "text_content" in document
        # )


    # Factory

    def _create_document(self, document: dict) -> Document:
        """
        Create a Document object from a validated dictionary.
        """
        
        metadata = document["metadata"]

        return Document(
            document_id=self._generate_document_id(metadata),
            metadata=document["metadata"],
            text_content=document["text_content"],
        )
        
    def _generate_document_id(self, metadata: dict[str, str]) -> str:
        """
        Generate a unique and deterministic document ID from metadata.

        Priority:
            1. MITRE ID
            2. CAPEC ID
            3. CVE ID (CISA / NVD)
            4. Sigma Rule ID
            5. Playbook Name
        """

        source = metadata.get("source")

        if not source:
            raise ValueError("Missing 'source' in document metadata.")

        # Search for the unique identifier field
        for field in self.ID_FIELDS:
            value = metadata.get(field)

            if value:
                return f"{source}_{value}"

        # Playbooks don't have a dedicated ID
        name = metadata.get("name")

        if name:
            return f"{source}_{name.strip().replace(' ', '_')}"

        raise ValueError(
            f"Unable to generate document ID for source '{source}'. "
            "No supported identifier found."
        )
        

    # Logging

    def _log_summary(
        self,
        total_files: int,
        loaded_files: int,
        failed_files: int,
        valid_documents: int,
        invalid_documents: int,
    ) -> None:
        """
        Log loader statistics.
        """

        logger.info("=" * 50)
        logger.info("DOCUMENT LOADING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Files Found       : {total_files}")
        logger.info(f"Files Loaded      : {loaded_files}")
        logger.info(f"Files Failed      : {failed_files}")
        logger.info(f"Valid Documents   : {valid_documents}")
        logger.info(f"Invalid Documents : {invalid_documents}")
        logger.info("=" * 50)