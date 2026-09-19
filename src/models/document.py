from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    """
    Represents a single document flowing through the RAG pipeline.
    """
    document_id: str 
    text_content: str 
    metadata: dict[str, Any] = field(default_factory=dict)