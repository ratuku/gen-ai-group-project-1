"""Small, local retrieval pipeline for the support knowledge base.

The vector store is deliberately in memory: it is rebuilt from the Markdown
knowledge base when the process starts and therefore requires neither a
database nor credentials. Its deterministic hash embeddings keep the course
project runnable without downloading an embedding model at runtime.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
VECTOR_DIMENSIONS = 256
KNOWLEDGE_BASE_DIRECTORY = Path(__file__).resolve().parents[1] / "knowledge_base"


@dataclass(frozen=True)
class KnowledgeDocument:
    """A searchable support procedure and the metadata returned to callers."""

    source: str
    category: str
    resolution: str
    content: str


class InMemoryVectorStore:
    """Store support documents as normalized, deterministic token vectors."""

    def __init__(self, documents: list[KnowledgeDocument]) -> None:
        if not documents:
            raise ValueError("The vector store needs at least one document.")
        self.documents = documents
        self._vectors = [self._embed(document.content) for document in documents]

    @classmethod
    def from_knowledge_base(
        cls, directory: Path = KNOWLEDGE_BASE_DIRECTORY
    ) -> "InMemoryVectorStore":
        """Load every Markdown procedure from the project knowledge base."""
        documents = [_parse_document(path) for path in sorted(directory.glob("*.md"))]
        return cls(documents)

    def similarity_search(self, query: str, limit: int = 1) -> list[KnowledgeDocument]:
        """Return the most relevant procedures for a natural-language query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("A non-empty text query is required.")
        if limit < 1:
            raise ValueError("Search limit must be at least one.")

        query_vector = self._embed(query)
        ranked = sorted(
            zip(self._vectors, self.documents),
            key=lambda item: self._dot_product(query_vector, item[0]),
            reverse=True,
        )
        return [document for _, document in ranked[:limit]]

    @staticmethod
    def _embed(text: str) -> dict[int, float]:
        """Create a normalized hashing vector without an external model download."""
        counts = Counter(TOKEN_PATTERN.findall(text.lower()))
        vector: dict[int, float] = {}
        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % VECTOR_DIMENSIONS
            vector[index] = vector.get(index, 0.0) + float(count)

        magnitude = math.sqrt(sum(value * value for value in vector.values()))
        if magnitude:
            return {index: value / magnitude for index, value in vector.items()}
        return vector

    @staticmethod
    def _dot_product(left: dict[int, float], right: dict[int, float]) -> float:
        if len(left) > len(right):
            left, right = right, left
        return sum(value * right.get(index, 0.0) for index, value in left.items())


def _parse_document(path: Path) -> KnowledgeDocument:
    """Extract the response fields from one consistently formatted procedure."""
    content = path.read_text(encoding="utf-8")
    category_match = re.search(r"^## Ticket Category\s*\n+(.+)$", content, re.MULTILINE)
    resolution_match = re.search(
        r"^## Resolution\s*\n+(.*?)(?=^## |\Z)", content, re.MULTILINE | re.DOTALL
    )
    if category_match is None or resolution_match is None:
        raise ValueError(f"Knowledge-base document has required sections missing: {path}")

    category = category_match.group(1).strip().lower().replace(" ", "_")
    resolution = " ".join(resolution_match.group(1).split())
    return KnowledgeDocument(
        source=path.name,
        category=category,
        resolution=resolution,
        content=content,
    )


class RagPipeline:
    """Retrieve the most relevant support procedure and return its ticket fields."""

    def __init__(self, vector_store: InMemoryVectorStore | None = None) -> None:
        self.vector_store = vector_store or InMemoryVectorStore.from_knowledge_base()

    def resolve(self, issue: str) -> dict[str, str]:
        """Return a category and resolution selected from the knowledge base."""
        document = self.vector_store.similarity_search(issue)[0]
        return {
            "category": document.category,
            "resolution": document.resolution,
        }
