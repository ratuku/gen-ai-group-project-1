  """Build a persistent semantic index from the Markdown knowledge base.

This module is intentionally limited to ingestion. It loads source documents,
splits them into chunks, creates semantic embeddings, and writes those vectors
plus their aligned document metadata to disk. Querying the index belongs to the
retrieval pipeline and is not implemented here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Protocol, Sequence

import faiss
from langchain_text_splitters import MarkdownTextSplitter
import numpy as np


DEFAULT_KNOWLEDGE_BASE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "knowledge_base"
)
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "faiss_store"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

INDEX_FILENAME = "index.faiss"
CHUNKS_FILENAME = "chunks.json"
MANIFEST_FILENAME = "manifest.json"
SCHEMA_VERSION = 1

CATEGORY_PATTERN = re.compile(
    r"^##[ \t]+Ticket Category[ \t]*\r?\n+([^\r\n]+)", re.MULTILINE
)


class EmbeddingModel(Protocol):
    """Minimal SentenceTransformer-compatible interface used by ingestion."""

    def encode(self, sentences: list[str], **kwargs: object) -> object:
        """Return one numeric embedding per input sentence."""


@dataclass(frozen=True)
class SourceDocument:
    """One Markdown source document and its required metadata."""

    source: str
    category: str
    content: str


@dataclass(frozen=True)
class DocumentChunk:
    """One chunk ready for embedding and persistence."""

    chunk_id: str
    text: str
    source: str
    category: str
    chunk_index: int


@dataclass(frozen=True)
class IngestionResult:
    """Summary of an ingestion run and its generated artifacts."""

    output_directory: Path
    document_count: int
    chunk_count: int
    embedding_dimension: int


def _normalize_category(category: str) -> str:
    """Match the category format already returned by the running application."""
    return re.sub(r"[^a-z0-9]+", "_", category.casefold()).strip("_")


def _extract_category(content: str, source: str) -> str:
    """Read and normalize the required Ticket Category Markdown section."""
    match = CATEGORY_PATTERN.search(content)
    if match is None:
        raise ValueError(
            f"Knowledge-base document is missing a Ticket Category section: {source}"
        )

    category = _normalize_category(match.group(1).strip())
    if not category:
        raise ValueError(f"Knowledge-base document has an empty category: {source}")
    return category


def load_markdown_documents(
    directory: Path = DEFAULT_KNOWLEDGE_BASE_DIRECTORY,
) -> list[SourceDocument]:
    """Load every top-level Markdown file in a knowledge-base directory."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Knowledge-base directory does not exist: {directory}")

    paths = sorted(directory.glob("*.md"))
    if not paths:
        raise ValueError(f"No Markdown documents found in: {directory}")

    documents: list[SourceDocument] = []
    for path in paths:
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            raise ValueError(f"Knowledge-base document is empty: {path.name}")
        documents.append(
            SourceDocument(
                source=path.name,
                category=_extract_category(content, path.name),
                content=content,
            )
        )
    return documents


def split_documents(
    documents: Sequence[SourceDocument],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Split Markdown on structural boundaries and copy metadata to every chunk."""
    if chunk_size < 1:
        raise ValueError("Chunk size must be at least one character.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be non-negative and smaller than chunk size."
        )

    splitter = MarkdownTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: list[DocumentChunk] = []
    for document in documents:
        split_texts = splitter.split_text(document.content)
        for chunk_index, text in enumerate(split_texts):
            clean_text = text.strip()
            if not clean_text:
                continue
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document.source}:chunk:{chunk_index:04d}",
                    text=clean_text,
                    source=document.source,
                    category=document.category,
                    chunk_index=chunk_index,
                )
            )

    if not chunks:
        raise ValueError("Document splitting did not produce any chunks.")
    return chunks


def create_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> EmbeddingModel:
    """Load the local SentenceTransformer used for semantic embeddings."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_chunks(
    chunks: Sequence[DocumentChunk], embedding_model: EmbeddingModel
) -> np.ndarray:
    """Generate normalized float32 embeddings for all chunks in one batch."""
    if not chunks:
        raise ValueError("At least one chunk is required for embedding.")

    encoded = embedding_model.encode(
        [chunk.text for chunk in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embeddings = np.ascontiguousarray(encoded, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("Embedding model must return a two-dimensional matrix.")
    if embeddings.shape[0] != len(chunks):
        raise ValueError("Embedding count does not match the number of chunks.")
    if embeddings.shape[1] < 1:
        raise ValueError("Embedding vectors must contain at least one dimension.")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding vectors must contain only finite values.")

    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(norms == 0):
        raise ValueError("Embedding vectors must not be zero vectors.")

    # Normalize here as well as requesting normalized model output so injected
    # implementations cannot accidentally violate the cosine-similarity contract.
    faiss.normalize_L2(embeddings)
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Create a flat inner-product index over normalized vectors."""
    if embeddings.ndim != 2 or embeddings.shape[0] < 1:
        raise ValueError("A non-empty embedding matrix is required.")

    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(embeddings)
    return index


def _write_json(path: Path, payload: object) -> None:
    """Write JSON atomically so readers never observe a partial metadata file."""
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def persist_faiss_store(
    index: faiss.Index,
    chunks: Sequence[DocumentChunk],
    output_directory: Path,
    *,
    model_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Persist the index and its position-aligned chunk metadata contract."""
    if index.ntotal != len(chunks):
        raise ValueError("FAISS vector count does not match the chunk count.")

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    index_path = output_directory / INDEX_FILENAME
    temporary_index_path = index_path.with_suffix(f"{index_path.suffix}.tmp")
    faiss.write_index(index, str(temporary_index_path))
    temporary_index_path.replace(index_path)

    chunk_records = [
        {
            "faiss_position": position,
            "id": chunk.chunk_id,
            "text": chunk.text,
            "metadata": {
                "source": chunk.source,
                "category": chunk.category,
                "chunk_index": chunk.chunk_index,
            },
        }
        for position, chunk in enumerate(chunks)
    ]
    _write_json(
        output_directory / CHUNKS_FILENAME,
        {"schema_version": SCHEMA_VERSION, "chunks": chunk_records},
    )

    _write_json(
        output_directory / MANIFEST_FILENAME,
        {
            "schema_version": SCHEMA_VERSION,
            "index": {
                "file": INDEX_FILENAME,
                "type": "IndexFlatIP",
                "metric": "cosine_via_normalized_inner_product",
                "vector_count": int(index.ntotal),
                "dimension": int(index.d),
            },
            "chunks_file": CHUNKS_FILENAME,
            "embedding": {
                "model": model_name,
                "normalize_embeddings": True,
            },
            "chunking": {
                "splitter": "MarkdownTextSplitter",
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            },
        },
    )


def ingest_knowledge_base(
    knowledge_base_directory: Path = DEFAULT_KNOWLEDGE_BASE_DIRECTORY,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embedding_model: EmbeddingModel | None = None,
) -> IngestionResult:
    """Run ingestion steps 1-4 and return a summary of the generated store."""
    documents = load_markdown_documents(knowledge_base_directory)
    chunks = split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    model = embedding_model or create_embedding_model(model_name)
    embeddings = embed_chunks(chunks, model)
    index = build_faiss_index(embeddings)
    persist_faiss_store(
        index,
        chunks,
        output_directory,
        model_name=model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return IngestionResult(
        output_directory=Path(output_directory),
        document_count=len(documents),
        chunk_count=len(chunks),
        embedding_dimension=int(embeddings.shape[1]),
    )


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a persistent FAISS index from Markdown support documents."
    )
    parser.add_argument(
        "--knowledge-base",
        type=Path,
        default=DEFAULT_KNOWLEDGE_BASE_DIRECTORY,
        help="Directory containing Markdown knowledge-base files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory where FAISS and metadata artifacts are written.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run ingestion from the command line."""
    options = _parse_arguments(arguments)
    result = ingest_knowledge_base(
        knowledge_base_directory=options.knowledge_base,
        output_directory=options.output,
        model_name=options.model,
        chunk_size=options.chunk_size,
        chunk_overlap=options.chunk_overlap,
    )
    print(f"Loaded {result.document_count} Markdown documents.")
    print(
        f"Stored {result.chunk_count} chunks as "
        f"{result.embedding_dimension}-dimensional embeddings."
    )
    print(f"FAISS store: {result.output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
