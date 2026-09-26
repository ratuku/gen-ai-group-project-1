"""Tests for Markdown chunking, semantic embedding, and FAISS persistence."""

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import unittest
from uuid import uuid4

import faiss
import numpy as np

from rag.ingest_documents import (
    CHUNKS_FILENAME,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MODEL_NAME,
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    ingest_knowledge_base,
    load_markdown_documents,
    split_documents,
)


class FakeEmbeddingModel:
    """Return deterministic nonzero vectors without downloading a model."""

    def __init__(self) -> None:
        self.received_texts: list[str] = []
        self.options: dict[str, object] = {}

    def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray:
        self.received_texts = list(sentences)
        self.options = kwargs
        return np.asarray(
            [
                [
                    float(index + 1),
                    float(len(text)),
                    float(text.count("#") + 1),
                    float(len(text.split()) + 1),
                ]
                for index, text in enumerate(sentences)
            ],
            dtype=np.float32,
        )


@contextmanager
def temporary_workspace_directory():
    """Create a writable temporary directory beside this test module."""
    path = Path(__file__).resolve().parent / f".tmp_rag_ingestion_{uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


class RagIngestionTests(unittest.TestCase):
    """Verify ingestion output without exercising ticket #13 retrieval."""

    def test_loads_all_markdown_sources_with_normalized_categories(self):
        documents = load_markdown_documents()

        self.assertEqual(len(documents), 7)
        categories = {document.source: document.category for document in documents}
        self.assertEqual(categories["password_reset.md"], "account_access")
        self.assertEqual(categories["account_access.md"], "account_access")
        self.assertEqual(categories["email.md"], "email")

    def test_splits_documents_and_preserves_metadata_on_every_chunk(self):
        documents = load_markdown_documents()
        chunks = split_documents(documents)

        self.assertGreater(len(chunks), len(documents))
        source_metadata = {
            document.source: document.category for document in documents
        }
        self.assertEqual({chunk.source for chunk in chunks}, set(source_metadata))
        self.assertEqual(len({chunk.chunk_id for chunk in chunks}), len(chunks))

        for source, category in source_metadata.items():
            source_chunks = [chunk for chunk in chunks if chunk.source == source]
            self.assertEqual(
                [chunk.chunk_index for chunk in source_chunks],
                list(range(len(source_chunks))),
            )
            for chunk in source_chunks:
                self.assertTrue(chunk.text.strip())
                self.assertLessEqual(len(chunk.text), DEFAULT_CHUNK_SIZE)
                self.assertEqual(chunk.category, category)

    def test_persists_position_aligned_faiss_index_and_chunk_metadata(self):
        fake_model = FakeEmbeddingModel()
        with temporary_workspace_directory() as temporary_directory:
            output_directory = temporary_directory / "store"
            result = ingest_knowledge_base(
                output_directory=output_directory,
                embedding_model=fake_model,
            )

            index = faiss.read_index(str(output_directory / INDEX_FILENAME))
            chunks_payload = json.loads(
                (output_directory / CHUNKS_FILENAME).read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (output_directory / MANIFEST_FILENAME).read_text(encoding="utf-8")
            )

        chunk_records = chunks_payload["chunks"]
        self.assertEqual(result.document_count, 7)
        self.assertEqual(result.chunk_count, len(chunk_records))
        self.assertEqual(result.embedding_dimension, 4)
        self.assertEqual(index.ntotal, len(chunk_records))
        self.assertEqual(index.d, 4)
        self.assertEqual(
            [record["faiss_position"] for record in chunk_records],
            list(range(len(chunk_records))),
        )
        self.assertTrue(all(record["text"] for record in chunk_records))
        self.assertTrue(
            all(
                {"source", "category", "chunk_index"}
                <= record["metadata"].keys()
                for record in chunk_records
            )
        )
        self.assertEqual(fake_model.received_texts, [r["text"] for r in chunk_records])
        self.assertTrue(fake_model.options["normalize_embeddings"])
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["embedding"]["model"], DEFAULT_MODEL_NAME)
        self.assertEqual(manifest["index"]["vector_count"], len(chunk_records))
        self.assertEqual(
            manifest["index"]["metric"],
            "cosine_via_normalized_inner_product",
        )
        self.assertEqual(manifest["chunking"]["chunk_size"], DEFAULT_CHUNK_SIZE)
        self.assertEqual(
            manifest["chunking"]["chunk_overlap"], DEFAULT_CHUNK_OVERLAP
        )

    def test_rebuild_replaces_the_store_instead_of_appending_duplicates(self):
        with temporary_workspace_directory() as temporary_directory:
            output_directory = temporary_directory / "store"
            first = ingest_knowledge_base(
                output_directory=output_directory,
                embedding_model=FakeEmbeddingModel(),
            )
            second = ingest_knowledge_base(
                output_directory=output_directory,
                embedding_model=FakeEmbeddingModel(),
            )
            index = faiss.read_index(str(output_directory / INDEX_FILENAME))
            payload = json.loads(
                (output_directory / CHUNKS_FILENAME).read_text(encoding="utf-8")
            )

        self.assertEqual(second.chunk_count, first.chunk_count)
        self.assertEqual(index.ntotal, first.chunk_count)
        self.assertEqual(len(payload["chunks"]), first.chunk_count)

    def test_rejects_source_without_ticket_category(self):
        with temporary_workspace_directory() as temporary_directory:
            knowledge_base = temporary_directory
            (knowledge_base / "missing_category.md").write_text(
                "# Example\n\nContent without the required metadata.\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Ticket Category"):
                load_markdown_documents(knowledge_base)


if __name__ == "__main__":
    unittest.main()
