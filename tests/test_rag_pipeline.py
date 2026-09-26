"""Tests for RAG steps 5-7: retrieval, prompting, and grounded output."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rag.ingest_documents import (
    DocumentChunk,
    build_faiss_index,
    persist_faiss_store,
)
from rag.pipeline import (
    FaissRetriever,
    InsufficientContextError,
    InvalidLlmResponse,
    RagPipeline,
    RetrievedChunk,
    parse_generated_result,
)


class FakeEmbeddingModel:
    def encode(self, sentences, **kwargs):
        vectors = []
        for text in sentences:
            vectors.append([1.0, 0.0] if "password" in text.lower() else [0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def retrieve(self, issue, *, top_k):
        self.calls.append((issue, top_k))
        return self.chunks


class FakeGenerator:
    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return json.dumps(self.payload)


PASSWORD_CHUNK = RetrievedChunk(
    content="After identity verification, reset the user's password.",
    source="password_reset.md",
    category="account_access",
    chunk_index=0,
    similarity=0.98,
)


class RagPipelineTests(unittest.TestCase):
    def test_faiss_retriever_returns_most_similar_chunk(self):
        chunks = [
            DocumentChunk("password:0", "password reset", "password_reset.md", "account_access", 0),
            DocumentChunk("network:0", "network connection", "network.md", "network", 0),
        ]
        embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        index = build_faiss_index(embeddings)

        with tempfile.TemporaryDirectory() as directory:
            persist_faiss_store(
                index,
                chunks,
                Path(directory),
                model_name="fake-model",
                chunk_size=100,
                chunk_overlap=10,
            )
            retriever = FaissRetriever(
                Path(directory), embedding_model=FakeEmbeddingModel()
            )
            result = retriever.retrieve("I forgot my password", top_k=1)

        self.assertEqual(result[0].source, "password_reset.md")
        self.assertEqual(result[0].category, "account_access")
        self.assertAlmostEqual(result[0].similarity, 1.0)

    def test_pipeline_passes_issue_and_context_to_constrained_prompt(self):
        retriever = FakeRetriever([PASSWORD_CHUNK])
        generator = FakeGenerator(
            {
                "status": "resolved",
                "category": "account_access",
                "resolution": "Verify identity, reset the password, and confirm login.",
                "sources": ["password_reset.md"],
            }
        )
        pipeline = RagPipeline(retriever=retriever, generator=generator, top_k=3)

        result = pipeline.resolve("I forgot my password.")

        self.assertEqual(retriever.calls, [("I forgot my password.", 3)])
        self.assertEqual(result["category"], "account_access")
        self.assertEqual(result["sources"], ["password_reset.md"])
        prompt = generator.prompts[0]
        self.assertIn("I forgot my password.", prompt)
        self.assertIn("password_reset.md", prompt)
        self.assertIn("reset the user's password", prompt)
        self.assertIn("Constrained and Guided Generation", prompt)

    def test_rejects_source_that_was_not_retrieved(self):
        response = json.dumps(
            {
                "status": "resolved",
                "category": "account_access",
                "resolution": "Reset the password.",
                "sources": ["invented.md"],
            }
        )
        with self.assertRaisesRegex(InvalidLlmResponse, "not retrieved"):
            parse_generated_result(response, [PASSWORD_CHUNK])

    def test_insufficient_context_stops_the_pipeline(self):
        response = json.dumps(
            {
                "status": "insufficient_context",
                "category": "",
                "resolution": "The retrieved procedures do not cover this request.",
                "sources": ["password_reset.md"],
            }
        )
        with self.assertRaisesRegex(InsufficientContextError, "do not cover"):
            parse_generated_result(response, [PASSWORD_CHUNK])


if __name__ == "__main__":
    unittest.main()
