"""Retrieve support context from FAISS and generate a grounded resolution."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Protocol, Sequence

import faiss
from groq import Groq
import numpy as np
from dotenv import load_dotenv

from rag.ingest_documents import (
    CHUNKS_FILENAME, DEFAULT_MODEL_NAME, DEFAULT_OUTPUT_DIRECTORY,
    INDEX_FILENAME, MANIFEST_FILENAME, EmbeddingModel, create_embedding_model,
)
from rag.prompt_renderer import render_support_prompt

DEFAULT_TOP_K = 3
DEFAULT_LLM_MODEL = "openai/gpt-oss-20b" # The model in env file takes priority of model defined here
ALLOWED_CATEGORIES = (
    "account_access", "hardware", "software", "network", "email", "security",
)


class RagPipelineError(RuntimeError):
    """Base error for retrieval or generation failures."""


class VectorStoreError(RagPipelineError):
    """Raised when the persisted FAISS store is unusable."""


class InvalidLlmResponse(RagPipelineError):
    """Raised when generation violates the Specialist contract."""


class InsufficientContextError(RagPipelineError):
    """Raised when retrieved knowledge cannot support a resolution."""


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str:
        """Return one JSON response for a rendered support prompt."""


@dataclass(frozen=True)
class RetrievedChunk:
    content: str
    source: str
    category: str
    chunk_index: int
    similarity: float

    def to_prompt_context(self) -> dict[str, object]:
        return {
            "content": self.content,
            "source": self.source,
            "category": self.category,
            "chunk_index": self.chunk_index,
            "similarity": round(self.similarity, 6),
        }


class FaissRetriever:
    """Load ingestion artifacts and retrieve semantically similar chunks."""

    def __init__(
        self,
        store_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
        *,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self.store_directory = Path(store_directory)
        self.index, self.chunk_records, self.model_name = self._load_store()
        self.embedding_model = embedding_model or create_embedding_model(self.model_name)

    def _load_store(self) -> tuple[faiss.Index, list[dict[str, Any]], str]:
        index_path = self.store_directory / INDEX_FILENAME
        chunks_path = self.store_directory / CHUNKS_FILENAME
        manifest_path = self.store_directory / MANIFEST_FILENAME
        missing = [p.name for p in (index_path, chunks_path, manifest_path) if not p.is_file()]
        if missing:
            raise VectorStoreError(
                f"RAG vector store is missing {', '.join(missing)}. "
                "Run `python -m rag.ingest_documents`."
            )
        try:
            index = faiss.read_index(str(index_path))
            chunks_payload = json.loads(chunks_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise VectorStoreError(f"Could not load the RAG vector store: {error}") from error

        records = chunks_payload.get("chunks")
        if not isinstance(records, list) or not records:
            raise VectorStoreError("chunks.json must contain a non-empty chunks array.")
        if index.ntotal != len(records):
            raise VectorStoreError("FAISS vector count does not match chunks.json.")
        if manifest.get("index", {}).get("dimension") != index.d:
            raise VectorStoreError("FAISS dimensions do not match manifest.json.")
        model_name = manifest.get("embedding", {}).get("model", DEFAULT_MODEL_NAME)
        if not isinstance(model_name, str) or not model_name.strip():
            raise VectorStoreError("manifest.json does not contain an embedding model.")
        return index, records, model_name

    def retrieve(self, issue: str, *, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
        if not isinstance(issue, str) or not issue.strip():
            raise ValueError("A non-empty support issue is required.")
        if top_k < 1:
            raise ValueError("top_k must be at least one.")
        encoded = self.embedding_model.encode(
            [issue.strip()], convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        query = np.ascontiguousarray(encoded, dtype=np.float32)
        if query.shape != (1, self.index.d) or not np.isfinite(query).all():
            raise VectorStoreError(
                f"Query embedding must have shape (1, {self.index.d}) and finite values."
            )
        if np.linalg.norm(query[0]) == 0:
            raise VectorStoreError("Query embedding must not be a zero vector.")
        faiss.normalize_L2(query)
        scores, positions = self.index.search(query, min(top_k, int(self.index.ntotal)))
        chunks: list[RetrievedChunk] = []
        for score, position in zip(scores[0], positions[0]):
            if position < 0:
                continue
            record = self.chunk_records[int(position)]
            metadata = record.get("metadata", {})
            chunks.append(RetrievedChunk(
                content=str(record.get("text", "")),
                source=str(metadata.get("source", "")),
                category=str(metadata.get("category", "")),
                chunk_index=int(metadata.get("chunk_index", 0)),
                similarity=float(score),
            ))
        if not chunks:
            raise InsufficientContextError("No knowledge-base chunks were retrieved.")
        return chunks


class GroqGenerator:
    """Generate schema-constrained support results with Groq."""

    def __init__(self, *, api_key: str | None = None, model: str | None = None,
                 client: Any | None = None) -> None:
        load_dotenv()
        resolved_key = api_key or os.getenv("GROQ_API_KEY")
        if client is None and not resolved_key:
            raise RagPipelineError("GROQ_API_KEY is required for RAG generation.")
        self.client = client or Groq(api_key=resolved_key)
        self.model = model or os.getenv("GROQ_MODEL", DEFAULT_LLM_MODEL)

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "support_resolution", "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string", "enum": ["resolved", "insufficient_context"]},
                            "category": {"type": "string"},
                            "resolution": {"type": "string"},
                            "sources": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["status", "category", "resolution", "sources"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise InvalidLlmResponse("The LLM returned an empty response.")
        return content


def parse_generated_result(response_text: str,
                           chunks: Sequence[RetrievedChunk]) -> dict[str, object]:
    """Validate generated JSON and its citations."""
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise InvalidLlmResponse("The LLM response was not valid JSON.") from error
    if not isinstance(payload, dict):
        raise InvalidLlmResponse("The LLM response must be a JSON object.")
    if payload.get("status") == "insufficient_context":
        raise InsufficientContextError(
            payload.get("resolution") or "The retrieved context was insufficient."
        )
    if payload.get("status") != "resolved":
        raise InvalidLlmResponse("The LLM returned an unknown status.")

    category, resolution, sources = (
        payload.get("category"), payload.get("resolution"), payload.get("sources")
    )
    if category not in ALLOWED_CATEGORIES:
        raise InvalidLlmResponse(f"The LLM returned an unsupported category: {category!r}.")
    if not isinstance(resolution, str) or not resolution.strip():
        raise InvalidLlmResponse("The LLM response requires a non-empty resolution.")
    if not isinstance(sources, list) or not sources or not all(
        isinstance(source, str) and source.strip() for source in sources
    ):
        raise InvalidLlmResponse("The LLM response requires at least one source filename.")
    if not set(sources).issubset({chunk.source for chunk in chunks}):
        raise InvalidLlmResponse("The LLM cited a source that was not retrieved.")
    return {
        "category": category,
        "resolution": resolution.strip(),
        "sources": list(dict.fromkeys(sources)),
    }


class RagPipeline:
    """Coordinate semantic retrieval and constrained grounded generation."""

    def __init__(self, retriever: FaissRetriever | None = None,
                 generator: TextGenerator | None = None, *, top_k: int = DEFAULT_TOP_K) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least one.")
        self.retriever = retriever or FaissRetriever()
        self.generator = generator or GroqGenerator()
        self.top_k = top_k

    def resolve(self, issue: str) -> dict[str, object]:
        chunks = self.retriever.retrieve(issue, top_k=self.top_k)
        prompt = render_support_prompt(
            issue=issue,
            context_chunks=[chunk.to_prompt_context() for chunk in chunks],
            allowed_categories=ALLOWED_CATEGORIES,
        )
        return parse_generated_result(self.generator.generate(prompt), chunks)
