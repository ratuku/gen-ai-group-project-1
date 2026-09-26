# Group Project 1: 2 Agent AI System

## Setup

From the project root, create and activate a virtual environment:

```bash
python3 -m venv proj1-env
source proj1-env/bin/activate
python -m pip install -r requirements.txt
python -m playwright install chromium
cp .env.example .env
```

Playwright needs the Chromium browser installed separately from its Python package.
Set `GROQ_API_KEY` in `.env` before starting the Specialist.

## Run the current vertical slice

Start the Specialist in one terminal:

```bash
source proj1-env/bin/activate
python -m specialist.server
```

In a second terminal, run the Requester from the project root:

```bash
source proj1-env/bin/activate
python -m requester.main
```

Enter the password-reset issue from `test_cases.json` (case 1). The RAG pipeline
retrieves the most relevant procedure from the local knowledge base and returns
its ticket category and resolution. The Requester sends them to the mock support
app through Playwright and verifies the confirmation.

## RAG pipeline

Build the persistent semantic index before starting the Specialist:

```bash
python -m rag.ingest_documents
```

The pipeline loads the Markdown files in `knowledge_base/`, splits them on
Markdown-aware boundaries using an 800-character target and 100-character
overlap, and embeds the chunks with
`sentence-transformers/all-MiniLM-L6-v2`. The first run may download this model.
It writes three generated files under `rag/faiss_store/`:

- `index.faiss`: normalized 384-dimensional vectors in a FAISS `IndexFlatIP`.
- `chunks.json`: chunk text and metadata in FAISS insertion order.
- `manifest.json`: the schema version, model, dimensions, metric, and chunking
  settings needed to consume the index.

Each chunk preserves its source filename, normalized ticket category, and
per-source chunk index. A record has this shape:

```json
{
  "faiss_position": 0,
  "id": "account_access.md:chunk:0000",
  "text": "# Account Access ...",
  "metadata": {
    "source": "account_access.md",
    "category": "account_access",
    "chunk_index": 0
  }
}
```

The array position in `chunks.json` is the corresponding vector position in
`index.faiss`. The artifacts are generated and ignored by Git.

For each support issue, `rag/pipeline.py` completes RAG steps 5-7:

1. Embed the issue with the same sentence-transformer used during ingestion.
2. Retrieve the three most similar chunks from FAISS.
3. Render `rag/prompts/support_answer.poml` with the issue, retrieved chunks,
   source filenames, and allowed ticket categories.
4. Ask Groq for a schema-constrained JSON result using the Constrained and
   Guided Generation technique.
5. Validate the category, resolution, and cited sources before returning them
   to the Specialist.

The pipeline rejects invalid JSON, unsupported categories, invented source
filenames, and responses that report insufficient context. The public
`RagPipeline.resolve(issue)` method remains the boundary used by the Specialist.

## Requester input design

Requester Step 1 is implemented in `requester/inputs.py`. It prompts for a support
issue, normalizes repeated whitespace, rejects blank, non-text, or excessively long
input, and creates a `RequestPlan`. The plan records that the downstream Specialist
must provide both a support `category` and a `resolution`, and it can produce the
`{"issue": "..."}` payload used by the task API. Keeping input handling separate
from transport and browser automation makes the boundary between the Requester
steps explicit and independently testable.

## Specialist task API

With the Specialist server running on `http://127.0.0.1:9999`, submit an issue with `POST /tasks`:

```json
{"issue": "I forgot my password and cannot log in."}
```

The server immediately responds with HTTP `202` and a unique task ID:

```json
{"task_id": "<uuid>", "status": "submitted"}
```

Poll `GET /tasks/{task_id}` for the latest status. A task moves from `submitted` to `working`, then to `completed` with a `result` containing `category` and `resolution`, or to `failed` with an `error`. An unknown task ID returns HTTP `404`; an invalid submission returns HTTP `400`.

Tasks are stored in memory and disappear when the Specialist server restarts. The current Requester still uses the original A2A message endpoint; it has not yet been connected to this task API.

## Run tests

```bash
source proj1-env/bin/activate
python -m unittest discover -s tests -v
```

`unittest` is included with Python; no separate test package is needed. The suite
checks vector-store loading and retrieval, the Specialist task API, the
password-reset result, the original A2A endpoint, the Requester handoff, and a
real Playwright submission. Install Chromium using the setup command above
before running the browser test.

To run the ingestion and complete RAG pipeline tests without downloading an
embedding model or calling Groq:

```bash
python -m unittest tests.test_rag_ingestion tests.test_rag_pipeline -v
```

## Coding Standards

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python formatting and naming.
- Use `snake_case` for functions, methods, and variables; use `CapWords` for classes.
- Give each module one clear responsibility. Use classes when they manage state or implement an interface, and functions for straightforward operations.
- Keep the contracts between Requester, Specialist, RAG, and Playwright clear and testable.
- Add or update regression tests when behavior changes.
- Use SOLID principal.

## Git Workflow

- Work on a feature branch.
- Run the tests locally before opening a pull request.
- Open a pull request to merge into `main`.
- Get at least one review before merging, except when the team agrees an urgent fix needs a different process.

## Project requirements

See the project instructions on Canvas for the complete requirements.
