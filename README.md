# Group Project 1: 2 Agent AI System

## Setup

From the project root, create and activate a virtual environment:

```bash
python3 -m venv proj1-env
source proj1-env/bin/activate
python -m pip install -r requirements.txt
python -m playwright install chromium
```

Playwright needs the Chromium browser installed separately from its Python package.

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

## RAG pipeline: in-memory vector store

RAG Step 1 is implemented in `rag/pipeline.py`. At startup, the pipeline loads
every Markdown file in `knowledge_base/` and stores each procedure as a
normalized vector in memory. Embeddings are produced with deterministic token
hashing, so retrieval does not require an external database, API key, network
connection, or model download.

For each support request, the pipeline:

1. Converts the request into a query vector.
2. Compares it with all stored knowledge-base vectors using cosine similarity.
3. Selects the most relevant procedure.
4. Returns its `category` and `resolution` to the Specialist.

Each knowledge-base document must contain `## Resolution` and
`## Ticket Category` sections. These sections keep the retrieval result aligned
with the existing Specialist and Playwright contract. Because the store is
in-memory, it is rebuilt whenever a new `RagPipeline` instance is created and is
not persisted between processes.

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

Tasks are stored in memory and disappear when the Specialist server restarts. The
Requester submits to this task API, polls while the status is `submitted` or
`working`, validates the completed `category` and `resolution`, and only then
starts the Playwright workflow. Failed tasks and tasks that exceed the Requester
timeout are reported without submitting an invalid support ticket.

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

To run only the RAG Step 1 tests:

```bash
python -m unittest tests.test_rag_vector_store -v
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
