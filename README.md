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

Enter the password-reset issue from `test_cases.json` (case 1). The RAG pipeline currently returns a fixed category and resolution. The Requester sends them to the mock support app through Playwright and verifies the confirmation.

## Run tests

```bash
source proj1-env/bin/activate
python -m unittest discover -s tests -v
```

`unittest` is included with Python; no separate test package is needed. The suite checks the password-reset result, the Specialist A2A endpoint, the Requester handoff, and a real Playwright submission. Install Chromium using the setup command above before running the browser test.

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
