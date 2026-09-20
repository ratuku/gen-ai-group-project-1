import asyncio
import json
import uvicorn
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from rag.pipeline import RagPipeline
from uuid import uuid4

# A2A protocol
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from a2a.types import Message, Part, Role


# This in-memory store is deliberately small for the course project.  A
# production service would persist tasks in a database or queue.
tasks: dict[str, dict[str, Any]] = {}


def task_response(task: dict[str, Any]) -> dict[str, Any]:
    """Return only the fields appropriate for the task's current state."""
    response = {"task_id": task["task_id"], "status": task["status"]}
    if task["status"] == "completed":
        response["result"] = task["result"]
    elif task["status"] == "failed":
        response["error"] = task["error"]
    return response


async def process_task(task_id: str) -> None:
    """Run RAG after acknowledgement, then update the task lifecycle."""
    task = tasks[task_id]
    task["status"] = "working"

    try:
        # RAG is synchronous today, so run it outside the event loop. This
        # lets the status endpoint remain responsive while work is running.
        result = await asyncio.to_thread(RagPipeline().resolve, task["issue"])
        task["result"] = result
        task["status"] = "completed"
    except Exception as error:
        task["error"] = f"Specialist processing failed: {error}"
        task["status"] = "failed"


async def submit_task(request: Request) -> JSONResponse:
    """Accept an issue and immediately acknowledge its asynchronous task."""
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "Request body must be valid JSON."}, status_code=400)

    issue = payload.get("issue") if isinstance(payload, dict) else None
    if not isinstance(issue, str) or not issue.strip():
        return JSONResponse({"error": "'issue' must be a non-empty string."}, status_code=400)

    task_id = str(uuid4())
    task = {
        "task_id": task_id,
        "issue": issue.strip(),
        "status": "submitted",
        "result": None,
        "error": None,
    }
    tasks[task_id] = task
    asyncio.create_task(process_task(task_id))
    return JSONResponse(task_response(task), status_code=202)


async def get_task(request: Request) -> JSONResponse:
    """Return a task's current status and its result once it has completed."""
    task = tasks.get(request.path_params["task_id"])
    if task is None:
        return JSONResponse({"error": "Task not found."}, status_code=404)
    return JSONResponse(task_response(task))


class SpecialistExecutor(AgentExecutor):
    """Handle A2A support requests using the RAG pipeline."""
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Resolve the incoming issue and send its result as an A2A message."""
        issue = context.message.parts[0].text
        print(f"Specialist received: {issue}")

        result = RagPipeline().resolve(issue)

        return await event_queue.enqueue_event(
            Message(
                message_id=str(uuid4()),
                role=Role.ROLE_AGENT,
                parts=[Part(text=json.dumps(result))]
            )
        )

    async def cancel(self, context, event_queue):
        """Report that task cancellation is not implemented for this demo."""
        return NotImplementedError("Cancellation is not used in this demo.")

card = AgentCard(
    name="Specialist",
    description="Temporary description for the first server check",
    version="0.1.0",
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    capabilities=AgentCapabilities(streaming=False),
    supported_interfaces=[
        AgentInterface(
            protocol_binding="JSONRPC",
            url="http://127.0.0.1:9999",
            protocol_version="1.0"
        )
    ],
    skills=[
        AgentSkill(
            id="support_issue",
            name="Support Issue",
            description="Temporary skill for the first server check.",
            tags=["support"],
        )
    ]
)

handler = DefaultRequestHandler(
    agent_executor = SpecialistExecutor(),
    task_store = InMemoryTaskStore(),
    agent_card = card 
)

app = Starlette(
    routes=[
        Route("/tasks", submit_task, methods=["POST"]),
        Route("/tasks/{task_id}", get_task, methods=["GET"]),
    ] + create_agent_card_routes(card) + create_jsonrpc_routes(handler, "/")
)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9999)
