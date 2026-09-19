import uvicorn
import json

from starlette.applications import Starlette
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
    routes= create_agent_card_routes(card) + create_jsonrpc_routes(handler, "/")
)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9999)