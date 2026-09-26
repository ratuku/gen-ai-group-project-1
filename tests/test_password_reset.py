"""Regression tests for the password-reset support flow."""

import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
from a2a.types import Message, Part, Role, StreamResponse

from specialist.server import app
from workflow.playwright_workflow import submit_ticket


CASES = json.loads((Path(__file__).resolve().parents[1] / "test_cases.json").read_text())
PASSWORD_CASE = next(case for case in CASES if case["id"] == 1)
SAMPLE_RESULT = {
    "category": "account_access",
    "resolution": "Use the approved password reset process.",
}


class PasswordResetSliceTests(unittest.IsolatedAsyncioTestCase):
    """Protect the password-reset contract across RAG, A2A, and the app."""

    async def test_specialist_a2a_endpoint_returns_rag_result(self):
        """An A2A SendMessage request reaches the Specialist and returns its result."""
        transport = httpx.ASGITransport(app=app)
        with patch("specialist.server.RagPipeline") as pipeline_class:
            pipeline_class.return_value.resolve.return_value = SAMPLE_RESULT
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                card = await client.get("/.well-known/agent-card.json")
                self.assertEqual(card.status_code, 200)
                self.assertEqual(card.json()["name"], "Specialist")

                response = await client.post(
                    "/",
                    headers={"A2A-Version": "1.0"},
                    json={
                        "jsonrpc": "2.0",
                        "id": "password-case-1",
                        "method": "SendMessage",
                        "params": {
                            "message": {
                                "messageId": str(uuid4()),
                                "role": "ROLE_USER",
                                "parts": [{"text": PASSWORD_CASE["request"]}],
                            }
                        },
                    },
                )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertNotIn("error", body)
        result = json.loads(body["result"]["message"]["parts"][0]["text"])
        self.assertEqual(result["category"], "account_access")
        self.assertIsInstance(result["resolution"], str)
        self.assertTrue(result["resolution"].strip())

    async def test_requester_passes_specialist_result_to_workflow(self):
        """Requester forwards the original issue and parsed A2A result to Playwright."""
        from requester import main as requester

        message = Message(
            message_id=str(uuid4()),
            role=Role.ROLE_AGENT,
            parts=[Part(text=json.dumps(SAMPLE_RESULT))],
        )
        response = StreamResponse(message=message)

        class FakeClient:
            """Return one Specialist message without opening a network connection."""

            def send_message(self, request):
                """Check the outgoing issue and yield the fixed Specialist response."""
                sent_issue = request.message.parts[0].text
                if sent_issue != PASSWORD_CASE["request"]:
                    raise AssertionError(f"Wrong issue sent: {sent_issue}")

                async def reply():
                    """Yield the A2A response in the SDK client shape."""
                    yield response

                return reply()

            async def close(self):
                """Match the SDK client's cleanup interface."""
                return None

        resolver = AsyncMock()
        resolver.get_agent_card.return_value = object()
        with (
            patch("builtins.input", return_value=PASSWORD_CASE["request"]),
            patch.object(requester, "A2ACardResolver", return_value=resolver),
            patch.object(requester, "create_client", new=AsyncMock(return_value=FakeClient())),
            patch.object(requester, "submit_ticket", new=AsyncMock(return_value="12345")) as workflow,
        ):
            await requester.main()

        workflow.assert_awaited_once_with(PASSWORD_CASE["request"], SAMPLE_RESULT)

    async def test_playwright_submits_and_verifies_ticket(self):
        """A real browser submits case 1 to the mock app and receives a ticket ID."""
        ticket_id = await submit_ticket(
            PASSWORD_CASE["request"],
            SAMPLE_RESULT,
            headless=True,
            slow_mo=0,
            confirmation_hold_ms=0,
        )
        self.assertRegex(ticket_id, r"^\d{5}$")


if __name__ == "__main__":
    unittest.main()
