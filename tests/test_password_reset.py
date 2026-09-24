"""Regression tests for the password-reset support flow."""

import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx

from rag.pipeline import RagPipeline
from requester.inputs import create_request_plan
from requester.main import complete_request
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

    def test_rag_returns_usable_password_reset_result(self):
        """Case 1 gets the app category and a nonempty resolution."""
        result = RagPipeline().resolve(PASSWORD_CASE["request"])
        self.assertEqual(result["category"], "account_access")
        self.assertIsInstance(result["resolution"], str)
        self.assertTrue(result["resolution"].strip())

    async def test_specialist_a2a_endpoint_returns_rag_result(self):
        """An A2A SendMessage request reaches the Specialist and returns its result."""
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
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
        """Requester polls the task API and forwards its result to Playwright."""
        workflow = AsyncMock(return_value="12345")
        expected_result = RagPipeline().resolve(PASSWORD_CASE["request"])
        transport = httpx.ASGITransport(app=app)
        plan = create_request_plan(PASSWORD_CASE["request"])

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            ticket_id = await complete_request(
                plan,
                client,
                ticket_submitter=workflow,
            )

        self.assertEqual(ticket_id, "12345")
        workflow.assert_awaited_once_with(PASSWORD_CASE["request"], expected_result)

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
