"""Regression tests for the password-reset support flow."""

import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from a2a.client import A2ACardResolver, ClientConfig, create_client

from rag.pipeline import RagPipeline
from requester.inputs import create_request_plan
from requester.main import complete_request
from requester.task_client import SpecialistTaskFailed, request_specialist_result
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
        """Agent Card discovery, submission, and polling return the RAG result."""
        transport = httpx.ASGITransport(app=app)
        http_client = httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        )
        resolver = A2ACardResolver(
            httpx_client=http_client,
            base_url="http://testserver",
        )
        card = await resolver.get_agent_card()
        self.assertEqual(card.name, "Specialist")
        client = await create_client(
            agent=card,
            client_config=ClientConfig(
                streaming=False,
                polling=True,
                httpx_client=http_client,
            ),
        )
        try:
            result = await request_specialist_result(
                client,
                PASSWORD_CASE["request"],
                poll_interval=0,
            )
        finally:
            await client.close()

        self.assertEqual(result["category"], "account_access")
        self.assertIsInstance(result["resolution"], str)
        self.assertTrue(result["resolution"].strip())

    async def test_requester_passes_specialist_result_to_workflow(self):
        """Requester polls A2A and forwards the completed result to Playwright."""
        workflow = AsyncMock(return_value="12345")
        expected_result = RagPipeline().resolve(PASSWORD_CASE["request"])
        transport = httpx.ASGITransport(app=app)
        plan = create_request_plan(PASSWORD_CASE["request"])
        http_client = httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        )
        resolver = A2ACardResolver(
            httpx_client=http_client,
            base_url="http://testserver",
        )
        card = await resolver.get_agent_card()
        client = await create_client(
            agent=card,
            client_config=ClientConfig(
                streaming=False,
                polling=True,
                httpx_client=http_client,
            ),
        )
        try:
            ticket_id = await complete_request(
                plan,
                client,
                ticket_submitter=workflow,
            )
        finally:
            await client.close()

        self.assertEqual(ticket_id, "12345")
        workflow.assert_awaited_once_with(PASSWORD_CASE["request"], expected_result)

    async def test_specialist_failure_is_returned_through_a2a(self):
        """A RAG error becomes a failed A2A task visible to the Requester."""
        transport = httpx.ASGITransport(app=app)
        http_client = httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        )
        resolver = A2ACardResolver(
            httpx_client=http_client,
            base_url="http://testserver",
        )
        card = await resolver.get_agent_card()
        client = await create_client(
            agent=card,
            client_config=ClientConfig(
                streaming=False,
                polling=True,
                httpx_client=http_client,
            ),
        )
        try:
            with patch(
                "specialist.server.RagPipeline.resolve",
                side_effect=RuntimeError("RAG unavailable"),
            ):
                with self.assertRaisesRegex(SpecialistTaskFailed, "RAG unavailable"):
                    await request_specialist_result(
                        client,
                        PASSWORD_CASE["request"],
                        poll_interval=0,
                    )
        finally:
            await client.close()

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
