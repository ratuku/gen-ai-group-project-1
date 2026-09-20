"""Tests for the Specialist's task-based A2A-style HTTP interface."""

import asyncio
import unittest
from unittest.mock import patch

import httpx

from specialist.server import app, tasks


class A2ATaskApiTests(unittest.IsolatedAsyncioTestCase):
    """Ensure task acknowledgement, polling, and failures follow the contract."""

    def setUp(self):
        tasks.clear()

    async def submit(self, client, issue="I forgot my password."):
        response = await client.post("/tasks", json={"issue": issue})
        self.assertEqual(response.status_code, 202)
        return response.json()

    async def wait_for_terminal_status(self, client, task_id):
        for _ in range(40):
            response = await client.get(f"/tasks/{task_id}")
            body = response.json()
            if body["status"] in {"completed", "failed"}:
                return body
            await asyncio.sleep(0.01)
        self.fail("Task did not reach a terminal status.")

    async def test_submit_acknowledges_task_with_id_and_submitted_status(self):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            acknowledgement = await self.submit(client)

        self.assertIn("task_id", acknowledgement)
        self.assertEqual(acknowledgement["status"], "submitted")
        self.assertNotIn("result", acknowledgement)

    async def test_completed_task_returns_rag_result_when_polled(self):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            acknowledgement = await self.submit(client)
            result = await self.wait_for_terminal_status(client, acknowledgement["task_id"])

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["result"]["category"], "account_access")
        self.assertIn("resolution", result["result"])

    async def test_unknown_task_returns_not_found(self):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/tasks/not-a-real-task")

        self.assertEqual(response.status_code, 404)

    async def test_processing_error_becomes_failed_task(self):
        transport = httpx.ASGITransport(app=app)
        with patch("specialist.server.RagPipeline.resolve", side_effect=RuntimeError("RAG unavailable")):
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                acknowledgement = await self.submit(client)
                result = await self.wait_for_terminal_status(client, acknowledgement["task_id"])

        self.assertEqual(result["status"], "failed")
        self.assertIn("RAG unavailable", result["error"])

    async def test_invalid_issue_is_rejected(self):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post("/tasks", json={"issue": "  "})

        self.assertEqual(response.status_code, 400)
