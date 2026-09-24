"""Tests for Requester task submission, polling, and response validation."""

import unittest

from requester.task_client import (
    InvalidSpecialistResponse,
    SpecialistTaskFailed,
    SpecialistTaskTimeout,
    submit_specialist_task,
    validate_specialist_result,
    wait_for_specialist_result,
)


class FakeResponse:
    """Small HTTP-response stand-in used by the task client tests."""

    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


class FakeClient:
    """Record submissions and return a configured sequence of task states."""

    def __init__(self, *, acknowledgement=None, states=None):
        self.acknowledgement = acknowledgement or {
            "task_id": "task-123",
            "status": "submitted",
        }
        self.states = list(states or [])
        self.posts = []
        self.gets = []

    async def post(self, path, json):
        self.posts.append((path, json))
        return FakeResponse(self.acknowledgement, status_code=202)

    async def get(self, path):
        self.gets.append(path)
        return FakeResponse(self.states.pop(0))


class RequesterTaskClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_submit_sends_issue_and_returns_task_id(self):
        client = FakeClient()

        task_id = await submit_specialist_task(client, {"issue": "Reset password"})

        self.assertEqual(task_id, "task-123")
        self.assertEqual(client.posts, [("/tasks", {"issue": "Reset password"})])

    async def test_polling_waits_for_completed_result(self):
        client = FakeClient(
            states=[
                {"task_id": "task-123", "status": "working"},
                {
                    "task_id": "task-123",
                    "status": "completed",
                    "result": {
                        "category": "account_access",
                        "resolution": "Reset the password.",
                        "sources": ["password_reset.md"],
                    },
                },
            ]
        )

        result = await wait_for_specialist_result(
            client,
            "task-123",
            poll_interval=0,
        )

        self.assertEqual(result["category"], "account_access")
        self.assertEqual(len(client.gets), 2)

    async def test_failed_task_stops_before_browser_handoff(self):
        client = FakeClient(
            states=[
                {
                    "task_id": "task-123",
                    "status": "failed",
                    "error": "RAG unavailable",
                }
            ]
        )

        with self.assertRaisesRegex(SpecialistTaskFailed, "RAG unavailable"):
            await wait_for_specialist_result(client, "task-123")

    async def test_polling_timeout_is_reported(self):
        client = FakeClient(states=[{"task_id": "task-123", "status": "working"}])
        times = iter([0.0, 1.0])

        with self.assertRaisesRegex(SpecialistTaskTimeout, "did not finish"):
            await wait_for_specialist_result(
                client,
                "task-123",
                timeout=0.5,
                clock=lambda: next(times),
            )

    def test_completed_result_requires_category_and_resolution(self):
        with self.assertRaisesRegex(InvalidSpecialistResponse, "resolution"):
            validate_specialist_result({"category": "account_access"})


if __name__ == "__main__":
    unittest.main()
