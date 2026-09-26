"""Tests for Requester A2A task submission, polling, and validation."""

import json
import unittest

from a2a.types import Message, Part, Role, StreamResponse, Task, TaskState, TaskStatus

from requester.task_client import (
    InvalidSpecialistResponse,
    SpecialistTaskFailed,
    SpecialistTaskTimeout,
    submit_specialist_task,
    validate_specialist_result,
    wait_for_specialist_result,
)


def make_task(state, *, text="", task_id="task-123"):
    """Build an A2A task in the requested state for a client test."""
    status = TaskStatus(state=state)
    if text:
        status.message.CopyFrom(
            Message(
                message_id="message-123",
                role=Role.ROLE_AGENT,
                parts=[Part(text=text)],
            )
        )
    return Task(id=task_id, context_id="context-123", status=status)


class FakeA2AClient:
    """Record A2A calls and return configured task states."""

    def __init__(self, *, acknowledgement=None, states=None):
        self.acknowledgement = acknowledgement or make_task(
            TaskState.TASK_STATE_SUBMITTED
        )
        self.states = list(states or [])
        self.sent_requests = []
        self.get_requests = []

    async def send_message(self, request):
        self.sent_requests.append(request)
        yield StreamResponse(task=self.acknowledgement)

    async def get_task(self, request):
        self.get_requests.append(request)
        return self.states.pop(0)


class RequesterTaskClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_submit_uses_a2a_message_and_returns_task_id(self):
        client = FakeA2AClient()

        task_id = await submit_specialist_task(client, "Reset password")

        self.assertEqual(task_id, "task-123")
        self.assertEqual(client.sent_requests[0].message.parts[0].text, "Reset password")

    async def test_polling_waits_for_completed_a2a_task(self):
        result = {
            "category": "account_access",
            "resolution": "Reset the password.",
            "sources": ["password_reset.md"],
        }
        client = FakeA2AClient(
            states=[
                make_task(TaskState.TASK_STATE_WORKING),
                make_task(TaskState.TASK_STATE_COMPLETED, text=json.dumps(result)),
            ]
        )

        completed = await wait_for_specialist_result(
            client,
            "task-123",
            poll_interval=0,
        )

        self.assertEqual(completed, result)
        self.assertEqual(len(client.get_requests), 2)
        self.assertEqual(client.get_requests[0].id, "task-123")

    async def test_failed_a2a_task_stops_before_browser_handoff(self):
        client = FakeA2AClient(
            states=[
                make_task(
                    TaskState.TASK_STATE_FAILED,
                    text="RAG unavailable",
                )
            ]
        )

        with self.assertRaisesRegex(SpecialistTaskFailed, "RAG unavailable"):
            await wait_for_specialist_result(client, "task-123")

    async def test_polling_timeout_is_reported(self):
        client = FakeA2AClient(
            states=[make_task(TaskState.TASK_STATE_WORKING)]
        )
        times = iter([0.0, 1.0])

        with self.assertRaisesRegex(SpecialistTaskTimeout, "did not finish"):
            await wait_for_specialist_result(
                client,
                "task-123",
                timeout=0.5,
                clock=lambda: next(times),
            )

    async def test_direct_message_is_not_accepted_as_task_acknowledgement(self):
        client = FakeA2AClient()

        async def send_message(_request):
            yield StreamResponse(
                message=Message(
                    message_id="message-123",
                    role=Role.ROLE_AGENT,
                    parts=[Part(text="not a task")],
                )
            )

        client.send_message = send_message
        with self.assertRaisesRegex(InvalidSpecialistResponse, "acknowledge"):
            await submit_specialist_task(client, "Reset password")

    def test_completed_result_requires_category_and_resolution(self):
        with self.assertRaisesRegex(InvalidSpecialistResponse, "resolution"):
            validate_specialist_result({"category": "account_access"})


if __name__ == "__main__":
    unittest.main()
