"""Official A2A task submission and polling for the Requester Agent."""

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

from a2a.helpers import new_text_message
from a2a.types import GetTaskRequest, Role, SendMessageRequest, TaskState


DEFAULT_POLL_INTERVAL = 0.25
DEFAULT_TIMEOUT = 15.0


class SpecialistTaskError(RuntimeError):
    """Base class for failures while communicating with the Specialist."""


class SpecialistTaskFailed(SpecialistTaskError):
    """Raised when the Specialist reports a failed A2A task."""


class SpecialistTaskTimeout(SpecialistTaskError):
    """Raised when an A2A task does not finish before the deadline."""


class InvalidSpecialistResponse(SpecialistTaskError):
    """Raised when an A2A response does not satisfy the shared contract."""


def validate_specialist_result(result: object) -> dict[str, Any]:
    """Return a completed result containing usable category and resolution fields."""
    if not isinstance(result, dict):
        raise InvalidSpecialistResponse("The completed task did not include a result object.")

    for field in ("category", "resolution"):
        value = result.get(field)
        if not isinstance(value, str) or not value.strip():
            raise InvalidSpecialistResponse(
                f"The Specialist result requires a non-empty '{field}' field."
            )
    return result


def _message_text(message: Any) -> str:
    """Return the first text part from an A2A status message."""
    if message is None or not message.parts:
        raise InvalidSpecialistResponse("The Specialist task did not include a message.")
    text = message.parts[0].text
    if not text:
        raise InvalidSpecialistResponse("The Specialist task message was empty.")
    return text


def _completed_result(task: Any) -> dict[str, Any]:
    """Decode and validate the JSON result carried by a completed A2A task."""
    try:
        result = json.loads(_message_text(task.status.message))
    except json.JSONDecodeError as error:
        raise InvalidSpecialistResponse(
            "The Specialist result was not valid JSON."
        ) from error
    return validate_specialist_result(result)


async def submit_specialist_task(client: Any, issue: str) -> str:
    """Submit an issue through A2A and return its acknowledged task ID."""
    request = SendMessageRequest(
        message=new_text_message(issue, role=Role.ROLE_USER)
    )

    async for response in client.send_message(request):
        if not response.HasField("task"):
            raise InvalidSpecialistResponse(
                "The Specialist did not acknowledge the request as an A2A task."
            )
        task = response.task
        if not task.id:
            raise InvalidSpecialistResponse("The A2A acknowledgement omitted the task ID.")
        if task.status.state not in {
            TaskState.TASK_STATE_SUBMITTED,
            TaskState.TASK_STATE_WORKING,
            TaskState.TASK_STATE_COMPLETED,
        }:
            raise InvalidSpecialistResponse(
                "The A2A acknowledgement returned an invalid initial status."
            )
        return task.id

    raise InvalidSpecialistResponse("The Specialist returned no A2A acknowledgement.")


async def wait_for_specialist_result(
    client: Any,
    task_id: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> dict[str, Any]:
    """Poll an A2A task until it completes, fails, or times out."""
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    if poll_interval < 0:
        raise ValueError("poll_interval cannot be negative")

    deadline = clock() + timeout
    while True:
        task = await client.get_task(GetTaskRequest(id=task_id))
        state = task.status.state

        if state == TaskState.TASK_STATE_COMPLETED:
            return _completed_result(task)
        if state in {
            TaskState.TASK_STATE_FAILED,
            TaskState.TASK_STATE_REJECTED,
            TaskState.TASK_STATE_CANCELED,
        }:
            try:
                message = _message_text(task.status.message)
            except InvalidSpecialistResponse:
                message = "The Specialist could not complete the task."
            raise SpecialistTaskFailed(message)
        if state not in {
            TaskState.TASK_STATE_SUBMITTED,
            TaskState.TASK_STATE_WORKING,
        }:
            state_name = TaskState.Name(state)
            raise InvalidSpecialistResponse(
                f"The Specialist returned an unexpected A2A task state: {state_name}."
            )

        remaining = deadline - clock()
        if remaining <= 0:
            raise SpecialistTaskTimeout(
                f"Specialist task {task_id} did not finish within {timeout:g} seconds."
            )
        await sleep(min(poll_interval, remaining))


async def request_specialist_result(
    client: Any,
    issue: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
) -> dict[str, Any]:
    """Submit an A2A task and wait for its validated completed result."""
    task_id = await submit_specialist_task(client, issue)
    return await wait_for_specialist_result(
        client,
        task_id,
        timeout=timeout,
        poll_interval=poll_interval,
    )
