"""Client-side task submission and polling for the Requester Agent."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any


DEFAULT_POLL_INTERVAL = 0.25
DEFAULT_TIMEOUT = 15.0


class SpecialistTaskError(RuntimeError):
    """Base class for failures while communicating with the Specialist."""


class SpecialistTaskFailed(SpecialistTaskError):
    """Raised when the Specialist reports a failed task."""


class SpecialistTaskTimeout(SpecialistTaskError):
    """Raised when a Specialist task does not finish before the deadline."""


class InvalidSpecialistResponse(SpecialistTaskError):
    """Raised when the Specialist response does not satisfy the contract."""


def _response_json(response: Any) -> dict[str, Any]:
    """Validate an HTTP response and return its JSON object."""
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise InvalidSpecialistResponse("The Specialist response must be a JSON object.")
    return payload


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


async def submit_specialist_task(client: Any, payload: dict[str, str]) -> str:
    """Submit a task and return the Specialist-generated task ID."""
    response = await client.post("/tasks", json=payload)
    acknowledgement = _response_json(response)

    task_id = acknowledgement.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise InvalidSpecialistResponse("The Specialist acknowledgement omitted task_id.")
    if acknowledgement.get("status") != "submitted":
        raise InvalidSpecialistResponse(
            "The Specialist acknowledgement must have status 'submitted'."
        )
    return task_id


async def wait_for_specialist_result(
    client: Any,
    task_id: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> dict[str, Any]:
    """Poll a submitted task until it completes, fails, or times out."""
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    if poll_interval < 0:
        raise ValueError("poll_interval cannot be negative")

    deadline = clock() + timeout
    while True:
        payload = _response_json(await client.get(f"/tasks/{task_id}"))
        status = payload.get("status")

        if status == "completed":
            return validate_specialist_result(payload.get("result"))
        if status == "failed":
            message = payload.get("error")
            if not isinstance(message, str) or not message.strip():
                message = "The Specialist could not complete the task."
            raise SpecialistTaskFailed(message)
        if status not in {"submitted", "working"}:
            raise InvalidSpecialistResponse(
                f"The Specialist returned an unknown task status: {status!r}."
            )

        remaining = deadline - clock()
        if remaining <= 0:
            raise SpecialistTaskTimeout(
                f"Specialist task {task_id} did not finish within {timeout:g} seconds."
            )
        await sleep(min(poll_interval, remaining))


async def request_specialist_result(
    client: Any,
    payload: dict[str, str],
    *,
    timeout: float = DEFAULT_TIMEOUT,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
) -> dict[str, Any]:
    """Submit a Specialist task and wait for its validated completed result."""
    task_id = await submit_specialist_task(client, payload)
    return await wait_for_specialist_result(
        client,
        task_id,
        timeout=timeout,
        poll_interval=poll_interval,
    )
