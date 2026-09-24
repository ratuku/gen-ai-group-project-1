import asyncio
from collections.abc import Awaitable, Callable

import httpx

from requester.inputs import InvalidUserRequest, RequestPlan, receive_user_request
from requester.task_client import SpecialistTaskError, request_specialist_result
from workflow.playwright_workflow import submit_ticket


SPECIALIST_URL = "http://127.0.0.1:9999"


async def complete_request(
    plan: RequestPlan,
    client: httpx.AsyncClient,
    *,
    ticket_submitter: Callable[[str, dict], Awaitable[str]] | None = None,
) -> str:
    """Complete Requester Steps 2 and 3 for one validated request plan."""
    result = await request_specialist_result(client, plan.to_task_payload())
    print(f"Specialist returned: {result}")
    submitter = submit_ticket if ticket_submitter is None else ticket_submitter
    return await submitter(plan.issue, result)


async def main() -> None:
    """Submit a user issue to the Specialist, then create the support ticket."""
    try:
        plan = receive_user_request()
    except InvalidUserRequest as error:
        print(error)
        return

    try:
        async with httpx.AsyncClient(base_url=SPECIALIST_URL) as client:
            ticket_id = await complete_request(plan, client)
    except (httpx.HTTPError, SpecialistTaskError) as error:
        print(f"Request could not be completed: {error}")
        return

    print(f"Ticket submitted and verified. ID: {ticket_id}")


if __name__ == "__main__":
    asyncio.run(main())
