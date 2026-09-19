import httpx
import asyncio
import json

# A2A Protocol
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import new_text_message
from a2a.types import Role, SendMessageRequest

from workflow.playwright_workflow import submit_ticket

async def main():
    """Send a user issue to the Specialist and submit its result in the app."""
    issue = input("Describe your issue: ").strip()

    if not issue:
        print("Please enter an issue.")
        return

    async with httpx.AsyncClient() as http_client:
        resolver = A2ACardResolver(
            httpx_client=http_client,
            base_url="http://127.0.0.1:9999"
        )
        card = await resolver.get_agent_card()

    client = await create_client(
        agent=card,
        client_config=ClientConfig(streaming=False)
    )

    try:
        request = SendMessageRequest(
            message=new_text_message(issue, role=Role.ROLE_USER)
        )

        async for response in client.send_message(request):
            if response.HasField("message"):
                result = json.loads(response.message.parts[0].text)
                print(f"Specialist returned: {result}")

                ticket_id = await submit_ticket(issue, result)
                print(f"Ticket submitted and verified. ID: {ticket_id}")


    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())