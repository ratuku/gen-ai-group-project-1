import httpx
import asyncio
import json

# A2A Protocol
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import new_text_message
from a2a.types import Role, SendMessageRequest

from requester.inputs import InvalidUserRequest, receive_user_request
from workflow.playwright_workflow import submit_ticket

A2A_TIMEOUT = 10.0  # Timeout threshold in seconds


async def main():
    """Send a user issue to the Specialist and submit its result in the app."""
    try:
        request_plan = receive_user_request()
    except InvalidUserRequest as error:
        print(f"Input Error: {error}")
        return

    issue = request_plan.issue

    # Connecting to Specialist server w/ timeout and error handling
    try:
        async with httpx.AsyncClient(timeout=A2A_TIMEOUT) as http_client:
            resolver = A2ACardResolver(
                httpx_client=http_client,
                base_url="http://127.0.0.1:9999"
            )
            card = await resolver.get_agent_card()

        client = await create_client(
            agent=card,
            client_config=ClientConfig(streaming=False)
        )
    except (httpx.ConnectError, httpx.TimeoutException):
        print("\n" + "=" * 50)
        print("A2A Error: Unable to reach Specialist Agent at http://127.0.0.1:9999")
        print("Reason: Server is offline or timed out.")
        print("=" * 50 + "\n")
        return
    except Exception as err:
        print(f"Error initializing A2A Client: {err}")
        return

    # Execute A2A Message exchange
    try:
        request = SendMessageRequest(
            message=new_text_message(issue, role=Role.ROLE_USER)
        )
        async for response in client.send_message(request):
            if response.HasField("message"):
                result = json.loads(response.message.parts[0].text)
                print(f"Specialist returned: {result}")

                if not isinstance(result, dict) or "category" not in result or "resolution" not in result:
                    print("Error: Specialist returned incomplete or malformed JSON data.")
                    return

                ticket_id = await submit_ticket(issue, result)
                print(f"Ticket submitted and verified successfully! ID: {ticket_id}")

    except asyncio.TimeoutError:
        print("A2A Error: Specialist Agent timed out during execution.")
    except Exception as error:
        print(f"Workflow Execution Error: {error}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
