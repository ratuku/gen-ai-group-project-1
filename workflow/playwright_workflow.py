import re
from pathlib import Path

from playwright.async_api import async_playwright


APP_URL = (
    Path(__file__).resolve().parents[1]
    / "mock_support_app"
    / "index.html"
).as_uri()


async def submit_ticket(
    issue: str,
    result: dict[str, str],
    *,
    headless: bool = False,
    slow_mo: int = 1500,
    confirmation_hold_ms: int = 5000,
) -> str:
    """Submit the issue and Specialist result, then return the verified ticket ID."""
    category = result["category"]
    resolution = result["resolution"]

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=headless,
            slow_mo=slow_mo
        )

        try:
            page = await browser.new_page()
            await page.goto(APP_URL)

            await page.locator("#issue").fill(issue)
            await page.locator("#category").select_option(category)
            await page.locator("#resolution").fill(resolution)
            await page.locator("#submit-ticket").click()

            await page.locator("#confirmation").wait_for(state="visible")

            shown_category = await page.locator("#ticket-category").inner_text()
            shown_resolution = await page.locator("#ticket-resolution").inner_text()
            ticket_id = (await page.locator("#ticket-id").inner_text()).strip()

            if shown_category != "Account Access":
                raise RuntimeError(f"Wrong category shown: {shown_category}")
            if shown_resolution != resolution:
                raise RuntimeError("Resolution shown does not match the Specialist result")
            if not re.fullmatch(r"\d{5}", ticket_id):
                raise RuntimeError(f"Invalid ticket ID: {ticket_id}")

            if confirmation_hold_ms:
                await page.wait_for_timeout(confirmation_hold_ms)
            return ticket_id
        finally:
            await browser.close()