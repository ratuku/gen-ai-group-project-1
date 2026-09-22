"""User-input and request-planning helpers for the Requester Agent."""

from dataclasses import dataclass
from typing import Callable


DEFAULT_PROMPT = "Describe your issue: "
MAX_ISSUE_LENGTH = 2_000
REQUIRED_SPECIALIST_FIELDS = ("category", "resolution")


class InvalidUserRequest(ValueError):
    """Raised when the Requester cannot create a task from user input."""


@dataclass(frozen=True)
class RequestPlan:
    """Validated input and the information required from the Specialist."""

    issue: str
    required_fields: tuple[str, ...] = REQUIRED_SPECIALIST_FIELDS

    def to_task_payload(self) -> dict[str, str]:
        """Return the task payload expected by the Specialist task API."""
        return {"issue": self.issue}


def validate_user_request(raw_issue: object) -> str:
    """Normalize a user issue and reject input that cannot form a useful task."""
    if not isinstance(raw_issue, str):
        raise InvalidUserRequest("The issue must be text.")

    issue = " ".join(raw_issue.split())
    if not issue:
        raise InvalidUserRequest("Please enter an issue.")
    if len(issue) > MAX_ISSUE_LENGTH:
        raise InvalidUserRequest(
            f"The issue must be {MAX_ISSUE_LENGTH} characters or fewer."
        )
    return issue


def create_request_plan(raw_issue: object) -> RequestPlan:
    """Determine the fields the Requester needs from the Specialist."""
    return RequestPlan(issue=validate_user_request(raw_issue))


def receive_user_request(
    input_reader: Callable[[str], str] | None = None,
    prompt: str = DEFAULT_PROMPT,
) -> RequestPlan:
    """Prompt once for an issue and return a validated Specialist request plan."""
    reader = input if input_reader is None else input_reader
    return create_request_plan(reader(prompt))
