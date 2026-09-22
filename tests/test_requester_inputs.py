"""Tests for Requester Step 1 user input and request planning."""

import unittest

from requester.inputs import (
    MAX_ISSUE_LENGTH,
    InvalidUserRequest,
    create_request_plan,
    receive_user_request,
    validate_user_request,
)


class RequesterInputTests(unittest.TestCase):
    """Verify input becomes a safe, explicit plan for the Specialist."""

    def test_receive_user_request_prompts_and_builds_plan(self):
        prompts = []

        def reader(prompt):
            prompts.append(prompt)
            return "  I cannot   access\nmy account.  "

        plan = receive_user_request(reader)

        self.assertEqual(prompts, ["Describe your issue: "])
        self.assertEqual(plan.issue, "I cannot access my account.")
        self.assertEqual(plan.required_fields, ("category", "resolution"))

    def test_plan_builds_specialist_task_payload(self):
        plan = create_request_plan("My password reset link expired.")

        self.assertEqual(
            plan.to_task_payload(),
            {"issue": "My password reset link expired."},
        )

    def test_blank_input_is_rejected(self):
        with self.assertRaisesRegex(InvalidUserRequest, "Please enter an issue"):
            validate_user_request("  \n\t")

    def test_non_text_input_is_rejected(self):
        with self.assertRaisesRegex(InvalidUserRequest, "must be text"):
            validate_user_request(None)

    def test_overlong_input_is_rejected(self):
        with self.assertRaisesRegex(InvalidUserRequest, "characters or fewer"):
            validate_user_request("x" * (MAX_ISSUE_LENGTH + 1))


if __name__ == "__main__":
    unittest.main()
