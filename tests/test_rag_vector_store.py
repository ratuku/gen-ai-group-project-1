"""Tests for the in-memory knowledge-base vector store."""

import unittest

from rag.pipeline import InMemoryVectorStore, RagPipeline


class InMemoryVectorStoreTests(unittest.TestCase):
    """Ensure support procedures are loaded and retrieved locally."""

    def setUp(self):
        self.store = InMemoryVectorStore.from_knowledge_base()

    def test_loads_all_knowledge_base_procedures(self):
        self.assertEqual(len(self.store.documents), 7)
        self.assertIn("password_reset.md", {document.source for document in self.store.documents})

    def test_retrieves_email_procedure_for_email_issue(self):
        document = self.store.similarity_search(
            "My Outlook messages are not synchronizing or sending."
        )[0]
        self.assertEqual(document.source, "email.md")
        self.assertEqual(document.category, "email")

    def test_retrieves_password_reset_procedure_for_forgotten_password(self):
        document = self.store.similarity_search(
            "I forgot my password and cannot log in to my account."
        )[0]
        self.assertEqual(document.source, "password_reset.md")
        self.assertEqual(document.category, "account_access")

    def test_pipeline_returns_existing_specialist_contract(self):
        result = RagPipeline(self.store).resolve("My laptop will not power on.")
        self.assertEqual(result["category"], "hardware")
        self.assertTrue(result["resolution"])

    def test_rejects_empty_search_query(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            self.store.similarity_search("   ")


if __name__ == "__main__":
    unittest.main()
