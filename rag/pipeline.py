class RagPipeline:
    """Provide a support category and resolution for an issue."""
    def resolve(self, issue: str) -> dict[str, str]:
        """Return a fixed result until knowledge base retrieval is implemented."""
        print(f"Rag pipeline received: {issue}")
    
        # Placeholder: use the issue for real retrieval later.
        return {
            "category": "account_access",
            "resolution": "Follow the password reset instructions"
        }