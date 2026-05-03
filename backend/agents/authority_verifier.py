from __future__ import annotations

from schemas import Citation, CitationVerification

from .base import Agent


class AuthorityVerifierAgent(Agent[list[CitationVerification]]):
    name = "AuthorityVerifierAgent"
    prompt = "Record that legal authorities cannot be source-verified unless authority text is provided in the case file. Do not use model memory or internet search."

    def run(self, citations: list[Citation]) -> list[CitationVerification]:
        return [
            CitationVerification(
                citation_id=citation.id,
                status="could_not_verify",
                issue="Authority text not provided",
                source_basis=(
                    "The supplied case file does not include primary authority text for this citation. "
                    "This pipeline intentionally does not use model memory or open-internet search to verify legal holdings."
                ),
                confidence=0.2,
                confidence_label="low",
                reasoning=(
                    "The citation was extracted from the motion, but the cited authority itself is not among the provided "
                    "documents, so support for the stated proposition cannot be verified from the available record."
                ),
            )
            for citation in citations
        ]
