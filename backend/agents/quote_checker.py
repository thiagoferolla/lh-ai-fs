from __future__ import annotations

from schemas import Citation, QuoteCheck

from .base import Agent


class QuoteCheckerAgent(Agent[list[QuoteCheck]]):
    name = "QuoteCheckerAgent"
    prompt = "Record direct quotes that require primary authority text. Do not classify quote accuracy from model memory or internet search."

    def run(self, citations: list[Citation]) -> list[QuoteCheck]:
        quoted_citations = [citation for citation in citations if citation.has_direct_quote and citation.direct_quote]
        return [
            QuoteCheck(
                citation_id=citation.id,
                quote=citation.direct_quote or "",
                status="could_not_verify",
                issue="Quote source text not provided",
                source_basis=(
                    "The motion includes a direct quote, but the primary authority text is not in the supplied case file. "
                    "The pipeline does not use model memory or free-form web search to judge quote accuracy."
                ),
                confidence=0.2,
                confidence_label="low",
                reasoning=(
                    "The quote can be extracted and reported, but exact quote accuracy cannot be verified without the "
                    "source authority text."
                ),
            )
            for citation in quoted_citations
        ]
