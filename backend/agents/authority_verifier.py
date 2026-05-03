from __future__ import annotations

import json

from llm import call_llm_json
from pydantic import BaseModel
from schemas import Citation, CitationVerification

from .base import Agent


class AuthorityVerificationResult(BaseModel):
    citation_verifications: list[CitationVerification]


class AuthorityVerifierAgent(Agent[list[CitationVerification]]):
    name = "AuthorityVerifierAgent"
    prompt = "Conservatively verify whether each cited authority supports the MSJ proposition. Do not invent holdings; use could_not_verify when source text is unavailable."

    llm_prompt = """You are a legal authority verification agent.

Given citations extracted from a Motion for Summary Judgment, assess whether each cited authority supports the proposition the MSJ uses it for.

Rules:
- Return one verification per input citation.
- Do not pretend you retrieved primary authority text unless it is provided in the input.
- Use could_not_verify for obscure authorities you cannot confidently assess.
- Use likely_fabricated only when the citation is facially suspicious or impossible.
- Be conservative with confidence when relying on general legal knowledge.
"""

    def run(self, citations: list[Citation]) -> list[CitationVerification]:
        result = call_llm_json(
            messages=[
                {"role": "system", "content": self.llm_prompt},
                {"role": "user", "content": json.dumps([citation.model_dump() for citation in citations])},
            ],
            schema=AuthorityVerificationResult,
        )
        return result.citation_verifications
