from __future__ import annotations

import json

from llm import call_llm_json
from pydantic import BaseModel, TypeAdapter
from schemas import Citation, QuoteCheck

from .base import Agent


class QuoteCheckResult(BaseModel):
    quote_checks: list[QuoteCheck]


class QuoteCheckerAgent(Agent[list[QuoteCheck]]):
    name = "QuoteCheckerAgent"
    prompt = "Check direct quotes adjacent to citations for exact-source limits and material overbreadth."

    llm_prompt = """You are a legal quote checking agent.

Given extracted citations, review only citations that include direct_quote. Assess whether the quote is accurate, materially overbroad, unsupported, or impossible to verify from the available information.

Return JSON only with this shape:
{
  "quote_checks": [
    {
      "citation_id": "citation_1",
      "quote": "quoted text",
      "status": "supported|partially_supported|not_supported|could_not_verify|likely_fabricated",
      "issue": "short issue or null",
      "source_basis": "what you relied on",
      "confidence": 0.0,
      "confidence_label": "low|medium|high",
      "reasoning": "concise explanation"
    }
  ]
}

Rules:
- Return checks only for citations with direct_quote.
- Do not claim exact quote verification unless primary source text is present. If no primary text is present, state that in source_basis.
- If a quote appears materially broader than the legal rule, flag that explicitly.
- Use could_not_verify when source text is unavailable and you lack a confident basis.
- Privette guardrail: a quote or paraphrase stating that a hirer is "never liable" or categorically immune for independent-contractor employee injuries is materially overbroad because Privette is a presumptive rule with exceptions and limits. Mark that quote not_supported, with an issue mentioning overbreadth or absolute/categorical framing, and confidence around 0.6-0.8.
- For other quotes without source text and no clear doctrinal overbreadth, prefer could_not_verify with confidence 0.5 or lower instead of inventing an accuracy finding.
"""

    def run(self, citations: list[Citation]) -> list[QuoteCheck]:
        quoted_citations = [citation for citation in citations if citation.has_direct_quote and citation.direct_quote]
        if not quoted_citations:
            return []
        result = call_llm_json(
            messages=[
                {"role": "system", "content": self.llm_prompt},
                {"role": "user", "content": json.dumps([citation.model_dump() for citation in quoted_citations])},
            ],
            adapter=TypeAdapter(QuoteCheckResult),
        )
        return result.quote_checks
