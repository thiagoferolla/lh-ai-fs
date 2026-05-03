from __future__ import annotations

import json

from llm import call_llm_json
from pydantic import BaseModel, TypeAdapter
from schemas import Citation, CitationVerification

from .base import Agent


class AuthorityVerificationResult(BaseModel):
    citation_verifications: list[CitationVerification]


class AuthorityVerifierAgent(Agent[list[CitationVerification]]):
    name = "AuthorityVerifierAgent"
    prompt = "Conservatively verify whether each cited authority supports the MSJ proposition, with explicit uncertainty when source text is unavailable."

    llm_prompt = """You are a legal authority verification agent.

Given citations extracted from a Motion for Summary Judgment, assess whether each cited authority supports the proposition the MSJ uses it for.

Return JSON only with this shape:
{
  "citation_verifications": [
    {
      "citation_id": "citation_1",
      "status": "supported|partially_supported|not_supported|could_not_verify|likely_fabricated",
      "issue": "short issue or null",
      "source_basis": "what you relied on, including whether this is LLM legal knowledge or lack of source text",
      "confidence": 0.0,
      "confidence_label": "low|medium|high",
      "reasoning": "concise explanation"
    }
  ]
}

Rules:
- Return one verification per input citation.
- Do not pretend you retrieved primary authority text unless it is provided in the input.
- If primary authority text is not provided, say so in source_basis using wording like "LLM-only; no primary source text retrieved" and keep confidence conservative.
- Use could_not_verify with confidence 0.5 or lower for obscure authorities you cannot confidently assess.
- Use likely_fabricated only when the citation is facially suspicious or impossible.
- Be conservative with confidence when relying on general legal knowledge.
- Privette doctrine: it is a presumptive no-liability rule for hirers of independent contractors, not an absolute never-liable rule. If the proposition says the hirer is never or categorically liable for all contractor-employee injuries, mark not_supported, identify the rule as overbroad, and keep confidence around 0.6-0.8 unless source text is supplied.
- Seabright/OSHA: for known Seabright-based propositions, do not treat OSHA compliance or Seabright as complete insulation from tort liability. If the proposition says Seabright or OSHA compliance effectively insulates Harmon from tort liability, mark partially_supported or not_supported with medium confidence.
- Do not mark obscure OSHA-presumption authorities unsupported merely because the rule sounds broad; without source text, mark those could_not_verify with confidence 0.5 or lower.
- Section 335.1 supports a two-year personal-injury limitations period. It does not by itself support a time-bar conclusion when the filing date is within two years of the incident date.
- Fabrication cues include impossible reporters, impossible court/date combinations, future-year citations, placeholder-looking names, or nonsensical reporter series such as F.9th.
"""

    def run(self, citations: list[Citation]) -> list[CitationVerification]:
        result = call_llm_json(
            messages=[
                {"role": "system", "content": self.llm_prompt},
                {"role": "user", "content": json.dumps([citation.model_dump() for citation in citations])},
            ],
            adapter=TypeAdapter(AuthorityVerificationResult),
        )
        return result.citation_verifications
