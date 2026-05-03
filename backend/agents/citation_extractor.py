from __future__ import annotations

from llm import call_llm_json
from pydantic import BaseModel
from schemas import Citation

from .base import Agent


class CitationExtractionResult(BaseModel):
    citations: list[Citation]


class CitationExtractorAgent(Agent[list[Citation]]):
    name = "CitationExtractorAgent"
    prompt = "Extract every legal citation from the MSJ, including short-form and footnote authorities, with the proposition each supports."

    llm_prompt = """You are a legal citation extraction agent. Extract every legal citation from the Motion for Summary Judgment with high recall.

Rules:
- Extract citations only from the MSJ, not from your own legal knowledge.
- Include short-form citations such as Id. citations when they are used as authority.
- Extract each authority in footnotes and semicolon-separated citation strings as its own citation.
- Extract statutes and code sections, including bare references such as California Code of Civil Procedure Section 335.1.
- State the proposition narrowly from the surrounding MSJ text. Preserve material qualifiers such as never liable, OSHA compliance, rebuttable presumption, tort liability, and time-barred.
- If a direct quote appears in the same sentence, line, or immediately before a citation, set has_direct_quote to true and put only the exact quoted words in direct_quote.
- Do not verify whether the citation is correct. This agent only extracts.
"""

    def run(self, motion_text: str) -> list[Citation]:
        result = call_llm_json(
            messages=[
                {"role": "system", "content": self.llm_prompt},
                {"role": "user", "content": motion_text},
            ],
            schema=CitationExtractionResult,
        )
        return [citation.model_copy(update={"id": f"citation_{index}"}) for index, citation in enumerate(result.citations, start=1)]
