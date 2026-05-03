from __future__ import annotations

import json

from llm import call_llm_json
from pydantic import BaseModel
from schemas import ConsistencyFinding, FactClaim

from .base import Agent


class ConsistencyCheckResult(BaseModel):
    consistency_findings: list[ConsistencyFinding]


class ConsistencyCheckerAgent(Agent[list[ConsistencyFinding]]):
    name = "ConsistencyCheckerAgent"
    prompt = "Compare structured MSJ fact claims against police, medical, and witness records. Return contradictions, support, or could_not_verify with evidence snippets."

    llm_prompt = """You are a cross-document legal fact consistency agent.

Given factual claims extracted from a Motion for Summary Judgment and the source case-file documents, compare each claim against the source documents.

Rules:
- Return one finding per input claim.
- Use only the provided source documents as evidence.
- Prefer exact source snippets for source_evidence.
- Mark contradicted only when source documents conflict with the MSJ claim.
- Mark not_found when the source set does not contain evidence for the MSJ claim.
- Do not add claims that were not in the input.
"""

    def run(self, claims: list[FactClaim], documents: dict[str, str]) -> list[ConsistencyFinding]:
        if not claims:
            return []
        result = call_llm_json(
            messages=[
                {"role": "system", "content": self.llm_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "claims": [claim.model_dump() for claim in claims],
                            "documents": documents,
                        }
                    ),
                },
            ],
            schema=ConsistencyCheckResult,
        )
        return result.consistency_findings
