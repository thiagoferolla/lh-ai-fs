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
    prompt = "Compare MSJ fact claims against external case-file records using exact evidence snippets and conservative statuses."

    llm_prompt = """You are a cross-document legal fact consistency agent.

Given factual claims extracted from a Motion for Summary Judgment and the source case-file documents, compare each claim against the non-MSJ source documents.

Rules:
- Return one finding per input claim.
- Treat motion_for_summary_judgment as the claim source, not as verification evidence.
- Use only non-MSJ documents as source evidence: police_report, medical_records_excerpt, and witness_statement.
- source_document must be one of those exact document names, or semicolon-separated exact names. Do not include motion_for_summary_judgment in source_document.
- source_evidence must be an exact copied substring from the named non-MSJ document. Do not paraphrase, combine snippets, or fix spacing. If no exact external snippet supports the finding, use null.
- Prefer the narrowest exact snippet that proves the point and omit unrelated employment, location, or injury details.
- Mark supported only when external documents support all material parts of the claim.
- Mark contradicted when external documents directly conflict with a material part of the MSJ claim.
- Mark partially_supported when some parts are supported but the claim is overbroad, incomplete, or materially qualified by external records.
- Mark not_found when the external source set does not contain evidence for a material MSJ assertion.
- Use could_not_verify when the source set is insufficient to assess the claim and there is no direct contradiction.
- Do not add claims that were not in the input.

Issue-specific guidance:
- Incident date: if the MSJ says March 14 but external records say March 12, mark contradicted with medium/high confidence and mention both dates.
- PPE: if the MSJ says Rivera lacked or was not wearing required PPE but external records say he was wearing a hard hat, harness, or safety gear, mark contradicted and include both wearing and not-wearing language in the reasoning.
- OSHA/IIPP/inspection history: do not treat Cal/OSHA notification or absence of citations at the scene as proof that Harmon passed inspections or complied with OSHA. If external documents do not verify the claimed OSHA inspections/compliance history, mark not_found with confidence between 0.5 and 0.85 and leave source_evidence null unless an exact external snippet directly addresses OSHA history.
- Apex/Harmon control: contradict or qualify an Apex-only/exclusive scaffolding-control claim only when external evidence shows Harmon/Donner directed the specific scaffold work or had/dismissed specific scaffold-safety concerns. Do not contradict based solely on Harmon being the general contractor, Donner being present, or general schedule/location comments.
- Limitations/time-bar: if the MSJ frames the claim as time-barred or rests on March 14 while external records show a March 12 incident and the MSJ filing date is March 10, 2023, explain that the filing appears within two years of either incident date. Use contradicted or partially_supported with medium confidence.
- Avoid false positives: Rivera's Apex employment, the 2200 West Olympic Boulevard location, and the left leg/lower back/left wrist injuries are supported background facts. Do not flag them as issues.
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
