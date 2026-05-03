from __future__ import annotations

from llm import call_llm_json
from pydantic import BaseModel
from schemas import FactClaim

from .base import Agent


class FactExtractionResult(BaseModel):
    fact_claims: list[FactClaim]


class FactExtractorAgent(Agent[list[FactClaim]]):
    name = "FactExtractorAgent"
    prompt = "Extract material, verifiable MSJ fact claims with stable IDs for downstream consistency checks."

    llm_prompt = """You are a legal fact extraction agent. Extract material factual claims from the Motion for Summary Judgment that could be verified against source documents.

Rules:
- Extract claims from the MSJ only.
- Do not verify the claims.
- Prefer factual premises that matter to liability, limitations, safety compliance, or source-document consistency. Do not extract harmless background facts as standalone claims unless they are part of a disputed theory.
- Keep claims narrow. Avoid bundling unrelated supported facts such as employment, site address, or injury list into disputed claims.
- Include dates, PPE assertions, OSHA/inspection assertions, control/supervision assertions, filing dates, limitations/accrual assertions, and injury-notice assertions.
- Use these canonical ids whenever the corresponding MSJ claim appears:
  - incident_date: the date of the incident or accident.
  - no_ppe: an assertion that Rivera lacked or was not wearing required PPE/fall-arrest equipment.
  - osha_compliance: assertions that Harmon had an IIPP, passed OSHA inspections, or had OSHA compliance/inspection history.
  - apex_control: assertions that Apex, not Harmon, controlled or was responsible for scaffolding operations or safety procedures.
  - limitations_elapsed: limitations, filing-date, time-bar, accrual, or immediate-injury assertions.
- For incident_date, write the claim as a date claim only; do not include the site address.
- For apex_control, focus on control/responsibility for scaffolding work, not the undisputed fact that Rivera was employed by Apex.
- Use stable, descriptive snake_case ids for any other claims because downstream agents refer to claim ids.
"""

    def run(self, motion_text: str) -> list[FactClaim]:
        result = call_llm_json(
            messages=[
                {"role": "system", "content": self.llm_prompt},
                {"role": "user", "content": motion_text},
            ],
            schema=FactExtractionResult,
        )
        return result.fact_claims
