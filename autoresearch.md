# Autoresearch: Optimize Agent Prompts for Legal Verification Pipeline

## Objective
Improve the quality of LLM agent prompts in the BS Detector legal verification pipeline. The agents extract citations, verify authority, check quotes, extract facts, check cross-document consistency, and synthesize judicial memos. The goal is to maximize the eval suite's core_recall (catching known flaws) while maintaining precision and low hallucination_rate.

## Metrics
- **Primary**: core_recall (unitless, higher is better) — fraction of 6 expected gold findings semantically matched
- **Secondary**: precision, hallucination_rate, expanded_recall, citation_extraction_recall, uncertainty_accuracy, mutation_pass_rate, evidence_grounding_rate, clean_case_false_positive_rate, fabricated_citation_detection_rate

## How to Run
`./autoresearch.sh` — outputs `METRIC name=number` lines for each metric.

## Files in Scope
- `backend/agents/citation_extractor.py` — CitationExtractorAgent prompt: extracts legal citations from MSJ
- `backend/agents/authority_verifier.py` — AuthorityVerifierAgent prompt: assesses if cited authority supports proposition
- `backend/agents/quote_checker.py` — QuoteCheckerAgent prompt: checks direct quotes for accuracy/overbreadth
- `backend/agents/fact_extractor.py` — FactExtractorAgent prompt: extracts verifiable factual claims from MSJ
- `backend/agents/consistency_checker.py` — ConsistencyCheckerAgent prompt: compares MSJ claims against source documents
- `backend/agents/judicial_memo.py` — JudicialMemoAgent prompt: synthesizes high-confidence findings into a paragraph
- `backend/agents/orchestrator.py` — Orchestrator that runs agents and builds flags (flag-building logic affects what becomes a flag)
- `backend/agents/base.py` — Base agent class with confidence_label helper
- `backend/schemas.py` — Pydantic models for inter-agent data flow
- `backend/run_evals.py` — Eval harness (may NOT be modified — that would be cheating)

## Off Limits
- `backend/run_evals.py` — the eval harness must NOT be modified. Changing the test to make it pass is cheating.
- `backend/llm.py` — LLM client config (model choice, API endpoints)
- `backend/documents/` — the case documents must NOT be modified
- `backend/schemas.py` — data models define the contract; changing them to make evals pass would be cheating

## Constraints
- Only modify agent prompts and orchestrator flag-building logic
- Must NOT modify the eval harness (run_evals.py)
- Must NOT modify the source documents
- Must NOT change the LLM model or add external data sources
- Prompts must remain honest — instruct agents to express uncertainty, not fabricate findings
- The pipeline must remain conservative: `could_not_verify` is acceptable for obscure authorities

## What's Been Tried
(Updated as experiments accumulate)

### Baseline
- Initial run to establish baseline metrics
