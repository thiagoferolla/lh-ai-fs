#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/backend"

# Run evals and parse metrics from JSON output
OUTPUT=$(python run_evals.py 2>/dev/null)

# Extract metrics using python
echo "$OUTPUT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
m = data['metrics']
# Primary metric
print(f\"METRIC core_recall={m['core_recall']}\")
# Secondary metrics
print(f\"METRIC precision={m['precision']}\")
print(f\"METRIC hallucination_rate={m['hallucination_rate']}\")
print(f\"METRIC expanded_recall={m['expanded_recall']}\")
print(f\"METRIC citation_extraction_recall={m['citation_extraction_recall']}\")
print(f\"METRIC uncertainty_accuracy={m['uncertainty_accuracy']}\")
print(f\"METRIC mutation_pass_rate={m['mutation_pass_rate']}\")
print(f\"METRIC evidence_grounding_rate={m['evidence_grounding_rate']}\")
print(f\"METRIC clean_case_false_positive_rate={m['clean_case_false_positive_rate']}\")
print(f\"METRIC fabricated_citation_detection_rate={m['fabricated_citation_detection_rate']}\")
print(f\"METRIC matched_count={m['matched_count']}\")
print(f\"METRIC flag_count={m['flag_count']}\")
# Also print agent errors count
errors = data.get('agent_errors', [])
print(f\"METRIC agent_error_count={len(errors)}\")
# Print missed findings for context
missed = data.get('core_gold', {}).get('missed', [])
if missed:
    for item in missed:
        print(f\"# MISSED: {item['id']} - {item.get('reason', 'unknown')}\")
weak = data.get('core_gold', {}).get('weak_matches', [])
if weak:
    for item in weak:
        print(f\"# WEAK: {item['id']} - status_ok={item.get('status_ok')}, confidence_ok={item.get('confidence_ok')}, actual_status={item.get('actual_status')}, actual_conf={item.get('actual_confidence')}\")
"

# Pass through exit code
EXIT_CODE=$?
exit $EXIT_CODE
