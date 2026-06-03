"""Scorers for evaluating the support triage agent.

Three deterministic scorers + two LLM judges. The deterministic ones run on every
prod trace; the LLM judges (especially escalation_correctness) are the ones aligned
to human labels in 01_eval_flywheel.
"""
from typing import Any

from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import scorer

from .config import JUDGE_MODEL

# NOTE: deterministic scorers below define their constants INSIDE the function body.
# Production monitoring executes registered code scorers in an isolated context where
# module-level globals/imports (e.g. config constants) are NOT available — referencing
# them raises CUSTOM_METRIC_ERROR ("name '...' is not defined"). Keep these self-contained.


@scorer
def response_length_ok(outputs: dict[str, Any]) -> int:
    """1 if the draft response word count is in the acceptable range (20-150)."""
    length_min, length_max = 20, 150
    draft = outputs.get("draft", "") if isinstance(outputs, dict) else ""
    word_count = len(draft.split())
    return 1 if length_min <= word_count <= length_max else 0


@scorer
def category_in_enum(outputs: dict[str, Any]) -> int:
    """1 if the agent's category is one of the allowed values."""
    categories = {"billing", "shipping", "refund", "technical", "other"}
    cat = outputs.get("category", "") if isinstance(outputs, dict) else ""
    return 1 if cat in categories else 0


@scorer
def tool_calls_valid(trace) -> int:
    """1 if no tool_call span in this trace resulted in an error."""
    spans = trace.search_spans(name="tool_call") if trace else []
    for span in spans:
        outputs = getattr(span, "outputs", None)
        if isinstance(outputs, dict) and "error" in outputs:
            return 0
    return 1


_ESCALATION_RUBRIC = """\
Evaluate whether the agent's escalation decision for the ticket in {{ inputs }} \
was correct, given the agent's final output in {{ outputs }}.

Escalation IS appropriate when:
- The customer explicitly threatens legal action, chargeback, or churn
- The issue involves fraud, account security, or safety
- The issue is outside what self-serve tools can resolve
- The customer is enterprise tier AND the issue remains unresolved after lookup
- The customer has 3 or more prior unresolved tickets

Escalation is NOT appropriate when:
- The issue is fully resolvable via available tools (order status, refund eligibility, KB lookup)
- The customer is frustrated but the issue is routine (password reset, shipping delay within SLA)
- The agent escalated without performing any investigation

Return your decision as one of: correct, incorrect.
Provide a brief rationale.
"""

_HELPFULNESS_RUBRIC = """\
Evaluate whether the draft response in {{ outputs }} is helpful for resolving \
the customer's specific issue described in {{ inputs }}.

A response is HELPFUL when it:
- Directly addresses the specific issue the customer raised
- Uses information retrieved from tools when relevant
- Provides a concrete next step or resolution, OR clearly explains why escalation is needed

A response is NOT HELPFUL when it:
- Only acknowledges the issue without proposing any action
- Ignores retrieved customer or order context
- Gives generic advice unrelated to ticket specifics
- Defers everything to a human without attempting to resolve

Return your decision as one of: helpful, not_helpful.
Provide a brief rationale.
"""


escalation_correctness = make_judge(
    name="escalation_correctness",
    instructions=_ESCALATION_RUBRIC,
    model=f"databricks:/{JUDGE_MODEL}",
)


helpfulness = make_judge(
    name="helpfulness",
    instructions=_HELPFULNESS_RUBRIC,
    model=f"databricks:/{JUDGE_MODEL}",
)


DETERMINISTIC_SCORERS = [response_length_ok, category_in_enum, tool_calls_valid]
JUDGE_SCORERS = [escalation_correctness, helpfulness]
ALL_SCORERS = DETERMINISTIC_SCORERS + JUDGE_SCORERS
