"""Customer support triage agent.

Reads its system prompt from the MLflow Prompt Registry (via alias) so prompt
promotion automatically updates the running agent. Falls back to the local
prompts/system_prompt_v1.md file if MLflow isn't reachable (useful for local dev).
"""
import ast
import json
from typing import Any

import mlflow
from openai import OpenAI

from .config import (
    AGENT_MODEL,
    AI_GATEWAY_BASE_URL,
    DATABRICKS_TOKEN,
    PROMPT_ALIAS,
    PROMPT_NAME,
    PROMPTS_DIR,
)
from .tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

_client = OpenAI(api_key=DATABRICKS_TOKEN, base_url=AI_GATEWAY_BASE_URL)

# Immutable output contract appended to whatever behaviour prompt is loaded.
# Kept in code (not the registered prompt) so it survives prompt edits when
# iterating on a candidate prompt — every prompt version still produces parseable JSON.
_OUTPUT_CONTRACT = """

---
OUTPUT FORMAT (required, do not ignore):
After any tool calls, your FINAL reply must be a single valid JSON object and NOTHING else — no prose, no explanation, no markdown code fences, and no tool-call text before or after it. Start the reply with `{` and end with `}`.
Use exactly these keys:
- "category": one of billing | shipping | refund | technical | other
- "escalate": JSON boolean true or false (lowercase, unquoted)
- "draft": the full response to send to the customer (a JSON string)
- "reasoning": a brief explanation of the category and escalate decisions (a JSON string)
Never invoke a tool by writing its name as text; use the tool interface only. When you have enough information to answer, output only the JSON object.
"""


def _load_system_prompt() -> str:
    # Loading from the registry is what links the prompt version to the trace. Retry a few times
    # before falling back to the local file, since transient registry failures under the traffic
    # loop's concurrency would otherwise silently drop the prompt link on those traces.
    import time
    last_err = None
    for attempt in range(3):
        try:
            prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@{PROMPT_ALIAS}")
            return prompt.template if hasattr(prompt, "template") else str(prompt)
        except Exception as e:
            last_err = e
            time.sleep(0.3 * (attempt + 1))
    print(f"[agent] load_prompt fell back to local prompt file (no registry link): {last_err}")
    with open(PROMPTS_DIR / "system_prompt_v1.md") as f:
        return f.read()


def _format_ticket(ticket: dict[str, Any]) -> str:
    lines = [f"Subject: {ticket['subject']}", f"Body: {ticket['body']}"]
    if ticket.get("customer_id"):
        lines.append(f"customer_id: {ticket['customer_id']}")
    if ticket.get("order_id"):
        lines.append(f"order_id: {ticket['order_id']}")
    return "\n".join(lines)


def _parse_final_output(content: str) -> dict[str, Any]:
    """Extract the JSON output block from the model's final message."""
    s = content.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.rsplit("```", 1)[0]
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1:
        return {"category": "other", "escalate": True, "draft": content, "reasoning": "Failed to parse structured output"}
    blob = s[start : end + 1]
    # The model occasionally emits a Python-style dict (single quotes, True/False)
    # instead of strict JSON. Try strict JSON first, then ast.literal_eval.
    for parse in (json.loads, ast.literal_eval):
        try:
            parsed = parse(blob)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError):
            continue
    return {"category": "other", "escalate": True, "draft": content, "reasoning": "Failed to parse structured output"}


@mlflow.trace
def run_agent(ticket: dict[str, Any], system_prompt: str | None = None, trace_tags: dict[str, str] | None = None, max_iters: int = 6) -> dict[str, Any]:
    """Run the triage agent on one ticket. Returns {category, escalate, draft, reasoning}.

    trace_tags, if provided, are attached to this call's trace (e.g. {"environment": "production"}).
    """
    if trace_tags:
        mlflow.update_current_trace(tags=trace_tags)
    system_prompt = (system_prompt or _load_system_prompt()) + _OUTPUT_CONTRACT
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _format_ticket(ticket)},
    ]

    for _ in range(max_iters):
        response = _client.chat.completions.create(
            model=AGENT_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            max_tokens=4000,
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                with mlflow.start_span(name="tool_call") as span:
                    span.set_inputs({"tool": tc.function.name, "arguments_raw": tc.function.arguments})
                    fn = TOOL_FUNCTIONS.get(tc.function.name)
                    try:
                        args = json.loads(tc.function.arguments)
                        result = fn(**args) if fn else {"error": f"Unknown tool {tc.function.name}"}
                    except Exception as e:
                        result = {"error": str(e)}
                    span.set_outputs(result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })
            continue

        return _parse_final_output(msg.content or "")

    return {"category": "other", "escalate": True, "draft": "Agent loop exceeded max iterations", "reasoning": "max_iters reached"}
