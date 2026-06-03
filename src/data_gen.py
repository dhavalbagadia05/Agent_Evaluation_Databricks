"""Synthetic ticket generation.

Generates 80 support tickets via LLM with explicit truth labels. Distribution:
- 22 billing, 20 shipping, 16 refund, 16 technical, 6 other
- Within each category: ~40% clear-self-serve, ~25% should-escalate, ~35% ambiguous

The (category, difficulty_bucket, should_escalate) triple is provided to the LLM
as input — labels are NOT derived from the generation. This avoids the failure
mode where the same LLM family produces both the ticket and the label.

Run from the repo root:
    python -m src.data_gen
"""
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

from .config import AI_GATEWAY_BASE_URL, DATA_DIR, DATABRICKS_TOKEN, JUDGE_MODEL


_client = OpenAI(api_key=DATABRICKS_TOKEN, base_url=AI_GATEWAY_BASE_URL)


# Per-category counts: (clear_self_serve, should_escalate, ambiguous)
DISTRIBUTION = {
    "billing":   (9, 5, 8),
    "shipping":  (8, 5, 7),
    "refund":    (6, 4, 6),
    "technical": (6, 4, 6),
    "other":     (3, 2, 1),
}


def _ambiguous_escalation_bias() -> bool:
    """Ambiguous bucket leans toward should_escalate=False (humans tend conservative)."""
    return random.random() < 0.35


def _build_plan(customers: list[dict], orders: list[dict]) -> list[dict]:
    rng = random.Random(42)
    orders_by_customer: dict[str, list[dict]] = {}
    for o in orders:
        orders_by_customer.setdefault(o["customer_id"], []).append(o)

    plan: list[dict] = []
    ticket_counter = 1

    for category, (n_clear, n_escalate, n_ambiguous) in DISTRIBUTION.items():
        specs: list[tuple[str, bool]] = []
        specs += [("clear_self_serve", False)] * n_clear
        specs += [("should_escalate", True)] * n_escalate
        for _ in range(n_ambiguous):
            specs.append(("ambiguous", _ambiguous_escalation_bias() if rng.random() < 1.0 else False))

        # For ambiguous bucket: random escalation truth biased toward False
        rebuilt = []
        for bucket, default_esc in specs:
            if bucket == "ambiguous":
                should_escalate = rng.random() < 0.35  # ~35% true, 65% false
            else:
                should_escalate = default_esc
            rebuilt.append((bucket, should_escalate))
        rng.shuffle(rebuilt)

        for bucket, should_escalate in rebuilt:
            customer = rng.choice(customers)
            attach_order = category in {"shipping", "refund"} or rng.random() < 0.4
            order = None
            if attach_order:
                cust_orders = orders_by_customer.get(customer["customer_id"], [])
                if cust_orders:
                    order = rng.choice(cust_orders)

            multi_issue = rng.random() < 0.13  # ~10 of 80 tickets are multi-issue

            plan.append({
                "ticket_id": f"T-{ticket_counter:04d}",
                "category": category,
                "difficulty_bucket": bucket,
                "should_escalate": should_escalate,
                "customer_id": customer["customer_id"],
                "order_id": order["order_id"] if order else None,
                "_customer": customer,
                "_order": order,
                "multi_issue": multi_issue,
            })
            ticket_counter += 1

    return plan


_GEN_INSTRUCTIONS = """\
You are generating a realistic customer support ticket for an online retailer.

Generate exactly one ticket matching the parameters below. The ticket should feel real — written in the customer's voice, with the level of clarity, frustration, or detail that a real customer would have.

Output ONLY valid JSON with these fields:
- "subject": short one-line subject
- "body": 1-4 sentences (or up to 5 if multi_issue=true) of the customer's message

Parameters:
- Category: {category}
- Difficulty: {bucket}
- Customer context: {customer_desc}
- Order context: {order_desc}
- Should be escalated by an agent: {should_escalate}
- Multi-issue (combine multiple problems in one ticket): {multi_issue}

Guidelines for difficulty:
- clear_self_serve: routine issue clearly resolvable via tools/KB. Customer may be mildly annoyed but problem is simple.
- should_escalate: contains clear escalation triggers (legal threat, fraud, safety, account compromise, repeated unresolved issue for high-tier customer, etc.) OR is genuinely outside what self-serve tools can handle.
- ambiguous: customer is frustrated or the situation is complex, but the underlying issue may still be resolvable. The right judgment call is debatable.

Do not include the truth label in the body. Do not include any markdown formatting. Output only the JSON object.
"""


def _customer_desc(c: dict) -> str:
    parts = [f"{c['name']} ({c['customer_id']})", f"{c['tier']} tier", f"account age {c['account_age_days']} days"]
    if c["prior_unresolved_tickets"] > 0:
        parts.append(f"{c['prior_unresolved_tickets']} prior unresolved tickets")
    return ", ".join(parts)


def _order_desc(o: dict | None) -> str:
    if o is None:
        return "no specific order mentioned"
    return (
        f"{o['order_id']} — {o['product']} (${o['amount']}, status: {o['status']}, "
        f"refund eligible: {o['refund_eligible']})"
    )


def _generate_one(spec: dict) -> dict:
    prompt = _GEN_INSTRUCTIONS.format(
        category=spec["category"],
        bucket=spec["difficulty_bucket"],
        customer_desc=_customer_desc(spec["_customer"]),
        order_desc=_order_desc(spec["_order"]),
        should_escalate=spec["should_escalate"],
        multi_issue=spec["multi_issue"],
    )
    response = _client.chat.completions.create(
        model=JUDGE_MODEL,  # use the stronger model for higher-quality data
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.9,
    )
    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("```", 2)[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.rsplit("```", 1)[0]
    start, end = content.find("{"), content.rfind("}")
    body_data = json.loads(content[start : end + 1])

    return {
        "ticket_id": spec["ticket_id"],
        "customer_id": spec["customer_id"],
        "order_id": spec["order_id"],
        "subject": body_data["subject"],
        "body": body_data["body"],
        "_truth": {
            "expected_category": spec["category"],
            "should_escalate": spec["should_escalate"],
            "difficulty_bucket": spec["difficulty_bucket"],
        },
    }


def generate_tickets(out_path: Path | None = None, max_workers: int = 8) -> list[dict]:
    customers = json.loads((DATA_DIR / "customers.json").read_text())
    orders = json.loads((DATA_DIR / "orders.json").read_text())

    plan = _build_plan(customers, orders)
    print(f"Generating {len(plan)} tickets...")

    tickets: list[dict] = [None] * len(plan)
    errors: list[tuple[int, Exception]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {pool.submit(_generate_one, spec): i for i, spec in enumerate(plan)}
        completed = 0
        for fut in as_completed(future_to_idx):
            i = future_to_idx[fut]
            try:
                tickets[i] = fut.result()
            except Exception as e:
                errors.append((i, e))
                tickets[i] = None
            completed += 1
            if completed % 10 == 0:
                print(f"  {completed}/{len(plan)}")

    tickets = [t for t in tickets if t is not None]

    if errors:
        print(f"WARNING: {len(errors)} tickets failed to generate")
        for i, e in errors[:5]:
            print(f"  idx {i}: {e}")

    out_path = out_path or (DATA_DIR / "tickets.json")
    out_path.write_text(json.dumps(tickets, indent=2))
    print(f"Wrote {len(tickets)} tickets to {out_path}")
    return tickets


if __name__ == "__main__":
    generate_tickets()
