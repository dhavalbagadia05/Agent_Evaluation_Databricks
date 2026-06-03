"""Tools available to the support triage agent.

Each tool queries a Unity Catalog table live via Spark SQL — the agent reads the
lakehouse on every call. The backing tables are seeded once from data/*.json by
src.data_load.load_seed_tables (run in 00_production_traffic_and_monitoring).
"""
from typing import Any

from pyspark.sql import SparkSession

from .config import CUSTOMERS_TABLE, KB_ARTICLES_TABLE, ORDERS_TABLE


def _spark() -> SparkSession:
    return SparkSession.builder.getOrCreate()


def _esc(value: str) -> str:
    """Escape single quotes for safe inlining into a SQL literal."""
    return str(value).replace("'", "''")


def lookup_customer(customer_id: str) -> dict[str, Any]:
    """Return account context for a customer: tier, account age, prior unresolved tickets."""
    rows = _spark().sql(
        f"SELECT * FROM {CUSTOMERS_TABLE} WHERE customer_id = '{_esc(customer_id)}' LIMIT 1"
    ).collect()
    if not rows:
        return {"error": f"Customer {customer_id} not found"}
    return rows[0].asDict(recursive=True)


def check_order_status(order_id: str) -> dict[str, Any]:
    """Return status, shipping, and refund eligibility for an order."""
    rows = _spark().sql(
        f"SELECT * FROM {ORDERS_TABLE} WHERE order_id = '{_esc(order_id)}' LIMIT 1"
    ).collect()
    if not rows:
        return {"error": f"Order {order_id} not found"}
    return rows[0].asDict(recursive=True)


def search_kb(query: str) -> list[dict[str, Any]]:
    """Return up to 3 KB articles matching the query by simple keyword overlap."""
    articles = [
        r.asDict(recursive=True)
        for r in _spark().sql(f"SELECT * FROM {KB_ARTICLES_TABLE}").collect()
    ]
    query_tokens = {t.lower().strip(".,?!") for t in query.split()}
    scored = []
    for a in articles:
        haystack = f"{a['title']} {a['content']} {' '.join(a['tags'])} {a['category']}".lower()
        score = sum(1 for t in query_tokens if t in haystack)
        if score > 0:
            scored.append((score, a))
    scored.sort(key=lambda x: -x[0])
    return [a for _, a in scored[:3]]


TOOL_FUNCTIONS = {
    "lookup_customer": lookup_customer,
    "check_order_status": check_order_status,
    "search_kb": search_kb,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_customer",
            "description": "Return account context for a customer (tier, account age, prior unresolved ticket count).",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "Customer ID, e.g. 'C-019'"}
                },
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_order_status",
            "description": "Return status, shipping info, and refund eligibility for an order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Order ID, e.g. 'O-1182'"}
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Search the knowledge base for articles relevant to a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (keywords or natural language)"}
                },
                "required": ["query"],
            },
        },
    },
]
