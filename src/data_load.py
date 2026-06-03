"""Load the seed JSON datasets into Unity Catalog Delta tables.

The JSON files in ``data/`` are the reproducible seed for the demo. This module
writes them once into managed Delta tables in the ``case_ticket`` schema, which
the agent's tools (live SQL) and the demo notebooks then read from. Run once from
``00_production_traffic_and_monitoring`` before the baseline.

Schemas are explicit so nested/optional fields and all-null columns load cleanly
on Unity Catalog clusters (no RDD APIs used).
"""
import json

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from .config import (
    CATALOG,
    CUSTOMERS_TABLE,
    DATA_DIR,
    HUMAN_LABELS_TABLE,
    KB_ARTICLES_TABLE,
    ORDERS_TABLE,
    SCHEMA,
    TICKETS_TABLE,
)

_CUSTOMERS_SCHEMA = StructType([
    StructField("customer_id", StringType(), False),
    StructField("name", StringType(), True),
    StructField("email", StringType(), True),
    StructField("tier", StringType(), True),
    StructField("account_age_days", LongType(), True),
    StructField("prior_unresolved_tickets", LongType(), True),
])

_ORDERS_SCHEMA = StructType([
    StructField("order_id", StringType(), False),
    StructField("customer_id", StringType(), True),
    StructField("product", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("status", StringType(), True),
    StructField("order_date", StringType(), True),
    StructField("shipping_date", StringType(), True),
    StructField("refund_eligible", BooleanType(), True),
])

_KB_ARTICLES_SCHEMA = StructType([
    StructField("article_id", StringType(), False),
    StructField("category", StringType(), True),
    StructField("title", StringType(), True),
    StructField("content", StringType(), True),
    StructField("tags", ArrayType(StringType()), True),
])

_TICKETS_SCHEMA = StructType([
    StructField("ticket_id", StringType(), False),
    StructField("customer_id", StringType(), True),
    StructField("order_id", StringType(), True),
    StructField("subject", StringType(), True),
    StructField("body", StringType(), True),
    StructField("_truth", StructType([
        StructField("expected_category", StringType(), True),
        StructField("should_escalate", BooleanType(), True),
        StructField("difficulty_bucket", StringType(), True),
    ]), True),
])

_HUMAN_LABELS_SCHEMA = StructType([
    StructField("ticket_id", StringType(), False),
    StructField("correct_category", StringType(), True),
    StructField("correct_should_escalate", BooleanType(), True),
    StructField("reviewer_notes", StringType(), True),
    StructField("difficulty_bucket", StringType(), True),
])

# filename -> (table, schema, ordered field list for normalization)
_SEEDS = [
    ("customers.json", CUSTOMERS_TABLE, _CUSTOMERS_SCHEMA),
    ("orders.json", ORDERS_TABLE, _ORDERS_SCHEMA),
    ("kb_articles.json", KB_ARTICLES_TABLE, _KB_ARTICLES_SCHEMA),
    ("tickets.json", TICKETS_TABLE, _TICKETS_SCHEMA),
    ("human_labels.json", HUMAN_LABELS_TABLE, _HUMAN_LABELS_SCHEMA),
]


def _normalize(records: list[dict], schema: StructType) -> list[dict]:
    """Ensure every record has every top-level field (missing -> None) so
    createDataFrame against an explicit schema never KeyErrors."""
    fields = [f.name for f in schema.fields]
    return [{name: rec.get(name) for name in fields} for rec in records]


def load_seed_tables(spark: SparkSession, overwrite: bool = True) -> dict[str, int]:
    """Create the ``case_ticket`` schema (if needed) and load every seed JSON
    into a managed Delta table. Returns {table_name: row_count}."""
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
    mode = "overwrite" if overwrite else "errorifexists"
    counts: dict[str, int] = {}
    for filename, table, schema in _SEEDS:
        records = json.loads((DATA_DIR / filename).read_text())
        df = spark.createDataFrame(_normalize(records, schema), schema=schema)
        (
            df.write.mode(mode)
            .option("overwriteSchema", "true")
            .saveAsTable(table)
        )
        counts[table] = df.count()
    return counts
