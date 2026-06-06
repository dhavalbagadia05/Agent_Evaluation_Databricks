"""Central configuration for the Eval Flywheel demo."""
import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PROMPTS_DIR = REPO_ROOT / "prompts"

load_dotenv(REPO_ROOT / ".env")

DATABRICKS_HOST = os.environ["DATABRICKS_HOST"].rstrip("/")
DATABRICKS_TOKEN = os.environ["DATABRICKS_TOKEN"]

AI_GATEWAY_BASE_URL = f"{DATABRICKS_HOST}/ai-gateway/mlflow/v1"

# Models are configurable via .env (defaults below). AGENT_MODEL backs the triage
# agent; JUDGE_MODEL backs the LLM judges. Both must be chat endpoints reachable
# through the AI Gateway.
AGENT_MODEL = os.environ.get("AGENT_MODEL", "databricks-gpt-5-4")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "databricks-claude-opus-4-6")

# Unity Catalog location for all demo assets (prompt registry, eval dataset, backing
# tables). Override CATALOG/SCHEMA in .env to run the demo in your own workspace — every
# fully-qualified name below is derived from these two, so they propagate everywhere.
CATALOG = os.environ.get("CATALOG", "your_catalog")
SCHEMA = os.environ.get("SCHEMA", "case_ticket")
PROMPT_NAME = f"{CATALOG}.{SCHEMA}.support_triage_prompt"
EVAL_DATASET_TABLE = f"{CATALOG}.{SCHEMA}.evaluation_set_from_prod"

# Backing data tables (seeded from data/*.json by src.data_load.load_seed_tables).
# The agent's tools query these live; the notebooks read tickets/labels from them too.
CUSTOMERS_TABLE = f"{CATALOG}.{SCHEMA}.customers"
ORDERS_TABLE = f"{CATALOG}.{SCHEMA}.orders"
KB_ARTICLES_TABLE = f"{CATALOG}.{SCHEMA}.kb_articles"
TICKETS_TABLE = f"{CATALOG}.{SCHEMA}.tickets"
HUMAN_LABELS_TABLE = f"{CATALOG}.{SCHEMA}.human_labels"

PROMPT_ALIAS = "production"

CATEGORIES = ["billing", "shipping", "refund", "technical", "other"]

RESPONSE_LENGTH_MIN = 20
RESPONSE_LENGTH_MAX = 150

# Tags stamped on production-traffic traces (by run_agent in 00) so 01/02 can select
# them and tell them apart from evaluation-run traces.
PRODUCTION_TRACE_TAGS = {"environment": "production"}
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "/Shared/eval_flywheel_support_triage")

# SQL warehouse ID used for production scorer monitoring (registered scorers run on
# sampled prod traces via this warehouse). Leave empty to skip monitoring registration
# in 00_production_traffic_and_monitoring. Requires CAN USE on the warehouse and CAN EDIT on the experiment.
MONITORING_WAREHOUSE_ID = os.environ.get("MONITORING_WAREHOUSE_ID", "")
