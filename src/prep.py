"""One-call environment prep for 00_production_traffic_and_monitoring.

Keeps the setup notebook focused on the eval/monitoring flow. `prepare(spark, reset=True)`
optionally resets prior state, ensures the synthetic tickets exist, loads all seed JSON into
Unity Catalog tables, and returns the tickets. `reset_all(spark)` does the clean-slate reset on
its own (drop tables/views, delete the prompt so it re-registers at v1, delete the experiment).
"""
import mlflow
from mlflow import MlflowClient

from . import config
from .data_gen import generate_tickets
from .data_load import load_seed_tables

# Legacy OTel trace tables from an earlier trace_location setup; reset drops them if present.
TRACE_TABLE_PREFIX = "support_triage"
_OTEL_SUFFIXES = ("otel_annotations", "otel_logs", "otel_metrics", "otel_spans",
                  "trace_metadata", "trace_unified")


def prepare(spark, reset: bool = True) -> list[dict]:
    if reset:
        reset_all(spark)

    # The synthetic tickets are committed for reproducibility; generate only if missing.
    if not (config.DATA_DIR / "tickets.json").exists():
        print("Generating synthetic tickets...")
        generate_tickets()

    counts = load_seed_tables(spark)
    for table, n in counts.items():
        print(f"{n:>4}  {table}")

    # Return only agent-facing columns — never hand the synthetic ground truth (_truth) to the
    # agent, so it can't leak into trace inputs / what the LLM judges read. (_truth stays in the
    # table for 02_human_review's difficulty-bucket selection.)
    tickets = [
        r.asDict(recursive=True)
        for r in spark.table(config.TICKETS_TABLE)
        .select("ticket_id", "customer_id", "order_id", "subject", "body")
        .orderBy("ticket_id")
        .collect()
    ]
    print(f"{len(tickets)} tickets loaded from {config.TICKETS_TABLE}")
    return tickets


# --------------------------------------------------------------------------- reset

def reset_all(spark) -> None:
    """Drop tables, delete the prompt, and delete the experiment for a clean re-run."""
    _drop_tables(spark)
    _delete_prompt(MlflowClient())
    _delete_experiment()
    print("RESET complete")


def _drop_tables(spark) -> None:
    objs = [
        config.CUSTOMERS_TABLE, config.ORDERS_TABLE, config.KB_ARTICLES_TABLE,
        config.TICKETS_TABLE, config.HUMAN_LABELS_TABLE, config.EVAL_DATASET_TABLE,
    ]
    objs += [f"{config.CATALOG}.{config.SCHEMA}.{TRACE_TABLE_PREFIX}_{s}" for s in _OTEL_SUFFIXES]
    for obj in objs:
        # Some OTel objects are VIEWS, so try DROP TABLE then fall back to DROP VIEW.
        try:
            spark.sql(f"DROP TABLE IF EXISTS {obj}")
        except Exception:
            try:
                spark.sql(f"DROP VIEW IF EXISTS {obj}")
            except Exception as e:
                print("  drop skipped:", obj, e)
    print("Dropped tables/views (seed + eval + OTel trace storage)")


def _delete_prompt(client) -> None:
    """UC requires removing the alias + all versions before deleting the prompt."""
    try:
        for drop_alias in (
            lambda: mlflow.genai.delete_prompt_alias(config.PROMPT_NAME, config.PROMPT_ALIAS),
            lambda: client.delete_prompt_alias(config.PROMPT_NAME, config.PROMPT_ALIAS),
        ):
            try:
                drop_alias(); break
            except Exception:
                continue
        # Enumerate versions (result may be a list or expose .prompt_versions); else numeric sweep.
        vers = []
        try:
            resp = client.search_prompt_versions(config.PROMPT_NAME)
            items = getattr(resp, "prompt_versions", resp)
            vers = [getattr(v, "version", v) for v in items]
        except Exception:
            pass
        if not vers:
            vers = list(range(1, 50))
        for v in vers:
            try:
                client.delete_prompt_version(config.PROMPT_NAME, str(v))
            except Exception:
                pass
        client.delete_prompt(config.PROMPT_NAME)
        print("Deleted prompt", config.PROMPT_NAME)
    except Exception as e:
        print("Prompt delete skipped (likely does not exist yet):", e)


def _delete_experiment() -> None:
    exp = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
    if exp is not None:
        mlflow.delete_experiment(exp.experiment_id)
        print("Deleted experiment", config.MLFLOW_EXPERIMENT_NAME)
