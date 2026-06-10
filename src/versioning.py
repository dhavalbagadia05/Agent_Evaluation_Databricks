"""Agent application versioning via MLflow LoggedModels.

`set_active_agent_model()` marks the active agent version (`config.AGENT_VERSION`) so every
trace produced afterward links to it (`trace.info.model_id`). This is the agent/app version
axis — DECOUPLED from the prompt version: the prompt is versioned separately in the registry
and rides through the `production` alias, linking to each trace on its own.

Bump `config.AGENT_VERSION` when the agent CODE / tools / model change — NOT when the prompt
changes (promoting a candidate prompt keeps the same agent version, by design).
"""
import mlflow

from . import config


def set_active_agent_model(experiment_id: str, params: dict | None = None) -> str:
    """Idempotently set the active LoggedModel for the agent version; return its model_id.

    Reuses the existing model named ``config.AGENT_VERSION`` in this experiment if present (so
    repeated runs don't stack duplicates); otherwise creates it and records ``params`` on it.
    """
    name = config.AGENT_VERSION
    model_id = None
    try:
        models = mlflow.search_logged_models(experiment_ids=[experiment_id], output_format="list")
        match = next((m for m in models if getattr(m, "name", None) == name), None)
        if match is not None:
            model_id = match.model_id
    except Exception as e:
        print(f"[versioning] could not search existing models ({e}); creating a new one.")

    if model_id:
        mlflow.set_active_model(model_id=model_id)
        print(f"Active agent version '{name}' (reused model_id={model_id})")
    else:
        active = mlflow.set_active_model(name=name)
        model_id = active.model_id
        if params:
            mlflow.log_model_params(model_id=model_id, params={k: str(v) for k, v in params.items()})
        print(f"Active agent version '{name}' (created model_id={model_id})")
    return model_id
