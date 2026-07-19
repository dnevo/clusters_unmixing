from __future__ import annotations

from pathlib import Path
from typing import Any

WANDB_ENTITY = "nev1958a_team"


def _import_wandb():
    """Import wandb lazily, so it's only required when use_wandb is actually enabled."""
    try:
        import wandb  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "use_wandb is enabled in the experiment config, but the 'wandb' package "
            "isn't installed. Install it with `pip install -e .[wandb]`."
        ) from exc
    return wandb


def log_model_run(
    *,
    project: str,
    group: str,
    run_name: str,
    config: dict[str, Any],
    metrics: dict[str, float],
    abundance_preview_rows: list[dict[str, Any]],
) -> None:
    """Log one model call (one entry of a run's `models` list) as its own W&B run.

    Scoping each model call to its own run - rather than one run per experiment
    run or one run for the whole batch - is what makes W&B's per-run comparison
    views (parallel coordinates, filtering/grouping by hyperparameter) actually
    useful for comparing models and hyperparameter sweeps against each other.
    All runs from one run_experiments() call share `group`, so they stay
    browsable together in the W&B UI despite being separate runs.
    """
    wandb = _import_wandb()
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=project,
        group=group,
        name=run_name,
        job_type="model_run",
        config=config,
        reinit=True,
    )
    try:
        run.log(metrics)
        if abundance_preview_rows:
            columns = list(abundance_preview_rows[0].keys())
            table = wandb.Table(
                columns=columns,
                data=[[row[c] for c in columns] for row in abundance_preview_rows],
            )
            run.log({"abundance_preview": table})
    finally:
        run.finish()


def log_batch_artifacts(
    *,
    project: str,
    group: str,
    run_name: str,
    artifact_name: str,
    csv_paths: dict[str, Path],
) -> None:
    """Upload a run_experiments() batch's aggregated output CSVs as one W&B artifact.

    These CSVs are only finalized once, after every model call in the batch has
    run, so they can't be attached to any single per-model run logged by
    log_model_run. This opens one extra run in the same `group`, clearly
    labeled as a summary, so the artifact stays discoverable next to the
    per-model runs it aggregates.
    """
    wandb = _import_wandb()
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=project,
        group=group,
        name=run_name,
        job_type="summary",
        reinit=True,
    )
    try:
        artifact = wandb.Artifact(name=artifact_name, type="experiment-outputs")
        for path in csv_paths.values():
            artifact.add_file(str(path))
        run.log_artifact(artifact)
    finally:
        run.finish()
