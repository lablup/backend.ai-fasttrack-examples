#!/usr/bin/env python3
"""Node 07 — arrange everything the two services need in the output mount.

In:   <data-root>/{03_indices,99_state}/ + this checkout
Out:  <data-root>/
        docs-rag/                     this code tree (minus data and caches)
        docs-rag/pipeline/data/03_indices/   the indices to serve
        99_state/service_credentials.json    login for both services
        model-definition-{fastapi,gradio}.yaml   if not already present

This is a custom task, so it stages into its own output root like every other
node in the chain. Deployment tasks can read the previous task's
/pipeline/outputs, so that is enough for the services to find the code and the
indices, and nothing here needs a model-storage mount or writes to a vfolder.

Set PIPELINE_MODEL_ROOT to stage into model storage instead — the layout is
identical, so the model definitions only need their start_command repointed.
That was the only option before deployment tasks could see /pipeline, and it
cost this node a mount that a custom task is not given by default.

Staging the code too means the services start with no network access and no git.

    python pipeline/tasks/07_stage_service.py --project all
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from docs_rag import credentials
from pipeline.io import SourcesConfig
from pipeline.paths import PROJECT_ROOT, STAGE_INDICES, STAGE_STATE
from pipeline.preflight import on_fasttrack, require_stage, require_writable
from pipeline.runner import run_task

log = logging.getLogger("07_stage_service")

# Never stage these: runtime data (large, and regenerated),
# build caches, and anything that could carry a secret.
CODE_EXCLUDES = shutil.ignore_patterns(
    "data", ".git", "__pycache__", "*.pyc", ".venv*", ".env", "vfroot", ".pytest_cache",
)

MODEL_DEFINITIONS = ("model-definition-fastapi.yaml", "model-definition-gradio.yaml")


def stage_code(destination: Path) -> int:
    """Copy this checkout to the staging root, replacing any previous copy."""
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(PROJECT_ROOT, destination, ignore=CODE_EXCLUDES)
    return sum(1 for p in destination.rglob("*") if p.is_file())


def stage_model_definitions(model_root: Path) -> list[str]:
    """Place the default model definitions, without clobbering custom ones.

    An operator who uploaded their own tuned definition keeps it; one who
    uploaded nothing still gets a working service instead of a start failure.
    """
    placed = []
    for name in MODEL_DEFINITIONS:
        target = model_root / name
        if target.exists():
            log.info("%s already present at the staging root — leaving it alone", name)
            continue
        source = PROJECT_ROOT / name
        if not source.is_file():
            log.warning("%s missing from the checkout — cannot stage it", name)
            continue
        shutil.copy2(source, target)
        placed.append(name)
    return placed


def stage_credentials(data_root: Path, model_root: Path) -> credentials.ServiceCredentials:
    """Issue this run's credentials and write them where the services look."""
    creds = credentials.resolve_for_run()
    run_copy = data_root / STAGE_STATE / credentials.CREDENTIALS_FILENAME
    service_copy = model_root / STAGE_STATE / credentials.CREDENTIALS_FILENAME

    credentials.save(creds, run_copy)
    try:
        credentials.save(creds, service_copy)
    except OSError as exc:
        # Non-fatal: the services fail closed on their own, and a clear log here
        # beats an exception that hides which of the two writes failed.
        log.error("could not write credentials to %s: %s", service_copy, exc)
    return creds


def handler(args: argparse.Namespace, cfg: SourcesConfig, data_root: Path) -> dict:
    source_indices = require_stage(data_root, STAGE_INDICES, args.project)

    # Resolved here, not at import: settings binds the environment once at module
    # load, which lands before runner.load_dotenv() and would make a value set in
    # .env silently fall back to the default.
    override = os.environ.get("PIPELINE_MODEL_ROOT")
    stage_root = Path(override) if override else data_root
    if override:
        log.info("staging into PIPELINE_MODEL_ROOT=%s instead of the output root", stage_root)
        if not stage_root.exists() and not on_fasttrack():
            log.info("%s does not exist — skipping staging (local run)", stage_root)
            return {"staged": False, "reason": f"{stage_root} not present"}

    require_writable(stage_root, "stage-service")

    code_root = stage_root / "docs-rag"
    files = stage_code(code_root)

    # The serving entrypoint resolves indices at <code>/pipeline/data/03_indices,
    # which is the same layout a local checkout uses — one code path, not two.
    served_indices = code_root / "pipeline" / "data" / STAGE_INDICES
    served_indices.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_indices, served_indices, dirs_exist_ok=True)

    definitions = stage_model_definitions(stage_root)
    creds = stage_credentials(data_root, stage_root)

    projects = sorted(p.name for p in served_indices.iterdir() if p.is_dir())
    log.info(
        "staged %d code files and %d index/indices (%s) into %s",
        files, len(projects), ", ".join(projects), stage_root,
    )

    # Printed last so it is the final thing in this node's log, where an
    # operator will actually look for it.
    log.info("%s", creds.banner("SERVICE ACCESS CREDENTIALS (this run)"))

    return {
        "staged": True,
        "stage_root": str(stage_root),
        "code_files": files,
        "projects": projects,
        "model_definitions_placed": definitions,
        "credentials_source": creds.source,
    }


if __name__ == "__main__":
    run_task("07_stage_service", handler, seed_stages=(STAGE_INDICES, STAGE_STATE))
