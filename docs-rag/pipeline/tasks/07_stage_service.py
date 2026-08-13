#!/usr/bin/env python3
"""Node 07 — stage everything the two services need into model storage.

In:   <data-root>/{03_indices,99_state}/ + this checkout
Out:  /models/
        docs-rag/                     this code tree (minus data and caches)
        docs-rag/pipeline/data/03_indices/   the indices to serve
        99_state/service_credentials.json    login for both services
        model-definition-{fastapi,gradio}.yaml   if not already present

Why model storage and not /pipeline/outputs: a deployment container is not part
of the task chain and gets no /pipeline mounts at all. Model storage is the one
place both container kinds can see — read-write here, read-only over there. It
is chosen in the Create Pipeline dialog rather than named in the YAML, which is
what keeps this pipeline portable across accounts.

Staging the code too means the services start with no network access and no git.

    python pipeline/tasks/07_stage_service.py --project all
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from docs_rag import credentials
from docs_rag.settings import MODEL_ROOT
from pipeline.io import SourcesConfig
from pipeline.paths import PROJECT_ROOT, STAGE_INDICES, STAGE_STATE
from pipeline.preflight import on_fasttrack, require_stage, require_writable
from pipeline.runner import run_task

log = logging.getLogger("07_stage_service")

# Never copy these into model storage: runtime data (large, and regenerated),
# build caches, and anything that could carry a secret.
CODE_EXCLUDES = shutil.ignore_patterns(
    "data", ".git", "__pycache__", "*.pyc", ".venv*", ".env", "vfroot", ".pytest_cache",
)

MODEL_DEFINITIONS = ("model-definition-fastapi.yaml", "model-definition-gradio.yaml")


def stage_code(destination: Path) -> int:
    """Copy this checkout into model storage, replacing any previous copy."""
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
            log.info("%s already present in model storage — leaving it alone", name)
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

    model_root = MODEL_ROOT
    if not model_root.exists() and not on_fasttrack():
        log.info("%s does not exist — skipping staging (local run)", model_root)
        return {"staged": False, "reason": f"{model_root} not present"}

    require_writable(model_root, "stage-service")

    code_root = model_root / "docs-rag"
    files = stage_code(code_root)

    # The serving entrypoint resolves indices at <code>/pipeline/data/03_indices,
    # which is the same layout a local checkout uses — one code path, not two.
    served_indices = code_root / "pipeline" / "data" / STAGE_INDICES
    served_indices.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_indices, served_indices, dirs_exist_ok=True)

    definitions = stage_model_definitions(model_root)
    creds = stage_credentials(data_root, model_root)

    projects = sorted(p.name for p in served_indices.iterdir() if p.is_dir())
    log.info(
        "staged %d code files and %d index/indices (%s) into %s",
        files, len(projects), ", ".join(projects), model_root,
    )

    # Printed last so it is the final thing in this node's log, where an
    # operator will actually look for it.
    log.info("%s", creds.banner("SERVICE ACCESS CREDENTIALS (this run)"))

    return {
        "staged": True,
        "model_root": str(model_root),
        "code_files": files,
        "projects": projects,
        "model_definitions_placed": definitions,
        "credentials_source": creds.source,
    }


if __name__ == "__main__":
    run_task("07_stage_service", handler, seed_stages=(STAGE_INDICES, STAGE_STATE))
