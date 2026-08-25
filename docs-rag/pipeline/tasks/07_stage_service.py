#!/usr/bin/env python3
"""Node 07 — arrange everything the two services need in the persistent folder.

In:   <data-root>/{03_indices,99_state}/ + this checkout
Out:  $PIPELINE_VFROOT/
        docs-rag/                     this code tree (minus data and caches)
        docs-rag/pipeline/data/03_indices/   the indices to serve
        99_state/service_credentials.json    login for both services
        model-definition-{fastapi,gradio}.yaml   read by the deployment nodes

Stage into /pipeline/vfroot, not /pipeline/outputs. A task's output mount is
scoped to the task chain: a deployment container can read it once — fetching its
model definition works — but the handle goes stale when the upstream container
ends, and a service that keeps reading its code and indices then dies on ESTALE.
The vfroot is the one /pipeline mount that outlives the run.

Set PIPELINE_MODEL_ROOT to stage into model storage instead. The layout is
identical either way, so only the definitions' start_command has to agree.

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
from docs_rag.settings import Settings, is_unset
from pipeline.io import SourcesConfig
from pipeline.paths import PROJECT_ROOT, STAGE_INDICES, STAGE_STATE, resolve_vfroot
from pipeline.preflight import on_fasttrack, require_stage, require_writable
from pipeline.runner import run_task

log = logging.getLogger("07_stage_service")

# Never stage these: runtime data (large, and regenerated),
# build caches, and anything that could carry a secret.
CODE_EXCLUDES = shutil.ignore_patterns(
    "data", ".git", "__pycache__", "*.pyc", ".venv*", ".env", "vfroot", ".pytest_cache",
)

MODEL_DEFINITIONS = ("model-definition-fastapi.yaml", "model-definition-gradio.yaml")

# Everything the services read, and nothing else. Deliberately derived from the
# settings model rather than listed by hand, so a new setting cannot be added
# without reaching the deployment. GITHUB_TOKEN and the PIPELINE_* paths are
# excluded by construction — a serving container has no use for either.
SERVICE_ENV_NAMES = tuple(name.upper() for name in Settings.model_fields)
SECRET_ENV_NAMES = frozenset({"OPENAI_API_KEY", "API_KEY", "GRADIO_PASSWORD"})


def stage_code(destination: Path) -> int:
    """Copy this checkout to the staging root, replacing any previous copy."""
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(PROJECT_ROOT, destination, ignore=CODE_EXCLUDES)
    return sum(1 for p in destination.rglob("*") if p.is_file())


def stage_model_definitions(model_root: Path, replace: bool = False) -> list[str]:
    """Place the default model definitions, without clobbering custom ones.

    An operator who uploaded their own tuned definition keeps it; one who
    uploaded nothing still gets a working service instead of a start failure.

    `replace` overrides that for model storage, which the pipeline owns rather
    than the operator: the deployment reads its definition from there, so a copy
    left over from an earlier run silently pins the old port or start command.
    """
    placed = []
    for name in MODEL_DEFINITIONS:
        target = model_root / name
        if target.exists() and not replace:
            log.info("%s already present at the staging root — leaving it alone", name)
            continue
        source = PROJECT_ROOT / name
        if not source.is_file():
            log.warning("%s missing from the checkout — cannot stage it", name)
            continue
        shutil.copy2(source, target)
        placed.append(name)
    return placed


def mirror_to_model_storage(stage_root: Path) -> list[str]:
    """Copy the small service files into model storage as well.

    FastTrack resolves a deployment's model_definition_path relative to the
    model mount, so an absolute path into /pipeline/vfroot becomes
    /models/pipeline/vfroot/... and is never found. The definition therefore has
    to exist under /models, whatever else lives elsewhere.

    Only the small files are mirrored. The code tree and the indices stay on the
    vfroot, which serving containers read for the whole of their lifetime — there
    is no reason to duplicate hundreds of megabytes into a second mount.

    Non-fatal when model storage is absent: a batch-only run is legitimate, and
    the deployment nodes report the missing definition clearly enough.
    """
    # Its own variable, not PIPELINE_MODEL_ROOT: that one moves the staging root,
    # while this is only where the small files are mirrored. A batch task mounts
    # the model vfolder by name, which lands it at /home/work/<name> rather than
    # at /models, so the two mounts of the same folder need separate paths.
    model_root = Path(os.environ.get("PIPELINE_MODEL_STORAGE", "/models"))
    if model_root == stage_root:
        log.info("staging root is model storage — nothing to mirror")
        return []
    if not model_root.is_dir() or not os.access(model_root, os.W_OK):
        message = (
            f"{model_root} is not mounted writable, so the model definitions "
            "cannot be written. The deployment nodes resolve "
            "model_definition_path relative to this mount and would fall back "
            "to the default runtime. Add the model vfolder to this task's "
            "mounts: list and point PIPELINE_MODEL_STORAGE at it."
        )
        if on_fasttrack():
            # Failing here names the wiring error. Succeeding would surface it
            # as two unrelated-looking deployment failures further on.
            raise SystemExit(message)
        log.warning("%s Skipping the copy (local run).", message)
        return []

    mirrored = stage_model_definitions(model_root, replace=True)
    for relative in (Path(".env"), Path(STAGE_STATE) / credentials.CREDENTIALS_FILENAME):
        source = stage_root / relative
        if not source.is_file():
            continue
        target = model_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        target.chmod(0o600)
        mirrored.append(str(relative))

    log.info("mirrored into %s: %s", model_root, ", ".join(mirrored))
    return mirrored


def stage_env(stage_root: Path) -> list[str]:
    """Write this run's service configuration to <stage_root>/.env.

    FastTrack does not deliver a deployment node's `envs` to its container, so a
    file on a shared mount is the only channel that reaches the services. Without
    it they start with no OPENAI_API_KEY and no login, and the failure surfaces as
    a service that boots and then answers nothing.

    Rewritten every run: this file describes the run that produced the indices
    beside it, and a stale copy claiming a rotated key is worse than none.
    """
    target = stage_root / ".env"
    lines = [
        "# Generated by the stage-service node — do not edit.",
        "# FastTrack does not pass a deployment node's envs to its container, so",
        "# the services read their configuration from here.",
    ]
    written = []
    for name in SERVICE_ENV_NAMES:
        value = os.environ.get(name)
        # A newline would silently truncate the file into a broken KEY=VALUE.
        if is_unset(value) or "\n" in value:
            continue
        lines.append(f"{name}={value}")
        written.append(name)

    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    target.chmod(0o600)
    shown = [n if n not in SECRET_ENV_NAMES else f"{n}=<{len(os.environ[n])} chars>"
             for n in written]
    log.info("wrote %s with %d setting(s): %s", target, len(written), ", ".join(shown))
    return written


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
    if override:
        stage_root = Path(override)
        log.info("staging into PIPELINE_MODEL_ROOT=%s", stage_root)
        if not stage_root.exists() and not on_fasttrack():
            log.info("%s does not exist — skipping staging (local run)", stage_root)
            return {"staged": False, "reason": f"{stage_root} not present"}
    else:
        vfroot = resolve_vfroot()
        if vfroot is None:
            if on_fasttrack():
                raise SystemExit(
                    "PIPELINE_VFROOT is not set, but this is a FastTrack run. The "
                    "services read their code and indices from /pipeline/vfroot for "
                    "the whole of their lifetime — /pipeline/outputs goes stale when "
                    "this container ends. Set PIPELINE_VFROOT=/pipeline/vfroot."
                )
            log.info("PIPELINE_VFROOT unset — staging into the output root (local run)")
        stage_root = vfroot if vfroot is not None else data_root

    stage_root.mkdir(parents=True, exist_ok=True)
    require_writable(stage_root, "stage-service")

    code_root = stage_root / "docs-rag"
    files = stage_code(code_root)

    # The serving entrypoint resolves indices at <code>/pipeline/data/03_indices,
    # which is the same layout a local checkout uses — one code path, not two.
    served_indices = code_root / "pipeline" / "data" / STAGE_INDICES
    served_indices.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_indices, served_indices, dirs_exist_ok=True)

    definitions = stage_model_definitions(stage_root)
    env_written = stage_env(stage_root)
    creds = stage_credentials(data_root, stage_root)

    mirrored = mirror_to_model_storage(stage_root)

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
        "env_settings_written": len(env_written),
        "mirrored_to_model_storage": mirrored,
        "credentials_source": creds.source,
    }


if __name__ == "__main__":
    run_task("07_stage_service", handler, seed_stages=(STAGE_INDICES, STAGE_STATE))
