"""Startup checks and evidence logging shared by every pipeline task.

Two jobs, both aimed at the same failure mode: a node that dies confusingly —
or worse, exits 0 having done nothing — because a precondition it never checked
was not met.

- `log_environment()` prints what the container actually saw (checkout commit,
  resolved roots and their contents, chaining state, secret presence) before any
  work starts. `runner.run_task()` calls it, so every node gets it for free.
- `require_*()` turn a missing precondition into an actionable message naming
  the node that should have produced the input, instead of a stack trace from
  deep inside the work.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from pipeline.paths import PROJECT_ROOT

log = logging.getLogger("preflight")

# Which node produces each stage — makes "missing input" messages actionable.
STAGE_PRODUCER = {
    "01_repos": "clone-docs (01_clone_sources)",
    "02_docs_md": "convert-docs (02_convert_docs)",
    "03_indices": "build-indices (03_build_indices)",
    "99_state": "verify-indices (04) / evaluate (05)",
}

# Logged by presence and length only — never by value.
_SECRETS = ("OPENAI_API_KEY", "GITHUB_TOKEN")


def on_fasttrack() -> bool:
    """True when running as a FastTrack task node.

    BACKENDAI_PIPELINE_JOB_INDEX is injected at runtime and is the one variable
    FastTrack reliably provides. It says *that* we are on FastTrack; it does not
    reliably say *where* in the DAG we are, so never branch on its value.
    """
    return bool(os.environ.get("BACKENDAI_PIPELINE_JOB_INDEX"))


def _git_head() -> str:
    """One-line provenance of this checkout, or why it cannot be determined."""
    try:
        proc = subprocess.run(
            ["git", "log", "-1", "--format=%h %ad %s", "--date=short"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "<git unavailable>"
    if proc.returncode != 0:
        return "<not a git working tree — provenance unknown>"
    return proc.stdout.strip()


def _describe_root(label: str, path: Path | None) -> str:
    # Never raises: this runs before everything, including the checks, and a
    # mount that is present but unreadable is itself a diagnosis worth printing.
    if path is None:
        return f"{label}=<unset>"
    try:
        if not path.exists():
            return f"{label}={path} exists=False"
        entries = sorted(c.name for c in path.iterdir()) if path.is_dir() else []
        return (
            f"{label}={path} exists=True writable={os.access(path, os.W_OK)} "
            f"entries={entries}"
        )
    except OSError as exc:
        return f"{label}={path} UNREADABLE ({type(exc).__name__}: {exc})"


def log_environment(
    task: str,
    data_root: Path,
    input_root: Path | None = None,
    vfroot: Path | None = None,
) -> None:
    """Log the full runtime picture before a task does any work."""
    log.info("task=%s args=%s", task, " ".join(sys.argv[1:]) or "<none>")
    log.info("cwd=%s python=%s", Path.cwd(), sys.executable)
    log.info("code=%s checkout=%s", PROJECT_ROOT, _git_head())
    log.info(
        "fasttrack=%s job_index=%s",
        on_fasttrack(),
        os.environ.get("BACKENDAI_PIPELINE_JOB_INDEX", "<unset>"),
    )
    log.info("%s", _describe_root("output_root", data_root))
    log.info("%s", _describe_root("input_root", input_root))
    log.info("%s", _describe_root("vfroot", vfroot))
    for name in _SECRETS:
        value = os.environ.get(name, "")
        log.info("%s=%s", name, f"<set, {len(value)} chars>" if value else "<UNSET>")


def require_env(*names: str) -> None:
    """Fail with an actionable message if any named env var is unset or empty."""
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise SystemExit(
            f"missing required env var(s): {', '.join(missing)}.\n"
            "  Set them in your .env. On FastTrack that means uploading .env to "
            "the model-storage folder (it mounts at /models), or defining them as "
            "GUI secrets and referencing them from the pipeline YAML's `envs` "
            "block with ${{ secrets.NAME }}."
        )


def require_binary(name: str) -> str:
    """Fail if a required external binary is not on PATH."""
    path = shutil.which(name)
    if not path:
        raise SystemExit(
            f"required binary '{name}' not found on PATH. pipeline/bootstrap.sh "
            "provisions the task venv (including pandoc) — was this task started "
            "through it?"
        )
    log.info("binary %s -> %s", name, path)
    return path


def require_stage(data_root: Path, stage: str, project: str | None = None) -> Path:
    """Fail if an upstream stage directory is missing or empty.

    Both branches carry the same guidance: `run_task` calls `ensure_dirs()`
    before the handler, so in a real DAG run a stage that was never produced
    shows up as *empty*, not missing. Putting the FastTrack hint only on the
    missing branch would make it dead text for every in-DAG caller.
    """
    # "all" is the CLI's whole-pipeline sentinel — check the stage root then.
    # Otherwise scope to the project, so `--project X` is not satisfied by some
    # other source having populated the stage.
    scoped = project not in (None, "all")
    path = data_root / stage / project if scoped else data_root / stage
    producer = STAGE_PRODUCER.get(stage, "an upstream node")
    hint = (
        f" Produced by {producer}. On FastTrack this usually means "
        "PIPELINE_INPUT_ROOT is unset or the upstream task wrote nothing."
    )
    if not path.exists():
        raise SystemExit(f"missing input {path}.{hint}")
    if path.is_dir() and not any(path.iterdir()):
        raise SystemExit(f"input {path} is empty.{hint}")
    return path


def require_writable(path: Path, what: str) -> Path:
    """Fail if a directory we must write to is missing or read-only.

    Used for the /models mount: model storage is read-only inside a serving
    container and read-write in a batch task, and the difference is invisible
    until a copy silently fails.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SystemExit(
            f"{what}: cannot create {path} ({type(exc).__name__}: {exc}).\n"
            "  On FastTrack this is the model-storage mount. Select or create a "
            "model storage folder in the Create Pipeline dialog."
        ) from exc
    if not os.access(path, os.W_OK):
        raise SystemExit(
            f"{what}: {path} exists but is not writable.\n"
            "  Model storage mounts read-only in serving containers and "
            "read-write in batch tasks — a read-only mount here means this node "
            "is not the batch task it is supposed to be."
        )
    return path
