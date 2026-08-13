"""Path resolution for the pipeline — the single source of truth.

Directory convention, under whichever data root a task resolves:

    <data-root>/01_repos/<source>/      node 01 output (cloned repos)
    <data-root>/02_docs_md/<source>/    node 02 output (converted markdown)
    <data-root>/03_indices/<source>/    node 03 output (FAISS + BM25)
    <data-root>/99_state/               reports, manifests, credentials

On FastTrack each task writes `/pipeline/outputs` and reads the previous task's
output at `/pipeline/input1`. `runner.seed_from_input()` copies the input root
forward so the output stays cumulative down the chain. `/pipeline/vfroot` is the
persistent folder shared by every task; it survives the run.

Environment variables, highest priority first:

    --data-root CLI arg     overrides everything (writes)
    PIPELINE_OUTPUT_ROOT    where this task writes      (/pipeline/outputs)
    PIPELINE_DATA_ROOT      single-root fallback        (./pipeline/data)
    PIPELINE_INPUT_ROOT     previous task's output      (/pipeline/input1)
    PIPELINE_VFROOT         persistent shared folder    (/pipeline/vfroot)

Locally none of these are set: output root == input root == ./pipeline/data, no
seeding happens, and every node reads and writes one shared tree.
"""

from __future__ import annotations

import os
from pathlib import Path

# Model storage (/models) is defined in docs_rag.settings, not here: the serving
# apps need it and docs_rag must not import pipeline. Import it from there.

PIPELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_ROOT.parent
DEFAULT_DATA_ROOT = PIPELINE_ROOT / "data"
SOURCES_YAML = PIPELINE_ROOT / "config" / "sources.yaml"
PIPELINE_YAML = PIPELINE_ROOT / "config" / "pipeline.yaml"
EVAL_SAMPLES_JSON = PIPELINE_ROOT / "config" / "eval_samples.json"

STAGE_REPOS = "01_repos"
STAGE_DOCS_MD = "02_docs_md"
STAGE_INDICES = "03_indices"
STAGE_STATE = "99_state"


def resolve_output_root(cli_value: str | None = None) -> Path:
    """The data root this task writes to.

    Priority: --data-root > PIPELINE_OUTPUT_ROOT > PIPELINE_DATA_ROOT >
    ./pipeline/data.
    """
    if cli_value:
        return Path(cli_value).resolve()
    env = os.environ.get("PIPELINE_OUTPUT_ROOT") or os.environ.get("PIPELINE_DATA_ROOT")
    if env:
        return Path(env).resolve()
    return DEFAULT_DATA_ROOT.resolve()


# The single-root resolver used by the serving entrypoints, which write nothing.
resolve_data_root = resolve_output_root


def resolve_input_root() -> Path | None:
    """Previous task's output, mounted by FastTrack at /pipeline/input1.

    None when unset — local runs, and the head node, which FastTrack gives no
    input mount because it declares no dependency.
    """
    env = os.environ.get("PIPELINE_INPUT_ROOT")
    return Path(env).resolve() if env else None


def resolve_vfroot() -> Path | None:
    """Persistent shared folder (/pipeline/vfroot). None when unset."""
    env = os.environ.get("PIPELINE_VFROOT")
    return Path(env).resolve() if env else None


def stage_dir(data_root: Path, stage: str, project: str | None = None) -> Path:
    base = data_root / stage
    return base / project if project else base


def repos_dir(data_root: Path, project: str | None = None) -> Path:
    return stage_dir(data_root, STAGE_REPOS, project)


def docs_md_dir(data_root: Path, project: str | None = None) -> Path:
    return stage_dir(data_root, STAGE_DOCS_MD, project)


def indices_dir(data_root: Path, project: str | None = None) -> Path:
    return stage_dir(data_root, STAGE_INDICES, project)


def state_dir(data_root: Path) -> Path:
    return stage_dir(data_root, STAGE_STATE)


def manifest_path(data_root: Path) -> Path:
    return state_dir(data_root) / "run_manifest.json"


def verify_report_path(data_root: Path) -> Path:
    return state_dir(data_root) / "verify_report.json"


def eval_report_path(data_root: Path) -> Path:
    return state_dir(data_root) / "eval_report.json"


def ensure_dirs(data_root: Path) -> None:
    for stage in (STAGE_REPOS, STAGE_DOCS_MD, STAGE_INDICES, STAGE_STATE):
        (data_root / stage).mkdir(parents=True, exist_ok=True)
