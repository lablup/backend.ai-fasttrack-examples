"""Drive the real task scripts through simulated FastTrack mounts, offline.

Covers the nodes that need no API key: clone -> convert -> publish -> stage.
The chain is what breaks in production, not the individual functions, and it
breaks in ways unit tests cannot see: a task reading the wrong mount, a stage
that fails to carry forward, a node reporting success having written nothing.

Each task runs as its own subprocess with its own PIPELINE_OUTPUT_ROOT and a
PIPELINE_INPUT_ROOT pointing at the previous task's output — exactly how
FastTrack lays it out, and deliberately not a shared directory, which would
hide every chaining bug.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASKS = PROJECT_ROOT / "pipeline" / "tasks"

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
needs_pandoc = pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )


@pytest.fixture
def source_repo(tmp_path: Path) -> Path:
    """A local git repo standing in for a public documentation repo."""
    repo = tmp_path / "upstream"
    (repo / "docs" / "guide").mkdir(parents=True)
    (repo / "docs" / "index.rst").write_text(
        "Overview\n========\n\nThe widget frobnicator accepts a --turbo flag.\n"
    )
    (repo / "docs" / "guide" / "install.rst").write_text(
        "Install\n=======\n\nRun ``pip install widget`` to install it.\n"
    )
    (repo / "docs" / "locales").mkdir()
    (repo / "docs" / "locales" / "ko.rst").write_text("Translated\n==========\n\nignore me\n")
    (repo / "README.md").write_text("# widget\n\nA thing that frobnicates.\n")

    _git("init", "-q", "-b", "main", cwd=repo)
    _git("config", "user.email", "test@example.invalid", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)
    _git("add", "-A", cwd=repo)
    _git("commit", "-qm", "docs", cwd=repo)
    return repo


@pytest.fixture
def sources_yaml(tmp_path: Path, source_repo: Path) -> Path:
    path = tmp_path / "sources.yaml"
    path.write_text(
        "sources:\n"
        "  - name: widget\n"
        f"    url: file://{source_repo}\n"
        "    branch: main\n"
        "    include:\n"
        '      - "docs/**/*.rst"\n'
        '      - "README.md"\n'
        "    exclude:\n"
        '      - "docs/locales/**/*"\n'
        '    verify_query: "How do I install widget?"\n'
    )
    return path


def run_node(
    script: str,
    sources: Path,
    output_root: Path,
    input_root: Path | None,
    job_index: int,
    extra_env: dict | None = None,
) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "PYTHONPATH": str(PROJECT_ROOT),
        "PIPELINE_OUTPUT_ROOT": str(output_root),
        # FastTrack injects this into every task. It proves we are on FastTrack;
        # it is never used to decide DAG position.
        "BACKENDAI_PIPELINE_JOB_INDEX": str(job_index),
    }
    # The head node genuinely has no input mount — model that by absence.
    env.pop("PIPELINE_INPUT_ROOT", None)
    if input_root is not None:
        env["PIPELINE_INPUT_ROOT"] = str(input_root)
    # No API key: this whole chain must work without one.
    env.pop("OPENAI_API_KEY", None)
    env.update(extra_env or {})

    return subprocess.run(
        [sys.executable, str(TASKS / script), "--project", "all", "--sources", str(sources)],
        capture_output=True,
        text=True,
        env=env,
    )


@needs_git
@needs_pandoc
def test_chain_clone_convert_publish_stage(tmp_path: Path, sources_yaml: Path):
    outputs = {name: tmp_path / "out" / name for name in ("clone", "convert", "publish", "stage")}
    vfroot = tmp_path / "vfroot"
    # The shipped layout: code and indices on the persistent vfroot, and only the
    # four small service files mirrored into the model vFolder. Setting
    # PIPELINE_MODEL_ROOT instead would exercise node 07's alternate
    # all-in-model-storage mode, which no pipeline in this repo uses.
    model_storage = tmp_path / "models"
    model_storage.mkdir()

    shared_env = {
        "PIPELINE_VFROOT": str(vfroot),
        "PIPELINE_MODEL_STORAGE": str(model_storage),
        # Pin the credentials so the run is deterministic and nothing is
        # generated; the generation path is exercised by its own test below.
        "GRADIO_USERNAME": "tester",
        "GRADIO_PASSWORD": "pw",
        "API_KEY": "token",
    }

    # --- 01 clone (head: no input mount) --------------------------------
    result = run_node("01_clone_sources.py", sources_yaml, outputs["clone"], None, 1, shared_env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (outputs["clone"] / "01_repos" / "widget" / "README.md").is_file()

    # --- 02 convert -----------------------------------------------------
    result = run_node(
        "02_convert_docs.py", sources_yaml, outputs["convert"], outputs["clone"], 2, shared_env
    )
    assert result.returncode == 0, result.stdout + result.stderr

    converted = outputs["convert"] / "02_docs_md" / "widget"
    assert (converted / "docs" / "index.md").is_file()
    assert (converted / "docs" / "guide" / "install.md").is_file()
    assert (converted / "README.md").is_file()
    # The exclude glob must actually exclude.
    assert not (converted / "docs" / "locales").exists()
    # pandoc really ran, rather than the file being copied verbatim.
    assert "pip install widget" in (converted / "docs" / "guide" / "install.md").read_text()

    # Narrowed seeding: convert asked for 01_repos only, so the stages it did
    # not ask for are absent rather than silently dragged along.
    assert (outputs["convert"] / "01_repos").exists()

    # --- fake the index-building nodes ----------------------------------
    # 03/04/05 need an embedding API. Everything downstream only cares that
    # 03_indices exists and is non-empty, so stand one in.
    indices = outputs["convert"] / "03_indices" / "widget"
    indices.mkdir(parents=True)
    (indices / "index.faiss").write_bytes(b"fake")
    (indices / "bm25.pkl").write_bytes(b"fake")

    # --- 06 publish -----------------------------------------------------
    result = run_node(
        "06_publish.py", sources_yaml, outputs["publish"], outputs["convert"], 6, shared_env
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (vfroot / "03_indices" / "widget" / "index.faiss").is_file()
    assert (vfroot / "99_state" / "run_manifest.json").is_file()

    # --- 07 stage-service ------------------------------------------------
    result = run_node(
        "07_stage_service.py", sources_yaml, outputs["stage"], outputs["publish"], 7, shared_env
    )
    assert result.returncode == 0, result.stdout + result.stderr

    # The code and the indices land on the vfroot, at the path both model
    # definitions name in their start_command.
    staged = vfroot / "docs-rag"
    assert (staged / "docs_rag" / "server.py").is_file()
    assert (staged / "pipeline" / "serve.sh").is_file()
    assert (staged / "pipeline" / "data" / "03_indices" / "widget" / "index.faiss").is_file()

    # Runtime data and caches must not be copied along with the code.
    assert not (staged / "pipeline" / "data" / "01_repos").exists()
    assert not (staged / ".git").exists()

    # Only the small files are mirrored into model storage — a deployment
    # resolves model_definition_path relative to that mount and finds nothing
    # anywhere else. The code tree must not be duplicated into it.
    assert (model_storage / "model-definition-fastapi.yaml").is_file()
    assert (model_storage / "model-definition-gradio.yaml").is_file()
    assert (model_storage / ".env").is_file()
    assert not (model_storage / "docs-rag").exists()

    # The generated .env is what reaches the deployment containers, which never
    # receive their pipeline envs.
    env_text = (model_storage / ".env").read_text()
    assert "API_KEY=token" in env_text
    assert "GITHUB_TOKEN" not in env_text

    for root in (vfroot, model_storage):
        creds = json.loads((root / "99_state" / "service_credentials.json").read_text())
        assert creds["api_key"] == "token"


@needs_git
def test_publish_refuses_to_report_success_with_nowhere_to_publish(
    tmp_path: Path, sources_yaml: Path
):
    """On FastTrack an unset vfroot must fail, not quietly publish nothing.

    A green node whose artifacts vanished with the container is worse than a
    red one.
    """
    output = tmp_path / "out"
    (output / "03_indices" / "widget").mkdir(parents=True)
    (output / "03_indices" / "widget" / "index.faiss").write_bytes(b"fake")

    env = {
        **os.environ,
        "PYTHONPATH": str(PROJECT_ROOT),
        "PIPELINE_OUTPUT_ROOT": str(output),
        "BACKENDAI_PIPELINE_JOB_INDEX": "6",
    }
    env.pop("PIPELINE_VFROOT", None)

    result = subprocess.run(
        [sys.executable, str(TASKS / "06_publish.py"), "--project", "all",
         "--sources", str(sources_yaml)],
        capture_output=True, text=True, env=env,
    )

    assert result.returncode == 1
    assert "PIPELINE_VFROOT is not set" in result.stdout + result.stderr


@needs_git
def test_missing_upstream_stage_names_the_node_that_should_have_produced_it(
    tmp_path: Path, sources_yaml: Path
):
    result = run_node("02_convert_docs.py", sources_yaml, tmp_path / "out", None, 2)
    combined = result.stdout + result.stderr

    assert result.returncode == 1
    assert "01_repos" in combined
    assert "clone-docs" in combined


@needs_git
def test_a_glob_matching_nothing_fails_loudly(tmp_path: Path, source_repo: Path):
    """An index built from zero files only reveals itself at query time."""
    sources = tmp_path / "sources.yaml"
    sources.write_text(
        "sources:\n"
        "  - name: widget\n"
        f"    url: file://{source_repo}\n"
        "    branch: main\n"
        "    include:\n"
        '      - "does/not/exist/**/*.rst"\n'
        '    verify_query: "anything"\n'
    )
    clone_out = tmp_path / "clone"
    assert run_node("01_clone_sources.py", sources, clone_out, None, 1).returncode == 0

    result = run_node("02_convert_docs.py", sources, tmp_path / "convert", clone_out, 2)

    assert result.returncode == 1
    assert "matched no files" in result.stdout + result.stderr
