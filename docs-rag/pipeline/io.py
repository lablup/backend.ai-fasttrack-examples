"""Pydantic IO contracts — SSoT for sources.yaml and the run-state schemas.

Everything written under `<data-root>/99_state/` is one of these models, so the
report shapes are validated on write and readable without guesswork.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

log = logging.getLogger("sources")


def _resolves_inside(path: Path, root: Path) -> bool:
    """True when `path`, with every symlink followed, stays under `root`."""
    try:
        return path.resolve(strict=True).is_relative_to(root)
    except (OSError, ValueError):
        # A broken link, a loop, or an unreadable path is not ingestible either.
        return False


class SourceSpec(BaseModel):
    """One git-based documentation source.

    `include`/`exclude` are pathlib glob patterns relative to the repo root.
    They replace a single "docs subdirectory" field because the real repos do
    not agree on where documentation lives: two keep it under `docs/` alongside
    translation catalogs that must be skipped, one splits it between `docs/`
    and the repo root, and one buries 42 benchmark dumps in `docs/`.
    """

    name: str
    url: str
    branch: str = "main"
    include: List[str] = Field(default_factory=lambda: ["docs/**/*.rst", "docs/**/*.md"])
    exclude: List[str] = Field(default_factory=list)
    verify_query: str = ""

    def select_files(self, repo_root: Path) -> List[Path]:
        """Files to ingest, as absolute paths, include minus exclude.

        Deterministically ordered so a re-run chunks documents identically.

        Anything resolving outside the repo is dropped. `Path.glob` returns
        symlinks and `is_file()` follows them, so without this a symlink
        committed to a source repo would be read *through* and its target's
        contents copied into the corpus, embedded, and answered from. The
        ingestion container has the operator's model storage mounted, so
        `docs/faq.md -> /models/.env` would put an API key in the index — and
        the source repos are third-party by design, which is the whole point of
        the tool. Legitimate symlinks pointing inside the repo still work.
        """
        root = repo_root.resolve()
        selected: set[Path] = set()
        for pattern in self.include:
            selected.update(p for p in repo_root.glob(pattern) if p.is_file())
        for pattern in self.exclude:
            selected.difference_update(p for p in repo_root.glob(pattern) if p.is_file())

        inside, escaping = [], []
        for path in selected:
            (inside if _resolves_inside(path, root) else escaping).append(path)
        if escaping:
            # Loud, not silent: this is either an attack or a broken repo, and
            # both are worth an operator's attention.
            log.warning(
                "%s: refusing %d file(s) that resolve outside the repository: %s",
                self.name, len(escaping),
                ", ".join(sorted(str(p.relative_to(repo_root)) for p in escaping)[:10]),
            )
        return sorted(inside)


class SourcesConfig(BaseModel):
    sources: List[SourceSpec]

    @classmethod
    def load(cls, path: Path) -> "SourcesConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def names(self) -> List[str]:
        return [s.name for s in self.sources]


class TaskResult(BaseModel):
    """One row appended to 99_state/run_manifest.json per task run."""

    task: str
    project: Optional[str] = None
    started_at: datetime
    finished_at: datetime
    success: bool
    details: Dict[str, object] = Field(default_factory=dict)
    error: Optional[str] = None


class VerifyProjectResult(BaseModel):
    project: str
    query: str
    passed: bool
    top_l2: Optional[float] = None
    top_source: Optional[str] = None
    lexical_hits: Optional[int] = None
    error: Optional[str] = None


class VerifyReport(BaseModel):
    """Node 04 output — 99_state/verify_report.json."""

    generated_at: datetime
    l2_threshold: float
    results: Dict[str, VerifyProjectResult]


class EvalQuestionResult(BaseModel):
    """One scored question in node 05's eval_report.json."""

    question: str
    project: str = ""
    source: str = ""
    top_l2: Optional[float] = None
    retrieved_chunks: int = 0
    response_chars: int = 0
    metrics: Dict[str, float] = Field(default_factory=dict)
    overall: float = 0.0
    error: Optional[str] = None


class EvalReport(BaseModel):
    """Node 05 output — 99_state/eval_report.json."""

    generated_at: datetime
    rag_model: str
    judge_model: str
    sample_size: int
    projects: List[str]
    aggregate: Dict[str, float] = Field(default_factory=dict)
    results: List[EvalQuestionResult] = Field(default_factory=list)
