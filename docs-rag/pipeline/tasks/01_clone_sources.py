#!/usr/bin/env python3
"""Node 01 — clone or update each documentation source.

In:   pipeline/config/sources.yaml
Out:  <data-root>/01_repos/<name>/

Idempotent. First run shallow-clones; later runs fetch and hard-reset to the
remote tip, so a re-run always produces the same tree as a fresh clone.

    python pipeline/tasks/01_clone_sources.py --project all
    python pipeline/tasks/01_clone_sources.py --project bssh
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Allow `from pipeline.X import ...` when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.io import SourceSpec, SourcesConfig
from pipeline.paths import repos_dir
from pipeline.preflight import require_binary
from pipeline.runner import run_task, select_sources

log = logging.getLogger("01_clone_sources")

# Mask "https://x-access-token:<token>@github.com/..." in anything we log.
_TOKEN_RE = re.compile(r"https://[^@/\s]+@")


def _auth_url(url: str) -> str:
    """Inject GITHUB_TOKEN into an HTTPS github.com URL, for private sources.

    The token is read from the environment at call time so sources.yaml stays
    token-free and safe to commit. All five default sources are public and do
    not need this.
    """
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token and url.startswith("https://github.com/"):
        return url.replace("https://", f"https://x-access-token:{token}@", 1)
    return url


def _git(*args: str, cwd: Path | None = None) -> None:
    cmd = ["git", *args]
    log.info("$ %s%s", _TOKEN_RE.sub("https://***@", " ".join(cmd)),
             f"  (cwd={cwd})" if cwd else "")
    # GIT_TERMINAL_PROMPT=0 turns a missing credential into an immediate error.
    # Without it, a private repo with no token hangs forever on an interactive
    # "Username for 'https://github.com':" prompt that nothing will ever answer.
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def _fresh_clone(spec: SourceSpec, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    _git("clone", "--depth", "1", "--branch", spec.branch, _auth_url(spec.url), str(target))
    # Do not leave a token baked into .git/config; it is re-injected per fetch.
    _git("remote", "set-url", "origin", spec.url, cwd=target)


def clone_or_update(spec: SourceSpec, target: Path) -> dict:
    if (target / ".git").exists():
        try:
            _git("fetch", "--depth", "1", _auth_url(spec.url), spec.branch, cwd=target)
            # `checkout -B <branch> FETCH_HEAD`, not `reset --hard origin/<branch>`:
            # a branch-scoped fetch populates FETCH_HEAD but does not necessarily
            # move the remote-tracking ref, so resetting to it can silently keep
            # the old commit.
            _git("checkout", "-B", spec.branch, "FETCH_HEAD", cwd=target)
            action = "updated"
        except subprocess.CalledProcessError:
            log.warning("%s: update failed, re-cloning from scratch", spec.name)
            _fresh_clone(spec, target)
            action = "re-cloned"
    else:
        _fresh_clone(spec, target)
        action = "cloned"

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=target, check=True, capture_output=True, text=True,
    ).stdout.strip()
    return {"action": action, "head_sha": sha}


def handler(args: argparse.Namespace, cfg: SourcesConfig, data_root: Path) -> dict:
    require_binary("git")
    sources = select_sources(cfg, args.project)
    log.info("cloning %d source(s): %s", len(sources), ", ".join(s.name for s in sources))

    summary: dict = {}
    for spec in sources:
        target = repos_dir(data_root, spec.name)
        log.info("%s: %s (branch %s) -> %s", spec.name, spec.url, spec.branch, target)
        info = clone_or_update(spec, target)
        summary[spec.name] = f"{info['action']}@{info['head_sha'][:8]}"
    return {"sources": summary, "count": len(summary)}


if __name__ == "__main__":
    # seed_stages=() — this node consumes no upstream stage. Declaring that here
    # makes head-ness a fact about our own DAG rather than an assumption that
    # the scheduler mounts the head no /pipeline/input1: if a stale or empty
    # input mount ever does appear, there is nothing for it to copy forward.
    run_task("01_clone_sources", handler, seed_stages=())
