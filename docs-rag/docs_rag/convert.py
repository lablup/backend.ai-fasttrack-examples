"""Normalise a repo's documentation into a flat tree of markdown.

`.rst` goes through pandoc; `.md` is copied through. Routing is by extension,
so a repo can mix both without declaring anything — and a markdown-native
source is not silently skipped by an RST-only converter.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

log = logging.getLogger("convert")

# tiktoken raises on these if they appear in text to be embedded. They occur in
# documentation that discusses prompt formats, which is exactly the kind of page
# we want to index.
SPECIAL_TOKENS = (
    "<|endoftext|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|endofprompt|>",
)

PANDOC_ARGS = ("-f", "rst", "-t", "gfm", "--wrap=none")


def sanitize(text: str) -> str:
    for token in SPECIAL_TOKENS:
        text = text.replace(token, token.replace("<|", "< |"))
    return text


def _output_path(source: Path, repo_root: Path, out_root: Path) -> Path:
    """Mirror the repo's directory structure, with an `.md` extension."""
    relative = source.relative_to(repo_root)
    return (out_root / relative).with_suffix(".md")


async def _run_pandoc(source: Path, destination: Path) -> None:
    proc = await asyncio.create_subprocess_exec(
        "pandoc", *PANDOC_ARGS, str(source),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"pandoc failed on {source} (exit {proc.returncode}): "
            f"{stderr.decode('utf-8', 'replace').strip()[:400]}"
        )
    # Record where this came from. The indexer keeps the relative path, but the
    # original extension is the only clue that a page was RST upstream.
    header = f"---\nsource_file: {source.name}\n---\n\n"
    destination.write_text(header + sanitize(stdout.decode("utf-8", "replace")), encoding="utf-8")


async def convert_files(
    files: Iterable[Path],
    repo_root: Path,
    out_root: Path,
) -> Tuple[int, List[str]]:
    """Convert or copy each file into `out_root`.

    Returns the number of files written and a list of human-readable failures.
    One unconvertible page must not abort a 100-page corpus, so failures are
    collected and reported by the caller.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    written = 0
    failures: List[str] = []

    for source in files:
        destination = _output_path(source, repo_root, out_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            if source.suffix.lower() == ".rst":
                await _run_pandoc(source, destination)
            else:
                text = sanitize(source.read_text(encoding="utf-8"))
                destination.write_text(text, encoding="utf-8")
            written += 1
        except (OSError, UnicodeDecodeError, RuntimeError) as exc:
            failures.append(f"{source.name}: {type(exc).__name__}: {exc}")

    return written, failures


def reset_output(out_root: Path) -> None:
    """Clear a project's converted output so a re-run cannot leave orphans.

    Without this, a page deleted upstream keeps being indexed forever.
    """
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
