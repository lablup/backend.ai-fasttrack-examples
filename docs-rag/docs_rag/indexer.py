"""Build FAISS + BM25 indices from a tree of markdown files.

Input layout (what node 02 produces):

    <docs_root>/<project>/**/*.md

Output layout (what node 03 produces, and what `Retriever` loads):

    <indices_root>/<project>/{index.faiss, index.pkl, bm25.pkl}
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import tiktoken
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from docs_rag.bm25 import BM25Store
from docs_rag.embeddings import make_embeddings
from docs_rag.settings import Settings

log = logging.getLogger("indexer")

# Skip files with less prose than this. A stub page contributes a chunk that
# matches everything weakly and nothing well.
MIN_CONTENT_CHARS = 120

# Documents per FAISS.from_documents call. Larger batches hit the embedding
# API's per-request size limit on documentation-sized chunks.
EMBED_BATCH_SIZE = 50


def extract_title(content: str, fallback: str) -> str:
    """First markdown heading, else the filename stem."""
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("#"):
            title = line.lstrip("#").strip()
            if title:
                return title
    return fallback


class DocumentIndexer:
    """Chunk a markdown tree and build one index pair per project."""

    def __init__(self, docs_root: Path, indices_root: Path, settings: Settings):
        self.docs_root = Path(docs_root)
        self.indices_root = Path(indices_root)
        self.settings = settings
        self.embeddings = make_embeddings(settings)

        # Chunk by tokens, not characters: the chunk size that matters is the
        # one the embedding model and the LLM context window see.
        encoding = tiktoken.get_encoding("o200k_base")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=lambda text: len(encoding.encode(text)),
            separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
        )

    def collect_documents(self, projects: List[str]) -> Dict[str, List[Document]]:
        """Chunk every markdown file under each named project.

        `projects` is a parameter rather than a scan of `docs_root` so a partial
        run (`--project bssh`) cannot silently re-index whatever else happens to
        be lying in the tree.
        """
        project_docs: Dict[str, List[Document]] = {}

        for project in projects:
            project_dir = self.docs_root / project
            if not project_dir.is_dir():
                log.warning("no converted docs for project %s at %s", project, project_dir)
                continue

            docs: List[Document] = []
            files = 0
            skipped = 0

            for file_path in sorted(project_dir.rglob("*.md")):
                try:
                    content = file_path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError) as exc:
                    log.warning("unreadable %s: %s", file_path, exc)
                    continue

                if len(content.strip()) < MIN_CONTENT_CHARS:
                    skipped += 1
                    continue

                files += 1
                relative_path = str(file_path.relative_to(project_dir))
                title = extract_title(content, file_path.stem)
                last_updated = datetime.fromtimestamp(file_path.stat().st_mtime)

                chunks = self.splitter.split_text(content)
                for chunk_index, chunk in enumerate(chunks):
                    # A retrieved chunk arrives at the LLM with no surrounding
                    # document, so the header is what tells it which project and
                    # page the text came from — and lets the answer cite it.
                    header = f"[{project} | {title} | {relative_path}]\n\n"
                    docs.append(
                        Document(
                            page_content=header + chunk,
                            metadata={
                                "project": project,
                                "relative_path": relative_path,
                                "filename": file_path.name,
                                "article_title": title,
                                "chunk_index": chunk_index,
                                "total_chunks": len(chunks),
                                "last_updated": last_updated.isoformat(),
                            },
                        )
                    )

            if docs:
                project_docs[project] = docs
                log.info(
                    "collected %s: %d files -> %d chunks (%d files too short)",
                    project, files, len(docs), skipped,
                )
            else:
                log.warning("collected %s: no chunks", project)

        return project_docs

    def _embed_with_retry(self, docs: List[Document], max_retries: int = 3) -> FAISS:
        """Embed a batch, retrying transient API failures with backoff."""
        for attempt in range(max_retries):
            try:
                return FAISS.from_documents(docs, self.embeddings)
            except Exception as exc:  # noqa: BLE001 - retry any API-side failure
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** (attempt + 1)
                log.warning(
                    "embedding failed (attempt %d/%d), retrying in %ds: %s",
                    attempt + 1, max_retries, wait, str(exc)[:200],
                )
                time.sleep(wait)
        raise AssertionError("unreachable")

    async def build(self, project_docs: Dict[str, List[Document]]) -> Dict[str, int]:
        """Write a FAISS index and a BM25 sidecar per project.

        Failures are raised, not swallowed: an index that silently does not
        exist turns into a confusing empty-retrieval bug three nodes later.
        """
        self.indices_root.mkdir(parents=True, exist_ok=True)
        built: Dict[str, int] = {}

        for project, docs in project_docs.items():
            if not docs:
                continue
            target = self.indices_root / project

            index = self._embed_with_retry(docs[:EMBED_BATCH_SIZE])
            for start in range(EMBED_BATCH_SIZE, len(docs), EMBED_BATCH_SIZE):
                batch = docs[start : start + EMBED_BATCH_SIZE]
                index.merge_from(self._embed_with_retry(batch))
                log.info(
                    "%s: embedded %d/%d chunks",
                    project, min(start + EMBED_BATCH_SIZE, len(docs)), len(docs),
                )
                # Yield to the event loop between batches so this stays
                # cancellable during a long build.
                await asyncio.sleep(0)

            index.save_local(str(target))

            bm25 = BM25Store()
            bm25.build(docs)
            bm25.save(target)

            built[project] = len(docs)
            log.info("%s: index written to %s (%d chunks)", project, target, len(docs))

        return built
