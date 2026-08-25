"""Load the built indices and answer queries against them.

Three modes:

    hybrid    (default) semantic + BM25, fused by Reciprocal Rank Fusion
    semantic  FAISS only, with the L2 confidence cutoff
    lexical   BM25 only — exact tokens, error codes, CLI flags

Hybrid degrades to semantic when no BM25 sidecar is loaded, so an index built
before the lexical arm existed still works.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_community.vectorstores import FAISS

from docs_rag.bm25 import BM25Store
from docs_rag.embeddings import make_embeddings
from docs_rag.fusion import flatten_with_project, fuse_results
from docs_rag.settings import Settings

log = logging.getLogger("retriever")

RETRIEVAL_MODES = ("hybrid", "semantic", "lexical")


class Retriever:
    """Per-project FAISS + BM25 indices, searched in parallel."""

    def __init__(self, indices_root: Path, settings: Settings):
        self.indices_root = Path(indices_root)
        self.settings = settings
        self.embeddings = make_embeddings(settings)
        self.indices: Dict[str, FAISS] = {}
        self.bm25: Dict[str, BM25Store] = {}

    @property
    def projects(self) -> List[str]:
        return sorted(self.indices)

    def load(self) -> List[str]:
        """Load every index directory found under `indices_root`."""
        if not self.indices_root.is_dir():
            log.warning("no indices directory at %s", self.indices_root)
            return []

        for project_dir in sorted(self.indices_root.iterdir()):
            if not project_dir.is_dir():
                continue
            project = project_dir.name
            try:
                self.indices[project] = FAISS.load_local(
                    str(project_dir),
                    self.embeddings,
                    # These files are produced by this pipeline, in this
                    # account's storage. The flag exists because FAISS indices
                    # unpickle; loading a third party's index would not be safe.
                    allow_dangerous_deserialization=True,
                )
            except Exception as exc:  # noqa: BLE001 - one bad index must not sink the rest
                log.error("failed to load index for %s: %s", project, exc)
                continue

            store = BM25Store()
            try:
                store.load(project_dir)
                self.bm25[project] = store
            except FileNotFoundError:
                log.info("%s: no BM25 sidecar — semantic only", project)
            except Exception as exc:  # noqa: BLE001
                log.error("failed to load BM25 index for %s: %s", project, exc)

        log.info("loaded indices: %s", ", ".join(self.projects) or "<none>")
        return self.projects

    @staticmethod
    def _format(docs_with_scores) -> List[dict]:
        return [
            {
                "content": doc.page_content,
                "metadata": dict(doc.metadata),
                "similarity_score": float(score),
            }
            for doc, score in docs_with_scores
        ]

    async def search_semantic(
        self, projects: List[str], query: str, k: int
    ) -> Dict[str, List[dict]]:
        """FAISS search across projects, embedding the query exactly once."""
        query_vector = self.embeddings.embed_query(query)

        async def one(project: str) -> Tuple[str, List[dict]]:
            index = self.indices.get(project)
            if index is None:
                return project, []
            try:
                hits = index.similarity_search_with_score_by_vector(query_vector, k=k)
                return project, self._format(hits)
            except Exception as exc:  # noqa: BLE001
                log.error("semantic search failed for %s: %s", project, exc)
                return project, []

        return dict(await asyncio.gather(*(one(p) for p in projects)))

    async def search_lexical(
        self, projects: List[str], query: str, k: int
    ) -> Dict[str, List[dict]]:
        """BM25 search across projects. Results carry `bm25_score`, not L2."""

        async def one(project: str) -> Tuple[str, List[dict]]:
            store = self.bm25.get(project)
            if store is None or store.is_empty:
                return project, []
            try:
                return project, [
                    {
                        "content": doc.page_content,
                        "metadata": dict(doc.metadata),
                        "bm25_score": score,
                    }
                    for doc, score in store.search(query, k=k)
                ]
            except Exception as exc:  # noqa: BLE001
                log.error("lexical search failed for %s: %s", project, exc)
                return project, []

        return dict(await asyncio.gather(*(one(p) for p in projects)))

    async def retrieve(
        self,
        query: str,
        projects: List[str],
        mode: str = "hybrid",
        top_k: int | None = None,
    ) -> List[dict]:
        """Retrieve and rank chunks across projects.

        Returns a flat list, best-first, capped at `top_k`.
        """
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"unknown retrieval mode {mode!r}; expected one of {RETRIEVAL_MODES}")

        projects = [p for p in projects if p in self.indices]
        if not projects:
            return []

        settings = self.settings
        top_k = top_k or settings.global_top_k
        k = settings.k_per_project

        if mode == "hybrid" and not self.bm25:
            log.info("hybrid requested but no BM25 index is loaded — using semantic")
            mode = "semantic"

        if mode == "semantic":
            per_project = await self.search_semantic(projects, query, k)
            chunks = flatten_with_project(per_project)
            chunks = [c for c in chunks if c["similarity_score"] <= settings.max_l2]
            chunks.sort(key=lambda c: c["similarity_score"])
            return chunks[:top_k]

        if mode == "lexical":
            per_project = await self.search_lexical(projects, query, k)
            chunks = flatten_with_project(per_project)
            chunks.sort(key=lambda c: c["bm25_score"], reverse=True)
            return chunks[:top_k]

        semantic, lexical = await asyncio.gather(
            self.search_semantic(projects, query, k),
            self.search_lexical(projects, query, k),
        )
        fused = fuse_results(
            semantic, lexical, rrf_k=settings.rrf_k, max_l2=settings.max_l2
        )
        return fused[:top_k]


def format_context(chunks: List[dict], max_chars_per_chunk: int = 4000) -> str:
    """Render retrieved chunks into the context block the LLM sees.

    Each chunk is headed by where it came from and how it was found, so the
    model can attribute its answer and weigh a strong match above a weak one.
    """
    blocks: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        parts = [
            f"Project: {meta.get('project', 'unknown')}",
            f"Article: {meta.get('article_title', 'untitled')}",
            f"Source: {meta.get('relative_path', 'unknown')}",
            f"Chunk: {meta.get('chunk_index', 0) + 1}/{meta.get('total_chunks', 1)}",
        ]
        if "similarity_score" in chunk:
            parts.append(f"Similarity(L2): {chunk['similarity_score']:.3f}")
        else:
            # A chunk with no L2 came from the lexical arm alone — say so rather
            # than printing a similarity it does not have.
            parts.append(f"Lexical: {chunk.get('bm25_score', 0.0):.3f}")

        body = chunk.get("content", "")
        if len(body) > max_chars_per_chunk:
            body = body[:max_chars_per_chunk] + "\n[... truncated ...]"

        blocks.append(f"--- Document {i} [{' | '.join(parts)}] ---\n{body}")

    return "\n\n".join(blocks)
