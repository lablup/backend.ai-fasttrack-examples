"""BM25 lexical index, built over exactly the chunks that go into FAISS.

One `bm25.pkl` per project, sitting beside that project's FAISS files. It is
built from the same chunk list, so it costs no extra embedding calls, and it
catches what dense retrieval reliably misses: error codes, CLI flags, config
keys, and any rare literal token whose embedding is dominated by its context.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

log = logging.getLogger("bm25")

INDEX_FILENAME = "bm25.pkl"


def tokenize(text: str) -> List[str]:
    """Whitespace tokenizer, used identically at index and query time."""
    return text.lower().split()


class BM25Store:
    """In-memory BM25 index over one project's chunks."""

    def __init__(self) -> None:
        self._index: Optional[BM25Okapi] = None
        self._documents: List[Document] = []
        self._tokenized_corpus: List[List[str]] = []

    @property
    def is_empty(self) -> bool:
        return self._index is None or not self._documents

    def build(self, documents: List[Document]) -> None:
        if not documents:
            log.warning("bm25: nothing to index")
            return
        self._documents = documents
        self._tokenized_corpus = [tokenize(doc.page_content) for doc in documents]
        self._index = BM25Okapi(self._tokenized_corpus)
        log.info("bm25: built index over %d chunks", len(documents))

    def save(self, directory: Path) -> None:
        if self._index is None:
            log.warning("bm25: no index to save")
            return
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / INDEX_FILENAME, "wb") as f:
            pickle.dump(
                {
                    "index": self._index,
                    "tokenized_corpus": self._tokenized_corpus,
                    "documents": self._documents,
                },
                f,
            )
        log.info("bm25: saved to %s", directory / INDEX_FILENAME)

    def load(self, directory: Path) -> None:
        index_path = Path(directory) / INDEX_FILENAME
        if not index_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {index_path}")
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        self._index = data["index"]
        self._tokenized_corpus = data["tokenized_corpus"]
        self._documents = data["documents"]
        log.info("bm25: loaded %d chunks from %s", len(self._documents), index_path)

    def search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Top-k (Document, score), scores min-max normalised to [0, 1].

        Normalisation is per-query, not per-corpus: BM25 scores are only
        meaningful relative to the other hits for the same query.
        """
        if self.is_empty:
            return []

        scores = self._index.get_scores(tokenize(query))

        k = min(k, len(self._documents))
        top_indices = sorted(
            range(len(self._documents)), key=lambda i: scores[i], reverse=True
        )[:k]

        min_score = float(min(scores))
        max_score = float(max(scores))
        score_range = max_score - min_score if max_score != min_score else 1.0

        results: List[Tuple[Document, float]] = []
        for idx in top_indices:
            normalized = (float(scores[idx]) - min_score) / score_range
            results.append((self._documents[idx], max(0.0, min(1.0, normalized))))
        return results
