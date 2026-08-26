"""Reciprocal Rank Fusion for combining semantic and lexical result lists.

RRF fuses by *rank*, not score, which sidesteps the scale mismatch between FAISS
L2 distance (lower is better, unbounded) and BM25 relevance (higher is better,
corpus-dependent). Normalising those onto a common scale requires knowing the
corpus; ranking does not.

Every input list must already be sorted best-first.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple


def chunk_identity(item: dict) -> tuple:
    """Stable identity for a chunk, used to dedup across the two lists."""
    m = item.get("metadata", {})
    return (m.get("project"), m.get("relative_path"), m.get("chunk_index"))


def flatten_with_project(results: Dict[str, List[dict]]) -> List[dict]:
    """Flatten {project: [chunks]} into one list, stamping project onto metadata."""
    chunks: List[dict] = []
    for project, docs in results.items():
        for doc in docs:
            doc["metadata"]["project"] = project
            chunks.append(doc)
    return chunks


def reciprocal_rank_fusion(
    ranked_lists: List[List[dict]],
    key_fn: Callable[[dict], tuple],
    k: int = 60,
) -> List[dict]:
    """Fuse best-first ranked lists of chunk dicts.

    A chunk's score is the sum over lists of `1 / (k + rank + 1)`, so appearing
    in both lists beats ranking highly in one. Larger `k` dampens the advantage
    of the top ranks.

    Returns deduplicated chunks sorted by descending score, each annotated with
    `rrf_score` normalised to [0, 1]. The first-seen dict for an identity is
    kept, so a chunk found by both arms retains its semantic-side fields.
    """
    fused: Dict[tuple, Tuple[dict, float]] = {}

    for results in ranked_lists:
        for rank, item in enumerate(results):
            identity = key_fn(item)
            contribution = 1.0 / (k + rank + 1)  # rank is 0-indexed
            if identity in fused:
                existing_item, existing_score = fused[identity]
                fused[identity] = (existing_item, existing_score + contribution)
            else:
                fused[identity] = (item, contribution)

    ordered = sorted(fused.values(), key=lambda pair: pair[1], reverse=True)

    max_score = ordered[0][1] if ordered else 0.0
    result: List[dict] = []
    for item, score in ordered:
        item = dict(item)
        item["rrf_score"] = score / max_score if max_score > 0 else 0.0
        result.append(item)
    return result


def fuse_results(
    semantic_results: Dict[str, List[dict]],
    lexical_results: Dict[str, List[dict]],
    rrf_k: int = 60,
    max_l2: float = 1.5,
) -> List[dict]:
    """Fuse per-project semantic and BM25 results into one ranked list.

    The L2 confidence filter is applied to the semantic arm *only*. A lexical
    hit has no L2 distance to threshold, and dropping it would discard exactly
    the queries BM25 exists to answer — an error code, a CLI flag, a config key
    that embeds poorly but matches literally.
    """
    semantic = flatten_with_project(semantic_results)
    semantic = [c for c in semantic if c.get("similarity_score", float("inf")) <= max_l2]
    semantic.sort(key=lambda c: c.get("similarity_score", float("inf")))

    lexical = flatten_with_project(lexical_results)
    lexical.sort(key=lambda c: c.get("bm25_score", 0.0), reverse=True)

    return reciprocal_rank_fusion([semantic, lexical], key_fn=chunk_identity, k=rrf_k)
