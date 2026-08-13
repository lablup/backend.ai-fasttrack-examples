"""Fusion is pure and stdlib-only, so it is worth pinning down exactly."""

from __future__ import annotations

import pytest

from docs_rag.fusion import chunk_identity, flatten_with_project, fuse_results


def chunk(project: str, path: str, index: int = 0, **scores) -> dict:
    return {
        "content": f"{path}#{index}",
        "metadata": {"relative_path": path, "chunk_index": index},
        **scores,
    }


def test_flatten_stamps_project_onto_metadata():
    flat = flatten_with_project({"bssh": [chunk("bssh", "a.md")]})
    assert flat[0]["metadata"]["project"] == "bssh"


def test_identity_distinguishes_chunks_of_the_same_file():
    a = chunk("bssh", "a.md", 0)
    b = chunk("bssh", "a.md", 1)
    a["metadata"]["project"] = b["metadata"]["project"] = "bssh"
    assert chunk_identity(a) != chunk_identity(b)


def test_l2_filter_applies_to_semantic_arm_only():
    """A far semantic hit is dropped; a lexical hit with no L2 survives.

    This is the whole point of the hybrid arm: an exact token match has no
    distance to threshold, and discarding it would lose the queries BM25 exists
    to answer.
    """
    semantic = {"p": [chunk("p", "far.md", similarity_score=9.0)]}
    lexical = {"p": [chunk("p", "exact.md", bm25_score=1.0)]}

    fused = fuse_results(semantic, lexical, max_l2=1.5)

    paths = [c["metadata"]["relative_path"] for c in fused]
    assert paths == ["exact.md"]


def test_appearing_in_both_arms_outranks_being_first_in_one():
    both = chunk("p", "both.md", similarity_score=0.5, bm25_score=0.9)
    semantic = {"p": [chunk("p", "sem_top.md", similarity_score=0.1), dict(both)]}
    lexical = {"p": [chunk("p", "lex_top.md", bm25_score=1.0), dict(both)]}

    fused = fuse_results(semantic, lexical, rrf_k=60)

    assert fused[0]["metadata"]["relative_path"] == "both.md"


def test_scores_are_normalised_and_descending():
    semantic = {"p": [chunk("p", f"{i}.md", similarity_score=i * 0.1) for i in range(5)]}
    fused = fuse_results(semantic, {"p": []})

    scores = [c["rrf_score"] for c in fused]
    assert scores[0] == pytest.approx(1.0)
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_empty_input_is_not_an_error():
    assert fuse_results({}, {}) == []
