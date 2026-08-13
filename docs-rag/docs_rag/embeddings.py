"""Embedding-model construction — one place, because a mismatch is silent.

An index built with one embedding model and queried with another returns
plausible-looking nonsense rather than an error, so the indexer, the retriever
and the SemScore metric all come through here.
"""

from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from docs_rag.settings import Settings


def make_embeddings(settings: Settings) -> OpenAIEmbeddings:
    kwargs = {
        "model": settings.embedding_model,
        "api_key": settings.openai_api_key,
        # Send strings straight to the API instead of pre-tokenising into token
        # arrays locally: the array form trips the per-request token limit on
        # long documentation pages.
        "check_embedding_ctx_length": False,
        # Documents per API call. The default of 1000 overshoots the request
        # size limit once chunks are ~1000 tokens each.
        "chunk_size": 200,
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return OpenAIEmbeddings(**kwargs)
