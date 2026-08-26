# Retrieval and answers

How a question turns into excerpts from your documentation, and how those
excerpts turn into an answer.

## How retrieval works

### Chunking

[`docs_rag/indexer.py`](../docs_rag/indexer.py) splits with LangChain's
`RecursiveCharacterTextSplitter`, but with a **token** length function
(`tiktoken`, `o200k_base`) rather than characters — the size that matters is the
one the embedding model and the context window see. Separators are tried in
order: `\n## `, `\n### `, `\n\n`, `\n`, ` `, `""`, so a split lands on a heading
boundary when it can.

Files with fewer than 120 characters of content are skipped: a stub page
contributes a chunk that matches everything weakly and nothing well.

Each chunk is stored with a provenance header prepended to its text —

```
[bssh | Running commands on many hosts | docs/usage.md]

<the chunk>
```

— because a retrieved chunk arrives at the model with no surrounding document,
and the header is what lets it attribute the answer. Metadata carried alongside:
`project`, `relative_path`, `filename`, `article_title`, `chunk_index`,
`total_chunks`, `last_updated`.

Embedding runs in batches of 50 documents, with exponential backoff over three
attempts on API failure, and yields to the event loop between batches so a long
build stays cancellable. Failures are raised, not swallowed — an index that
silently does not exist becomes a confusing empty-retrieval bug three nodes
later.

### The three modes

Selectable per request with `retrieval_mode`:

- **`hybrid`** (default) — runs semantic and lexical search in parallel and
  fuses them with Reciprocal Rank Fusion. RRF combines by *rank*, which
  sidesteps the fact that FAISS L2 distance and BM25 relevance have no common
  scale. Normalising them onto one scale requires knowing the corpus; ranking
  does not.
- **`semantic`** — FAISS only, with the L2 confidence cutoff.
- **`lexical`** — BM25 only. Best for exact tokens: error codes, CLI flags,
  config keys.

Hybrid **degrades to semantic** when no BM25 sidecar is loaded, so an index
built before the lexical arm existed still works.

Retrieval is two-stage: `k_per_project` chunks are pulled from each named corpus
in parallel, then the pooled result is cut to `top_k`. The query is embedded
exactly once and the same vector is searched against every index.

### Fusion

[`docs_rag/fusion.py`](../docs_rag/fusion.py). A chunk's fused score is

```
score = Σ  1 / (rrf_k + rank + 1)      over the lists it appears in
```

so appearing in **both** arms beats ranking highly in one. Larger `rrf_k`
dampens the advantage of the very top ranks. Scores are normalised to `[0, 1]`
and exposed as `rrf_score`. Chunks are deduplicated on
`(project, relative_path, chunk_index)`, and the first-seen dict wins, so a
chunk found by both arms keeps its semantic-side fields.

**The L2 cutoff applies to the semantic arm only.** A lexical hit has no
distance to threshold, and dropping it would discard exactly the queries BM25
exists to answer.

**One BM25 caveat worth knowing.** `rank_bm25` floors IDF to zero for a term
that appears in most documents, so a query of only common words scores every
document identically. [`docs_rag/bm25.py`](../docs_rag/bm25.py) therefore ranks
only documents that actually contain a query token, and returns nothing when
none do — rather than returning the whole corpus in arbitrary order.

### The context the model sees

[`format_context()`](../docs_rag/retriever.py) renders each chunk as:

```
--- Document 3 [Project: bssh | Article: Running commands | Source: docs/usage.md
    | Chunk: 2/7 | Similarity(L2): 0.812] ---
```

A chunk that came from the lexical arm alone shows `Lexical: <score>` instead of
a similarity it does not have. Bodies longer than `MAX_CHARS_PER_CHUNK` are
truncated with a visible marker.

## How answers are generated

[`docs_rag/rag.py`](../docs_rag/rag.py) is deliberately small: a retriever, a chat
model, and a bounded history. The system prompt does the work, and it enforces
context-only answering:

1. Use only what is in the context; do not guess.
2. If the context does not answer the question, say so plainly — then say what
   related information the context *does* contain.
3. Never invent specific values. IPs, ports, paths, hostnames, versions,
   environment variables, CLI flags and commands must appear verbatim in the
   context or not at all.
4. Excerpts are labelled with their match score; L2 is lower-is-better, and
   `Lexical` means keyword-matched. Prefer closer matches, and say when every
   excerpt is weak.
5. Cite the source file per substantive claim, e.g. `(backendai: docs/install/index.rst)`.
6. The corpus spans separate projects — do not blend them into one procedure.

History is bounded at six turns (twelve messages) because the whole history is
re-sent on every request, so an unbounded one silently inflates cost and
latency. `seed_history()` loads prior turns from an OpenAI-style `messages`
array and **drops any caller-supplied system message** on purpose: the grounding
rules above are the server's, and honouring an injected one would let a request
opt out of them.

Answers stream token by token. The assembled answer, not the last fragment, is
what goes into history.

---

[← All documentation](../README.md)
