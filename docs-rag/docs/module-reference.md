# Module reference and tests

What every file is for, and how to run the suites.

## Module reference

```
fasttrack_pipeline.yaml         key from the Secrets store or a .env
fasttrack_pipeline_inline.yaml  every value typed in; nothing to upload
model-definition-*.yaml         staged into model storage automatically

docs_rag/            the RAG library
  settings.py          layered .env loading; every tunable; blank-means-unset
  convert.py           rst → markdown via pandoc, md passthrough, collision check
  embeddings.py        one place that constructs OpenAIEmbeddings
  indexer.py           chunk → FAISS + BM25, with provenance headers
  bm25.py              lexical index; ranks only documents holding a query token
  retriever.py         semantic / lexical / hybrid search, context formatting
  fusion.py            Reciprocal Rank Fusion, L2 filter on the semantic arm
  rag.py               retrieve → prompt → stream, bounded history
  evaluate.py          LLM judge, SemScore, weighted overall
  credentials.py       service auth: issue, save, resolve, fail closed
  server.py            OpenAI-compatible API
  ui.py                Gradio chat
pipeline/            the FastTrack DAG
  paths.py             path resolution (single source of truth)
  runner.py            task entrypoint, input seeding, run manifest
  preflight.py         precondition checks and startup evidence
  io.py                sources.yaml and report schemas
  bootstrap.sh         venv + pandoc, then run a task
  serve.sh             deployment entrypoint for both services (+ `setup`)
  config/sources.yaml  what to ingest
  config/eval_samples.json  the graded question set
  tasks/01..07         the DAG nodes
```

`docs_rag/` never imports `pipeline/`. The library is usable on its own — that
is what lets the two services run from a staged code tree with no pipeline
context at all.

## Tests

```bash
PYTHONPATH=$(pwd) pytest tests/unit -q     # pure, fast, no network
PYTHONPATH=$(pwd) pytest tests/e2e -q      # real task scripts, still offline
```

| Suite | Covers |
|---|---|
| `test_settings.py` | layered `.env`, blank-means-unset, placeholder handling, convert collisions |
| `test_fusion.py` | RRF ordering and normalisation, the L2 filter, BM25 no-overlap |
| `test_server_auth.py` | 401 / 503 / `ALLOW_UNAUTHENTICATED`, health readiness |
| `test_pipeline_wiring.py` | both YAMLs agree; only `stage-service` mounts a vFolder; placeholders intact |
| `test_offline_chain.py` | the whole DAG as subprocesses against a `file://` repo |

The e2e suite drives the actual task scripts as subprocesses against a local
`file://` git repository, with each node given its own output root and an input
root pointing at the previous node's — the way FastTrack lays it out, and
deliberately not a shared directory, which would hide every chaining bug. It
needs no API key.

```bash
ruff check .
```

---

[← All documentation](../README.md)
