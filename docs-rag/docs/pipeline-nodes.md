# The DAG, node by node

What each node reads, writes and fails on — and what a complete run costs in
wall-clock time.

Every batch node runs the same way: `bash pipeline/bootstrap.sh
pipeline/tasks/<task>.py --project all`. `bootstrap.sh` builds a shared
virtualenv once at `$PIPELINE_VFROOT/.venv`, installs
[`requirements.txt`](../requirements.txt) plus a static pandoc binary, then
activates it and execs the task. Every node accepts `--project <name>` and
`--data-root <path>`.

| Node | Reads | Writes | Fails when |
|---|---|---|---|
| `fetch-code` | — | `/pipeline/vfroot/src/docs-rag` | GitHub unreachable |
| `01_clone_sources` | `sources.yaml` | `01_repos/<source>/` | a clone fails, or a private repo has no token |
| `02_convert_docs` | `01_repos/` | `02_docs_md/<source>/` | globs match nothing; two sources collide on one destination |
| `03_build_indices` | `02_docs_md/` | `03_indices/<source>/` | the embedding API fails after 3 retries |
| `04_verify_indices` | `03_indices/` | `99_state/verify_report.json` | any source's dual canary fails |
| `05_evaluate` | `03_indices/`, `eval_samples.json` | `99_state/eval_report.json` | the chat or judge model is unreachable |
| `06_publish` | `03_indices/`, `99_state/` | `$PIPELINE_VFROOT` | on FastTrack with no vfroot set |
| `07_stage_service` | `03_indices/`, `99_state/` | vfroot + model storage | model storage is not mounted writable |

**`fetch-code`** is the only pure-shell node — it is the node that fetches the
Python, so it cannot itself be Python. It clones this repository into
`/pipeline/vfroot/src`, which FastTrack creates automatically, and every later
node runs from that one checkout. Set `DOCSRAG_REF` to a tag or branch to pin
the revision.

**`01_clone_sources`** is self-healing. It resets to `FETCH_HEAD` via `git
checkout -B` (a branch-scoped fetch does not move the remote-tracking ref),
re-clones a corrupted working copy, and injects `GITHUB_TOKEN` / `GH_TOKEN` into
the URL for private repositories. It runs git with `GIT_TERMINAL_PROMPT=0` so a
missing token fails in seconds instead of blocking forever on a credential
prompt nothing will answer. It declares `seed_stages=()` — see
[head-node detection](paths-and-data.md).

**`02_convert_docs`** converts `.rst` with pandoc and **copies `.md` through**.
Routing is by extension, so one repository can mix both. Markdown-native sources
would otherwise produce nothing, since the converter only handles `.rst`. Before
converting it checks that no two sources map to the same destination, which
would otherwise leave one source silently overwriting the other.

**`03_build_indices`** chunks, embeds, and writes both indices. Details in
[How retrieval works](retrieval.md#how-retrieval-works).

**`04_verify_indices`** runs the dual canary per source. See
[The reports](evaluation.md#the-reports).

**`05_evaluate`** grades the committed question set. See
[Evaluation](evaluation.md).

**`06_publish`** copies `03_indices/` and `99_state/` to `PIPELINE_VFROOT`. A
no-op locally when unset; a **hard failure** under FastTrack, because publishing
nothing while reporting success is how a run looks green and delivers nothing.

**`07_stage_service`** assembles everything the two deployments need. It copies
the code tree (excluding `data`, `.git`, `__pycache__`, `.venv*`, `.env`,
`vfroot`, `.pytest_cache`), copies the indices to
`<code>/pipeline/data/03_indices` so the serving entrypoint resolves them
exactly as a local checkout would, writes a `.env`, issues the credentials, and
mirrors the four small files into model storage. See
[How it stays portable](architecture.md).

## What a run looks like

Measured on a 26.4.x cluster against the five default sources: 826 chunks
indexed in 27.7 s, all five canaries passing with top-hit L2 between 0.73 and
0.93 — each retrieving its own documentation, not merely scoring under the
threshold — and 20/20 graded questions scoring 0.94 overall, groundedness 0.96,
in 94 s.

| Node | Time | Result |
|---|---|---|
| `clone-docs` | 55 s | 5 repos |
| `build-indices` | 28 s | 826 chunks, FAISS + BM25 |
| `verify-indices` | 2 s | 5/5 canaries |
| `evaluate` | 94 s | 0.9414 overall |
| `publish` / `stage-service` | <1 s | vfroot + model storage |

Both services then answer on `:8080`, enforce their bearer token and login, and
scope retrieval per request — `{"projects": ["mlxcel"]}` narrows to one corpus,
an unknown name is a `400` rather than a `500`.

---

[← All documentation](../README.md)
