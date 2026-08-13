# docs-rag

A complete retrieval-augmented generation pipeline on Backend.AI FastTrack: it
clones open-source documentation repositories, converts and indexes them,
measures how good the answers are, and deploys two services — a Gradio chat UI
and an OpenAI-compatible HTTP API.

```
fetch-code → clone-docs → convert-docs → build-indices → verify-indices
   → evaluate → publish → stage-service → ┬─ serve-fastapi
                                          └─ serve-gradio
```

It ships with five public Backend.AI sources — [backend.ai], [backend.ai-docs-webui],
[mlxcel], [all-smi] and [bssh] — but the source list is a config file, so
pointing it at your own documentation is an edit, not a fork.

[backend.ai]: https://github.com/lablup/backend.ai
[backend.ai-docs-webui]: https://github.com/lablup/backend.ai-docs-webui
[mlxcel]: https://github.com/lablup/mlxcel
[all-smi]: https://github.com/lablup/all-smi
[bssh]: https://github.com/lablup/bssh

## Running it on FastTrack

**Pick a pipeline file.** Both describe the identical DAG and differ only in
where your API key comes from:

| File | Your key lives in | Upload anything? |
|---|---|---|
| [`fasttrack_pipeline_inline.yaml`](fasttrack_pipeline_inline.yaml) | this file, typed in | **no** |
| [`fasttrack_pipeline.yaml`](fasttrack_pipeline.yaml) | the Secrets store, or a `.env` | only if you choose `.env` |

Use the inline file unless you specifically want the key out of the pipeline
definition. Every setting in it is already filled in with its default, so
`OPENAI_API_KEY` is the only line you have to touch.

Then:

1. **Fill in `OPENAI_API_KEY`** at the top of your chosen file.
2. **Create a model storage folder** and select it in the Create Pipeline
   dialog. **Leave it empty** — you never upload anything into it. The
   `stage-service` node writes the indices, the services' code and this run's
   credentials there itself, taking them from the previous task's output. The
   folder exists only because a Backend.AI model service always mounts one.
3. **Create the pipeline**, then **Dry Run** and **Run**.

That is the whole setup. This pipeline names **no vFolder anywhere** — vFolder
names are account-scoped, so a pipeline that hardcodes one cannot be shared with
anybody. See [How it stays portable](#how-it-stays-portable) for how the code
and the indices get where they need to be without one.

> Both files declare the same service names (`docs-rag-api`, `docs-rag-ui`), and
> service names must be unique within a resource group. If you want to run both
> variants at once, rename the services in one of them.

The `model-definition-*.yaml` files are staged into model storage automatically
by the `stage-service` node. Upload your own copies first if you want to
customise them — the staging node will not overwrite a file that is already
there.

### Where to find things afterwards

| What | Where |
|---|---|
| Service login and API token | last lines of the `stage-service` task log |
| Indices and reports | the pipeline vFolder, `<pipeline-name>-<id>/.pipeline/` |
| Retrieval health per source | `99_state/verify_report.json` |
| Answer quality scores | `99_state/eval_report.json` |
| Per-task success and timings | `99_state/run_manifest.json` |

## Running it locally

```bash
pip install -r requirements-service.txt
cp .env.example .env          # then fill in OPENAI_API_KEY
export PYTHONPATH=$(pwd)

python pipeline/tasks/01_clone_sources.py  --project all
python pipeline/tasks/02_convert_docs.py   --project all
python pipeline/tasks/03_build_indices.py  --project all
python pipeline/tasks/04_verify_indices.py --project all
python pipeline/tasks/05_evaluate.py       --project all
```

Locally none of the `PIPELINE_*` variables are set, so every node reads and
writes one shared `pipeline/data/` tree and no input seeding happens. Each node
is idempotent — re-run any one of them on its own.

`--project <name>` restricts a run to one source, which is much faster while you
are iterating on globs or chunk sizes.

Then serve it:

```bash
bash pipeline/serve.sh fastapi   # OpenAI-compatible API on :8000
bash pipeline/serve.sh gradio    # chat UI on :8000
```

```bash
curl -H "Authorization: Bearer $API_KEY" \
     -H 'Content-Type: application/json' \
     -d '{"messages":[{"role":"user","content":"How do I stop bssh on the first host failure?"}],
          "projects":["bssh"]}' \
     localhost:8000/v1/chat/completions
```

Requirements for a local run: `git`, `pandoc`, and an `OPENAI_API_KEY`. On
FastTrack `pipeline/bootstrap.sh` provides pandoc itself.

## Configuration

There is one configuration mechanism — environment variables — and three ways to
set them. Every setting appears in both pipeline files' `environment.envs`
block, so you never have to leave the pipeline definition.

**Typed into the pipeline YAML.** The default, and the only route that needs no
upload and no vFolder:

```yaml
environment:
  envs:
    OPENAI_API_KEY: "sk-..."
    LLM_MODEL: "gpt-4.1"
    CHUNK_SIZE: "1500"
    GLOBAL_TOP_K: "20"
```

**From the Secrets store**, with `${{ secrets.NAME }}` in any `envs` value — the
route `fasttrack_pipeline.yaml` uses for the API key, so the pipeline definition
never contains a credential.

**From a `.env` file**, optional, for when you would rather not put settings in
either. Copy [`.env.example`](.env.example) and upload it to model storage.

Resolution order, first hit wins:

| Order | Source | Set by |
|---|---|---|
| 1 | `os.environ` | a value typed into the pipeline YAML, a GUI secret, or a shell export |
| 2 | `/models/.env` | you, uploading to model storage (optional) |
| 3 | `/pipeline/vfroot/.env` | you, uploading to the auto-created pipeline vFolder |
| 4 | built-in default | [`docs_rag/settings.py`](docs_rag/settings.py) |

Layer 2 is the one that also reaches the deployed services: a serving container
gets no `/pipeline` mounts at all.

> **Blank means unset.** An empty value, or an unresolved `${{ secrets.NAME }}`
> placeholder, falls through to the next layer instead of winning as an empty
> string. That is what makes it safe to ship the YAML with every setting listed
> and blank, and what stops a reference to a secret you never created from
> silently blanking your API key.

Every tunable — chunk size, retrieval depth, RRF constant, model names, the L2
cutoff — has a default, so the only value you must supply is `OPENAI_API_KEY`.
There is deliberately no config YAML on top of this: a second place to set a
chunk size is how an index gets built with one value and queried with another.

## How retrieval works

Three modes, selectable per request with `retrieval_mode`:

- **`hybrid`** (default) — runs semantic and lexical search in parallel and
  fuses them with Reciprocal Rank Fusion. RRF combines by *rank*, which
  sidesteps the fact that FAISS L2 distance and BM25 relevance have no common
  scale.
- **`semantic`** — FAISS only, with an L2 confidence cutoff.
- **`lexical`** — BM25 only. Best for exact tokens: error codes, CLI flags,
  config keys.

The BM25 index is built from exactly the chunks that go into FAISS, so it costs
no extra embedding calls. The L2 cutoff applies to the semantic arm only — a
lexical hit has no distance to threshold, and dropping it would discard the
queries BM25 exists to answer.

`verify-indices` runs a **dual canary** per source: the top semantic hit must be
closer than the threshold *and* the lexical arm must return a hit. Either check
alone passes on a broken index, because a FAISS index built from empty files
still returns its nearest neighbour.

## Adding or changing a source

Edit [`pipeline/config/sources.yaml`](pipeline/config/sources.yaml). No code
changes.

```yaml
sources:
  - name: myproject
    url: https://github.com/me/myproject.git
    branch: main
    include: ["docs/**/*.md", "README.md"]
    exclude: ["docs/generated/**/*"]
    verify_query: "How do I install myproject?"
```

`include`/`exclude` are pathlib globs relative to the repo root. They are globs
rather than a single "docs directory" because real repositories do not agree on
where documentation lives — of the five defaults, one keeps it under `docs/`
next to translation catalogs that must be skipped, one keeps its real
documentation at the repo root, and one buries 42 benchmark dumps in `docs/`.

`.rst` is converted with pandoc, `.md` is copied through; routing is by
extension, so a repo can mix both. A source whose globs match nothing fails the
node loudly rather than producing an empty index that only reveals itself at
query time.

Private repositories work too: set `GITHUB_TOKEN` in your `.env` and the clone
node injects it. It runs git with `GIT_TERMINAL_PROMPT=0`, so a missing token
fails immediately instead of hanging forever on a credential prompt nothing will
answer.

## How it stays portable

Three mechanisms replace the usual "mount a code vFolder" approach.

**The code arrives by git clone.** The `fetch-code` node — the only one that is
pure shell, because it is the node that fetches the Python — clones this repo
into `/pipeline/vfroot/src`, which FastTrack creates automatically. Every later
node runs from that one checkout, so the whole DAG is guaranteed to run the same
revision. Pin it by setting `DOCSRAG_REF` to a tag.

**The indices reach the services through model storage.** A deployment container
is not part of the task chain and gets no `/pipeline` mounts, which is why
`stage-service` copies the code, the indices and this run's credentials into
`/models`. Model storage is selected in the Create Pipeline dialog rather than
named in the YAML — that is what keeps the file account-agnostic. The services
then start with no network access and no git.

**Config arrives layered**, as described above, so it works whichever of those
mounts a given cluster actually provides.

Output stays cumulative down the chain because `runner.run_task()` seeds each
node's output from the previous node's, but each node declares *which* stages it
needs. Without that narrowing every node would drag the cloned repos and the
converted markdown through the output mount — hundreds of megabytes per hop for
data nothing downstream reads.

> **Head-node detection is structural.** A task that declares no dependency is
> given no `/pipeline/input1`, so seeding no-ops on the absent path. Do not gate
> it on `BACKENDAI_PIPELINE_JOB_INDEX == "1"`: that value is not a reliable DAG
> position, and a cluster that reports `1` for a mid-chain task makes every node
> skip seeding and die on an empty upstream stage.

## Service authentication

Both services **refuse to start without credentials**. An empty password meaning
"no login required" is how a documentation service ends up publicly readable
without anyone noticing.

- Set `GRADIO_USERNAME`, `GRADIO_PASSWORD` and `API_KEY` in your `.env` to pin
  them. All three or none — a half-filled login is ignored on purpose.
- Leave them blank and `stage-service` generates fresh ones per run, prints them
  at the end of its log, and writes them to `/models/99_state/service_credentials.json`
  with mode 0600.
- `ALLOW_UNAUTHENTICATED=1` serves deliberately open, with a warning.

Those credentials are printed on purpose — they are yours, and the task log is
visible only to you. `OPENAI_API_KEY` and `GITHUB_TOKEN` are never printed
anywhere, only reported as set or unset with a character count.

## Adapting to your cluster

`fasttrack_pipeline.yaml` carries the shape a 26.4.x cluster accepted. If yours
follows the published schema, change these four:

| Field | This file | Published schema |
|---|---|---|
| `version` | `26.4.4rc6` | your cluster's release |
| ownership | `domain:` + `scope: user` | `domain_name:` + `scope: project` |
| serving nodes | `type: deployment` | `type: serving` |
| `scaling-group` | `default` | your resource group |

Export an existing pipeline from your cluster to see which shape it uses.

The batch nodes run a CPU image (`cr.backend.ai/multiarch/python:3.10-ubuntu20.04`,
4 CPU / 8 GB) and request no GPU. Embedding is an HTTPS call to a third-party
API, so nothing in this pipeline touches an accelerator — and a CPU image
schedules on any cluster.

## Layout

```
fasttrack_pipeline.yaml         key from the Secrets store or a .env
fasttrack_pipeline_inline.yaml  every value typed in; nothing to upload
model-definition-*.yaml         staged into model storage automatically

docs_rag/            the RAG library
  settings.py          layered .env loading; every tunable
  convert.py           rst → markdown via pandoc, md passthrough
  indexer.py           chunk → FAISS + BM25
  retriever.py         semantic / lexical / hybrid search, context formatting
  fusion.py            Reciprocal Rank Fusion
  rag.py               retrieve → prompt → stream
  evaluate.py          LLM judge + SemScore
  credentials.py       service auth, fail-closed
  server.py            OpenAI-compatible API
  ui.py                Gradio chat
pipeline/            the FastTrack DAG
  paths.py             path resolution (single source of truth)
  runner.py            task entrypoint, input seeding, run manifest
  preflight.py         precondition checks and startup evidence
  io.py                sources.yaml and report schemas
  bootstrap.sh         venv + pandoc, then run a task
  serve.sh             deployment entrypoint for both services
  config/sources.yaml  what to ingest
  config/eval_samples.json  the graded question set
  tasks/01..07         the DAG nodes
```

## Tests

```bash
PYTHONPATH=$(pwd) pytest tests/unit -q     # pure, fast, no network
PYTHONPATH=$(pwd) pytest tests/e2e -q      # real task scripts, still offline
```

The e2e suite drives the actual task scripts as subprocesses against a local
`file://` git repository, with each node given its own output root and an input
root pointing at the previous node's — the way FastTrack lays it out, and
deliberately not a shared directory, which would hide every chaining bug. It
needs no API key.

## What a run looks like

Against the five default sources: 201 documentation files, 808 chunks, indices
built in about 25 seconds, all five canaries passing with top-hit L2 between
0.73 and 0.93, and an aggregate answer score around 0.94 (groundedness 0.96)
across the 20 graded questions.
