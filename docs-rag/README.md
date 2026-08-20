# docs-rag

**Ask questions about your documentation and get answers that cite the page they
came from.**

docs-rag reads documentation out of GitHub repositories and puts two things in
front of you: a chat window you can talk to, and an API your own software can
call. Ask *"how do I stop bssh on the first host failure?"* and the answer comes
back assembled from the actual documentation, with the file it was taken from
named underneath — not from a language model's memory of the internet.

It runs on Backend.AI FastTrack: you upload one file, press Run, and a few
minutes later both services are live. It ships pointed at five public Backend.AI
projects — [backend.ai], [backend.ai-docs-webui], [mlxcel], [all-smi] and
[bssh] — and pointing it at your own documentation is an edit to one config
file, not a fork.

[backend.ai]: https://github.com/lablup/backend.ai
[backend.ai-docs-webui]: https://github.com/lablup/backend.ai-docs-webui
[mlxcel]: https://github.com/lablup/mlxcel
[all-smi]: https://github.com/lablup/all-smi
[bssh]: https://github.com/lablup/bssh

## What you get

| | |
|---|---|
| **A chat UI** | a web page with a login, one checkbox per documentation set, and example questions |
| **An HTTP API** | OpenAI-compatible, so any existing OpenAI client library works against it unchanged |
| **A quality report** | every run grades its own answers and says whether each documentation set is actually being found |

## How it works, in one picture

```
fetch-code → clone-docs → convert-docs → build-indices → verify-indices
   → evaluate → publish → stage-service → ┬─ serve-fastapi
                                          └─ serve-gradio
```

In words: fetch this code, clone the documentation repositories, convert
everything to markdown, index it, check that the index can actually answer a
question, grade the answers, publish the results, and start the two services.

## Quickstart

### On FastTrack

1. **Download [`fasttrack_pipeline_inline.yaml`](fasttrack_pipeline_inline.yaml)** —
   every setting in it is already filled in with a working default.
2. **Fill in `OPENAI_API_KEY`** at the top. That is the only line you *have* to
   touch.
3. **Create an empty model storage vFolder** and put its name in the three
   places the file marks `your_model_folder`. You never upload anything into it;
   the pipeline writes what it needs there itself.
4. **Set `project` and `scaling-group`** to your cluster's, if they are not
   `default` — then **Create the pipeline**, **Dry Run**, and **Run**.

Full walkthrough, including what to change for a different cluster version:
**[Running it on FastTrack](docs/running-on-fasttrack.md)**.

### On your own machine

```bash
pip install -r requirements-service.txt
cp .env.example .env          # then fill in OPENAI_API_KEY
export PYTHONPATH=$(pwd)

python pipeline/tasks/01_clone_sources.py  --project all
python pipeline/tasks/02_convert_docs.py   --project all
python pipeline/tasks/03_build_indices.py  --project all
python pipeline/tasks/04_verify_indices.py --project all
python pipeline/tasks/05_evaluate.py       --project all

bash pipeline/serve.sh setup     # build the venv once
bash pipeline/serve.sh gradio    # chat UI on :8080
```

Needs `git`, `pandoc` and an `OPENAI_API_KEY`. Details, including the API call
and the one `.env` trap worth knowing about:
**[Running it locally](docs/running-locally.md)**.

## Where things land after a run

| What | Where |
|---|---|
| Service login and API token | last lines of the `stage-service` task log |
| Indices and reports | the pipeline vFolder, `<pipeline-name>-<id>/.pipeline/` |
| Retrieval health per source | `99_state/verify_report.json` |
| Answer quality scores | `99_state/eval_report.json` |
| Per-task success and timings | `99_state/run_manifest.json` |

## Documentation

| Article | What is in it |
|---|---|
| [Running it on FastTrack](docs/running-on-fasttrack.md) | Uploading, configuring and running the pipeline; where the outputs land; adapting the file to your cluster |
| [Running it locally](docs/running-locally.md) | The same DAG on one machine, one node at a time |
| [The DAG, node by node](docs/pipeline-nodes.md) | What each node reads, writes and fails on — and the timings from a real run |
| [Configuration](docs/configuration.md) | Every setting, its default, and the three places you can set it |
| [Data layout and path resolution](docs/paths-and-data.md) | Where the data lives and how each node hands it to the next |
| [Retrieval and answers](docs/retrieval.md) | Chunking, hybrid search, and the rules the answer is written under |
| [Evaluation and reports](docs/evaluation.md) | How answers are graded, and how to read the three reports |
| [The HTTP API](docs/http-api.md) | Endpoints, request fields, status codes |
| [Service authentication](docs/authentication.md) | Where the login and the API token come from, and how to pin your own |
| [Adding or changing a source](docs/adding-sources.md) | Pointing the pipeline at your own documentation |
| [Architecture and design decisions](docs/architecture.md) | How the code and the indices reach the services, and why the code looks the way it does |
| [Module reference and tests](docs/module-reference.md) | A file-by-file map, and how to run the suites |

## Common questions

**Do I need a GPU?** No. The batch nodes run a small CPU image and request no
accelerator — the embedding work is an HTTPS call to an API. See
[Adapting to your cluster](docs/running-on-fasttrack.md#adapting-to-your-cluster).

**There are two pipeline files. Which one?** `fasttrack_pipeline_inline.yaml`
unless you specifically need your API key kept out of the pipeline definition,
in which case use `fasttrack_pipeline.yaml`, which reads it from the Secrets
store or a `.env`. They describe the identical DAG.

**Is the chat UI open to anyone who has the URL?** No. Both services refuse to
start without credentials. Leave them blank and each run generates fresh ones,
prints them at the end of the `stage-service` log, and writes them to
`99_state/service_credentials.json`. See
[Service authentication](docs/authentication.md).

**I edited `sources.yaml` and nothing changed.** On FastTrack the nodes run the
checkout that `fetch-code` cloned — commit and push your change, then re-run
from `fetch-code`. Locally, re-run from `01_clone_sources`; if you changed only
the globs, `02_convert_docs` onward is enough.

**A clone hangs, then times out.** That is a private repository with no
`GITHUB_TOKEN`. Set the token, or make the source public.

**`verify-indices` failed.** Read `99_state/verify_report.json` — the symptom
table in [Evaluation and reports](docs/evaluation.md#the-reports) maps each
failure to what to change.

**My local run is writing to `/pipeline/...`.** Your `.env` still has the
`PIPELINE_*` mount paths set. `.env.example` ships them commented out for this
reason; comment them out again for local work.

**Can I use something other than OpenAI?** Yes — set `OPENAI_BASE_URL` to any
OpenAI-compatible endpoint, such as a vLLM server or a Backend.AI model service.
Every model name is a setting; see [Configuration](docs/configuration.md).

**Can I search just one documentation set?** Yes — pass
`{"projects": ["mlxcel"]}` in the request, or tick the boxes in the chat UI. An
unknown name is a `400`, not a silent search of everything else.

## What a run looks like

Measured on a 26.4.x cluster against the five default sources: 826 chunks
indexed in 27.7 s, all five canaries passing with top-hit L2 between 0.73 and
0.93 — each retrieving its own documentation, not merely scoring under the
threshold — and 20/20 graded questions scoring 0.94 overall, groundedness 0.96,
in 94 s. The node-by-node breakdown is in
[The DAG, node by node](docs/pipeline-nodes.md#what-a-run-looks-like).
