# Running it locally

Everything the pipeline does on a cluster, on your own machine — one node at a
time, in one folder.

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

> `.env.example` ships the `PIPELINE_*` mount paths **commented out** for this
> reason. Copying a populated FastTrack `.env` verbatim redirects a local run to
> `/pipeline/...`; comment them out for local work. The unit and e2e suites
> scrub these variables in their fixtures for exactly this reason.

Then serve it:

```bash
bash pipeline/serve.sh setup     # build the venv once (deployments do this
                                 # in the definition's pre_start_actions)
bash pipeline/serve.sh fastapi   # OpenAI-compatible API on :8080
bash pipeline/serve.sh gradio    # chat UI on :8080
```

```bash
curl -H "Authorization: Bearer $API_KEY" \
     -H 'Content-Type: application/json' \
     -d '{"messages":[{"role":"user","content":"How do I stop bssh on the first host failure?"}],
          "projects":["bssh"]}' \
     localhost:8080/v1/chat/completions
```

Requirements for a local run: `git`, `pandoc`, and an `OPENAI_API_KEY`. On
FastTrack `pipeline/bootstrap.sh` provides pandoc itself.

---

[← All documentation](../README.md)
