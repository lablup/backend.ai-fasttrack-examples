# Configuration

One mechanism — environment variables — three places to set them, and every
setting there is.

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
either. Copy [`.env.example`](../.env.example) and upload it to model storage.

Resolution order, first hit wins:

| Order | Source | Set by |
|---|---|---|
| 1 | `os.environ` | a value typed into a **batch** task's `envs`, a GUI secret, or a shell export |
| 2 | `/models/.env` | model storage, as a serving container mounts it |
| 3 | `$PIPELINE_MODEL_STORAGE/.env` | the same vFolder, as a batch task mounts it (`/home/work/<name>`) |
| 4 | `/pipeline/vfroot/.env` | you, uploading to the auto-created pipeline vFolder |
| 5 | built-in default | [`docs_rag/settings.py`](../docs_rag/settings.py) |

Layers 2 and 3 are one folder seen from the two container kinds, so a single
upload serves both.

> **FastTrack does not deliver a deployment node's `envs` to its container.**
> Anything you type into the `envs` block of `serve-fastapi` or `serve-gradio`
> is silently dropped, which is why `stage-service` generates a `.env` next to
> the staged code and mirrors it into model storage. The two services read their
> API key and their login from that file, never from the pipeline definition.
> The generated file is derived from `Settings.model_fields`, so a new setting
> cannot be added without reaching the deployment — and `GITHUB_TOKEN` and the
> `PIPELINE_*` paths are excluded by construction, since a serving container has
> no use for either.

> **Blank means unset.** An empty value, or an unresolved `${{ secrets.NAME }}`
> placeholder, falls through to the next layer instead of winning as an empty
> string. That is what makes it safe to ship the YAML with every setting listed
> and blank, and what stops a reference to a secret you never created from
> silently blanking your API key. Without it, a blank line in the YAML would
> also mean an empty string reaching an `int` field.

> **Put nothing after the value on a `KEY=VALUE` line in a `.env`.** Everything
> after the first `=` is the value; inline `#` comments are **not** stripped,
> because `#` is legal inside a password or a URL fragment and guessing wrong
> there silently corrupts a secret.

## Every setting

| Setting | Default | Effect | Rebuild? |
|---|---|---|---|
| `OPENAI_API_KEY` | — | **required**; used by node 03 and node 05 | — |
| `OPENAI_BASE_URL` | *(blank = api.openai.com)* | any OpenAI-compatible endpoint | no |
| `LLM_MODEL` | `gpt-4.1` | answers questions | no |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | builds the FAISS index | **yes** |
| `ENABLE_THINKING` | `false` | Qwen CoT; only applied when `OPENAI_BASE_URL` is set | no |
| `TEMPERATURE` | `0.2` | sampling temperature for answers | no |
| `MAX_TOKENS` | `4096` | answer length cap | no |
| `CHUNK_SIZE` | `1000` | tokens per chunk | **yes** |
| `CHUNK_OVERLAP` | `100` | tokens shared between neighbours | **yes** |
| `K_PER_PROJECT` | `10` | chunks pulled per corpus before pooling | no |
| `GLOBAL_TOP_K` | `15` | chunks sent to the model | no |
| `RRF_K` | `60` | fusion constant; higher flattens rank weighting | no |
| `MAX_L2` | `1.5` | semantic confidence cutoff, and node 04's canary threshold | no |
| `MAX_CHARS_PER_CHUNK` | `4000` | truncation per excerpt when formatting context | no |
| `API_KEY` | *(generated)* | FastAPI bearer token | no |
| `GRADIO_USERNAME` / `GRADIO_PASSWORD` | *(generated)* | Gradio login | no |
| `ALLOW_UNAUTHENTICATED` | *(unset)* | `1` serves both apps with no login | no |
| `EVAL_JUDGE_MODEL` | `gpt-4.1` | grading model, always at temperature 0 | no |
| `EVAL_SAMPLE_SIZE` | `0` | `0` grades every question in the fixture | no |
| `GITHUB_TOKEN` | *(blank)* | only for private documentation sources | no |
| `DOCSRAG_REF` | `main` | branch or tag `fetch-code` clones | — |
| `DOCSRAG_SRC` | `/pipeline/vfroot/src/docs-rag` | where the other nodes run from | — |
| `DOCSRAG_PORT` | `8080` | port both services listen on | — |

"Rebuild" means re-running `03_build_indices` and everything after it. There is
deliberately no config YAML on top of this: a second place to set a chunk size
is how an index gets built with one value and queried with another.

---

[← All documentation](../README.md)
