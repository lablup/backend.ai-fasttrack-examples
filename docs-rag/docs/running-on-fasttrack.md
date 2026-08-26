# Running it on FastTrack

Upload one file, change three or four values, press Run. The long form of the
quickstart in the README.

**Pick a pipeline file.** Both describe the identical DAG and differ only in
where your API key comes from:

| File | Your key lives in | Upload anything? |
|---|---|---|
| [`fasttrack_pipeline_inline.yaml`](../fasttrack_pipeline_inline.yaml) | this file, typed in | **no** |
| [`fasttrack_pipeline.yaml`](../fasttrack_pipeline.yaml) | the Secrets store, or a `.env` | only if you choose `.env` |

Use the inline file unless you specifically want the key out of the pipeline
definition. Every setting in it is already filled in with its default, so
`OPENAI_API_KEY` is the only line you have to touch.

Then:

1. **Fill in `OPENAI_API_KEY`** at the top of your chosen file.
2. **Create an empty model storage vFolder** and put its name in the three
   places the file marks `your_model_folder`: the `mounts:` list on the
   `stage-service` task, `PIPELINE_MODEL_STORAGE`, and each serving node's
   `service.model`. You never upload anything into it — `stage-service` writes
   the model definitions, a generated `.env` and this run's credentials there
   itself.
3. **Set `project` and `scaling-group`** to your cluster's, if they are not
   `default`.
4. **Create the pipeline**, then **Dry Run** and **Run**.

The vFolder name is the one account-scoped value in the file, and it is needed
because a deployment resolves `model_definition_path` **relative to its model
mount** — so the definition has to be inside that folder however the rest is
laid out. Everything else the services need travels through `/pipeline/vfroot`.
See [How it stays portable](architecture.md).

> Both files declare the same service names (`docs-rag-api`, `docs-rag-ui`), and
> service names must be unique within a resource group. If you want to run both
> variants at once, rename the services in one of them.

The `model-definition-*.yaml` files are written into your model vFolder by the
`stage-service` node on every run, **replacing** what is there. That is
deliberate: the deployment reads its start command and port from them, and a
copy left over from an earlier run silently pins the old values. Customise them
in the checkout, not in the vFolder.

## Where to find things afterwards

| What | Where |
|---|---|
| Service login and API token | last lines of the `stage-service` task log |
| Indices and reports | the pipeline vFolder, `<pipeline-name>-<id>/.pipeline/` |
| Retrieval health per source | `99_state/verify_report.json` |
| Answer quality scores | `99_state/eval_report.json` |
| Per-task success and timings | `99_state/run_manifest.json` |

## Adapting to your cluster

`fasttrack_pipeline.yaml` carries the shape a 26.4.x cluster accepted. If yours
follows the published schema, change these:

| Field | This file | Published schema |
|---|---|---|
| `version` | `26.4.4rc6` | your cluster's release |
| ownership | `domain:` + `scope: user` | `domain_name:` + `scope: project` |
| serving nodes | `type: deployment` | `type: serving` |
| `scaling-group` | `nvidia-H100` | your resource group |
| `project` | `H100` | your project |
| model vFolder | `your_model_folder` | the empty vFolder you created |

Export an existing pipeline from your cluster to see which shape it uses.

The batch nodes run a CPU image (`cr.backend.ai/testing/python:3.11-ubuntu22.04`,
2–4 CPU / 4–8 GB) and request no GPU. Embedding is an HTTPS call to a
third-party API, so nothing in this pipeline touches an accelerator — and a CPU
image schedules on any cluster.

> **`/bin/sh` is dash** on Ubuntu 20.04/22.04/24.04 images, so task commands use
> `set -eu`, not `set -euo pipefail`. Some images also ship Python without
> `ensurepip`; `bootstrap.sh` falls back to `venv --without-pip` plus
> `get-pip.py` when that happens.

---

[← All documentation](../README.md)
