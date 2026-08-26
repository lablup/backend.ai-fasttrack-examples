# Data layout and path resolution

Where the data lives, how each node finds it, and how one node's output becomes
the next node's input on FastTrack.

[`pipeline/paths.py`](../pipeline/paths.py) is the single source of truth. Under
whichever data root a task resolves:

```
<data-root>/01_repos/<source>/      cloned repositories       (node 01)
<data-root>/02_docs_md/<source>/    converted markdown        (node 02)
<data-root>/03_indices/<source>/    index.faiss, index.pkl, bm25.pkl (node 03)
<data-root>/99_state/               reports, manifest, credentials
```

Environment variables, highest priority first:

| Variable | Meaning | FastTrack value |
|---|---|---|
| `--data-root` (CLI) | overrides everything | — |
| `PIPELINE_OUTPUT_ROOT` | where this task **writes** | `/pipeline/outputs` |
| `PIPELINE_DATA_ROOT` | single-root fallback | — |
| `PIPELINE_INPUT_ROOT` | previous task's output | `/pipeline/input1` |
| `PIPELINE_VFROOT` | persistent shared folder | `/pipeline/vfroot` |
| `PIPELINE_MODEL_STORAGE` | model vFolder, as a batch task mounts it | `/home/work/<name>` |

**With none of these set** — a local run — output root equals input root equals
`./pipeline/data`, no seeding happens, and every node reads and writes one
shared tree.

**Input seeding.** On FastTrack each task gets a fresh, empty
`/pipeline/outputs`, so without help every node would start from nothing.
`runner.run_task()` calls `seed_from_input()` to copy the previous task's output
forward, which keeps the output cumulative down the chain. Each node declares
*which stages* it needs (`seed_stages=`), because seeding everything would drag
the cloned repositories and the converted markdown through every hop — hundreds
of megabytes for data nothing downstream reads.

**Head-node detection is structural.** Node 01 passes `seed_stages=()`: it
declares that it consumes nothing, so seeding copies nothing regardless of what
FastTrack mounts. Relying on "the head is given no `/pipeline/input1`" left a
stale or empty mount free to be copied forward.

> **Do not gate seeding on `BACKENDAI_PIPELINE_JOB_INDEX == "1"`.** That value is
> not a reliable DAG position — a production cluster reported `1` for
> `02_convert_docs`, which skipped seeding chain-wide and killed the run on an
> empty `01_repos`. Treat it only as "am I on FastTrack?". A node that declares
> stages and seeds none logs a warning naming the likely cause.

**Preflight.** Every node proves its preconditions and logs its runtime picture
before doing work — the fix for two production runs that died with no usable
diagnostics. `bootstrap.sh` prints the checkout's git HEAD, the resolved
`PIPELINE_*` paths and secret presence (length only), and verifies the task
script exists **before** the venv bootstrap, so a stale checkout fails in about
a second with an `ls` of `pipeline/tasks/` rather than after a four-minute
install. `runner.run_task()` then logs the roots, their contents, the job index
and what seeding carried forward; nodes call `require_env`, `require_binary`,
`require_stage` and `require_writable`. `require_stage` names the node that
should have produced a missing input.

---

[← All documentation](../README.md)
