# Architecture and design decisions

Why the code travels by git clone, how it reaches the two services, and the
reasoning behind the choices in the code.

## How it stays portable

Three mechanisms replace the usual "mount a code vFolder" approach.

**The code arrives by git clone.** The `fetch-code` node — the only one that is
pure shell, because it is the node that fetches the Python — clones this repo
into `/pipeline/vfroot/src`, which FastTrack creates automatically. Every later
node runs from that one checkout, so the whole DAG is guaranteed to run the same
revision. Pin it by setting `DOCSRAG_REF` to a tag.

**The code and indices reach the services through `/pipeline/vfroot`.** That is
the one `/pipeline` mount which outlives the run, and a serving container can
read it for the whole of its lifetime. A task's own `/pipeline/outputs` cannot
be used for this: a deployment can read it once, but the handle goes stale when
the upstream container ends and the service dies mid-startup on `ESTALE`. So
`stage-service` puts the code tree, the indices and the credentials on the
vfroot, and mirrors only four small files — the two model definitions, the
generated `.env` and the credentials file — into your model vFolder, because a
deployment resolves `model_definition_path` relative to its model mount and
would not find them anywhere else. Under FastTrack, an unwritable model mount is
a **hard failure** at this node: succeeding would surface the same wiring error
later as two unrelated-looking deployment failures.

**Config arrives layered**, as described above, so it works whichever of those
mounts a given cluster actually provides.

Output stays cumulative down the chain because `runner.run_task()` seeds each
node's output from the previous node's, but each node declares *which* stages it
needs. Without that narrowing every node would drag the cloned repos and the
converted markdown through the output mount — hundreds of megabytes per hop for
data nothing downstream reads.

### The two deployments

Both `model-definition-*.yaml` files are the same file with a different argument
to `serve.sh`:

- `model_path: /models` — the model mount itself. The platform validates this,
  and a path outside the mount gets the definition rejected in favour of the
  default runtime (which is how a service ends up trying to launch vLLM).
- `pre_start_actions` runs `serve.sh setup`, which builds the virtualenv before
  the service is expected to listen. A fresh container has no venv, and building
  one inline kept the port closed long enough for the platform to kill it.
- `start_command` points at `/pipeline/vfroot/docs-rag/pipeline/serve.sh`, so
  the code and indices come from the persistent mount rather than the model one.
- `port: 8080`, health check on `/` with a 300-second initial delay.
- `runtime_variant: custom` on the pipeline side. Left at its default, the
  platform tries to start vLLM instead.

The services run the full-app requirements
([`requirements-service.txt`](../requirements-service.txt)), not the lean ingestion
set.

## Design decisions

The short version of why the code looks the way it does.

| Decision | Because |
|---|---|
| No config YAML — environment variables only | a second place to set a chunk size is how an index gets built with one value and queried with another |
| Blank counts as unset | so the shipped YAML can list every setting blank, and an unresolved secret reference cannot blank your API key |
| RRF instead of score normalisation | L2 and BM25 share no scale, and normalising requires knowing the corpus; ranking does not |
| L2 cutoff on the semantic arm only | a lexical hit has no distance, and dropping it discards the queries BM25 exists to answer |
| Dual canary in verify | a FAISS index built from empty files still returns its nearest neighbour, so either check alone passes on a broken index |
| Judge **and** SemScore | each fails differently; the judge can be argued into agreeing, SemScore cannot but is blind to correctness |
| Groundedness weighted highest | a fluent, relevant, ungrounded answer is the failure this system exists to prevent |
| Fail closed on credentials | an empty password meaning "no login" is how a service ends up publicly readable |
| Structural head detection | `BACKENDAI_PIPELINE_JOB_INDEX` is not a DAG position |
| Stage to vfroot, not outputs | `/pipeline/outputs` goes stale when the upstream container ends; the service then dies on `ESTALE` |
| Hard-fail publish under FastTrack | publishing nothing while reporting success is worse than failing |
| Globs, not a docs directory | real repositories do not agree on where documentation lives |
| A source matching nothing fails loudly | an empty index only reveals itself at query time |
| Errors raised, not swallowed, in the indexer | a silently missing index becomes a confusing empty-retrieval bug three nodes later |
| Caller system messages dropped in `seed_history` | the grounding rules are the server's; honouring an injected one lets a request opt out of them |
| Bounded chat history | the whole history is re-sent per request, so unbounded silently inflates cost and latency |

---

[← All documentation](../README.md)
