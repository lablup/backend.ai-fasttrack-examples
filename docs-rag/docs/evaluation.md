# Evaluation and reports

How the pipeline grades its own answers, and how to read the three reports it
leaves behind.

## Evaluation

`05_evaluate` scores the committed question set in
[`pipeline/config/eval_samples.json`](../pipeline/config/eval_samples.json) — 20
questions, each with a known-good answer and its source file:

```json
{
  "question": "What are the local TOML configuration files for Backend.AI components called?",
  "answer": "Each service component reads a TOML file named after itself ...",
  "project": "backendai",
  "source": "docs/concepts/configuration.rst"
}
```

A sample with a `project` field is retrieved **against that corpus only**, which
keeps the score a measure of that corpus rather than of cross-corpus contention.

Two independent signals are computed, because each fails differently:

- **An LLM judge** reads the answer against the retrieved context. It catches
  unsupported claims — but it is another language model and can be talked into
  agreeing. Run at temperature 0 so a re-run reproduces the scores.
- **SemScore**, cosine similarity between the answer's embedding and the
  reference answer's. Mechanical, cheap and blind to correctness, but impossible
  to argue with.

| Metric | Weight | What the judge is told |
|---|---|---|
| `relevance` | 0.20 | Does it address the question asked? |
| `groundedness` | **0.35** | Is every claim supported by the excerpts? Penalise invented commands, paths, flags, ports and versions heavily. An answer that correctly says the documentation does not cover the question scores **high** here |
| `completeness` | 0.20 | Does it cover the substance of the reference? A different but equally valid approach is fine — no word-for-word match required |
| `usability` | 0.10 | Could a reader act on it? Concrete steps and citations score well |
| `semscore` | 0.15 | embedding cosine similarity, not a judge score |

Groundedness carries the highest weight because a fluent, relevant, ungrounded
answer is the failure mode this whole system exists to prevent.

**Adding questions** is appending to the JSON. Write the reference answer from
the documentation, not from memory, and name the `source` file — a question
whose answer is not actually in the corpus measures nothing but the model's
prior knowledge. Set `EVAL_SAMPLE_SIZE` to cap how many are scored while
iterating; `0` scores all of them.

## The reports

All three land in `99_state/`.

**`verify_report.json`** — did retrieval work at all?

```json
{
  "l2_threshold": 1.5,
  "results": {
    "bssh": {
      "query": "How do I run a command on multiple hosts at once?",
      "passed": true,
      "top_l2": 0.926,
      "top_source": "README.md",
      "lexical_hits": 5,
      "error": null
    }
  }
}
```

`top_source` is the field to read. A canary can pass on distance while
retrieving the wrong document — check that the file named is the one you would
have picked by hand.

| Symptom | Cause | Fix |
|---|---|---|
| `passed: false`, high `top_l2` | the corpus does not contain the answer | check `include` globs, or rewrite `verify_query` |
| `lexical_hits: 0` | the query shares no words with any document | rephrase using the documentation's own vocabulary |
| passes, but `top_source` is wrong | noise in the index | tighten `exclude` — changelogs, benchmark dumps and translation catalogues are the usual culprits |

**`eval_report.json`** — are the answers any good? Carries `aggregate` (the five
metrics plus `overall` and `mean_top_l2`) and one entry per question with its
own scores, `top_l2`, `retrieved_chunks` and `response_chars`.

| Result | What to change |
|---|---|
| low groundedness | raise `GLOBAL_TOP_K`, or lower `CHUNK_SIZE` for more focused chunks |
| low completeness | raise `GLOBAL_TOP_K` or `K_PER_PROJECT` |
| low relevance | usually retrieval, not generation — read `verify_report.json` first |
| low SemScore alone | probably nothing. It compares wording to one reference; a correct answer phrased differently scores lower without being worse |

**`run_manifest.json`** — one appended entry per task with its status, timing and
returned summary. The first place to look when a run half-succeeded.

---

[← All documentation](../README.md)
