# The HTTP API

The OpenAI-compatible endpoints, the extra request fields, and what each status
code means.

[`docs_rag/server.py`](../docs_rag/server.py). Enough of the OpenAI Chat
Completions protocol that any OpenAI client library works unchanged.

| Endpoint | Auth | Notes |
|---|---|---|
| `POST /v1/chat/completions` | bearer | streaming (SSE) or not |
| `GET /v1/models` | bearer | one card, `docs-rag` |
| `GET /health` | none | 503 until at least one index is loaded |
| `GET /` | none | mirrors `/health` |

Request extensions on top of the standard fields:

| Field | Type | Default | Effect |
|---|---|---|---|
| `projects` | `list[str]` | every loaded index | which corpora to search |
| `retrieval_mode` | `"hybrid" \| "semantic" \| "lexical"` | `hybrid` | — |
| `top_k` | `int > 0` | `GLOBAL_TOP_K` | chunks sent to the model |
| `stream` | `bool` | `false` | SSE frames, `[DONE]` terminated |

Status codes are meant to be distinguishable:

- **`400`** — an explicitly named project that is not loaded. Every named
  project must exist; dropping the unknown ones and answering from the rest
  looks like success while quietly searching somewhere the caller did not ask
  for. The error lists what *is* available.
- **`401`** — missing, malformed or wrong bearer token. Compared with
  `secrets.compare_digest`.
- **`503`** — either no index is loaded (readiness), or the server is running
  with no `API_KEY` at all (misconfiguration). Both model definitions treat 200
  as ready, and answering 200 with no corpus would route traffic to a service
  that replies fluently from empty context.

Conversation history works as an OpenAI client expects: resend the whole
`messages` array and follow-ups resolve against the earlier turns.

CORS allows any origin with `allow_credentials=False` — this API authenticates
with a bearer header and sets no cookies, so credentialed CORS buys nothing and
pairing it with a wildcard origin is the footgun that would bite if a session
were ever added.

---

[← All documentation](../README.md)
