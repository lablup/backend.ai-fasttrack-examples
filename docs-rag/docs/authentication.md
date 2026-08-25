# Service authentication

Both services refuse to start without a login. Where the credentials come from,
and how to pin your own.

Both services **refuse to start without credentials**. An empty password meaning
"no login required" is how a documentation service ends up publicly readable
without anyone noticing.

- Set `GRADIO_USERNAME`, `GRADIO_PASSWORD` and `API_KEY` in your `.env` to pin
  them. All three or none — a half-filled login is ignored on purpose.
- Leave them blank and `stage-service` generates fresh ones per run, prints them
  at the end of its log, and writes them to `99_state/service_credentials.json`
  with mode 0600 — on the vfroot and mirrored into model storage. This is the
  recommended setting: a generated credential beats any default a shipped
  example could suggest.
- `ALLOW_UNAUTHENTICATED=1` serves deliberately open, with a warning.

The guards live at the **serving entry points** —
`server.require_auth_configured()` and `ui.resolve_auth()`, both called from
their module's `main()` — not at import, so
importing the modules stays side-effect free and the test suite can exercise
them. `verify_token` re-checks independently, so anything that serves the ASGI
object directly (`uvicorn docs_rag.server:app`, a gunicorn worker, an overridden
container command) still fails closed with a 503 rather than quietly serving the
corpus.

Those credentials are printed on purpose — they are yours, and the task log is
visible only to you. `OPENAI_API_KEY` and `GITHUB_TOKEN` are never printed
anywhere, only reported as set or unset with a character count.

---

[← All documentation](../README.md)
