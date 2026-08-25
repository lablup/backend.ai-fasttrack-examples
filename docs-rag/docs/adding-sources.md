# Adding or changing a source

Point the pipeline at your own documentation — one config file, no code
changes.

Edit [`pipeline/config/sources.yaml`](../pipeline/config/sources.yaml). No code
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

`include`/`exclude` are pathlib globs relative to the repo root: `**` matches
directories recursively, so `docs/**/*.md` also matches `docs/top-level.md`.
Note the trailing `/*` in an exclude — `docs/locales/**` matches the
*directories*, `docs/locales/**/*` matches the files inside them.

They are globs rather than a single "docs directory" because real repositories
do not agree on where documentation lives — of the five defaults, one keeps it
under `docs/` next to 129 translation catalogs that must be skipped, one keeps
its real documentation at the repo root, and one buries 42 benchmark dumps in
`docs/`.

`verify_query` becomes node 04's canary. Choose something only the right
document could answer: "How do I mount a virtual folder into a session?" is a
better canary than "What is X?".

`.rst` is converted with pandoc, `.md` is copied through; routing is by
extension, so a repo can mix both. A source whose globs match nothing fails the
node loudly rather than producing an empty index that only reveals itself at
query time.

Private repositories work too: set `GITHUB_TOKEN` in your `.env` and the clone
node injects it. It runs git with `GIT_TERMINAL_PROMPT=0`, so a missing token
fails immediately instead of hanging forever on a credential prompt nothing will
answer.

**After editing**, re-run from `clone-docs`; if you changed only globs,
`convert-docs` onward is enough. On FastTrack, remember that the nodes run the
checkout `fetch-code` cloned — commit and push, then re-run from `fetch-code`.

---

[← All documentation](../README.md)
