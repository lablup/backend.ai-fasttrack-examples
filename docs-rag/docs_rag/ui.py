"""Gradio chat UI over the same retriever the API uses.

Runs in-process against `Retriever`/`RAGChat` rather than calling the FastAPI
service, so the two deployments are independent — neither can take the other
down.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr

from docs_rag import credentials
from docs_rag.rag import RAGChat
from docs_rag.retriever import Retriever, format_context
from docs_rag.settings import Settings, get_settings, load_layered_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("ui")


def resolve_auth(settings: Settings) -> Optional[Tuple[str, str]]:
    """The Gradio login pair, or None when serving deliberately open."""
    if settings.gradio_username and settings.gradio_password:
        return (settings.gradio_username, settings.gradio_password)
    if os.environ.get("ALLOW_UNAUTHENTICATED", "").strip().lower() in {"1", "true", "yes"}:
        log.warning(
            "GRADIO_USERNAME/GRADIO_PASSWORD are empty and ALLOW_UNAUTHENTICATED is "
            "set — serving WITHOUT a login."
        )
        return None
    raise SystemExit(
        "refusing to start: no Gradio credentials.\n"
        "  Set GRADIO_USERNAME and GRADIO_PASSWORD in your .env, or let the "
        "stage-service pipeline node generate them (it prints them in its log).\n"
        "  To serve deliberately without a login, set ALLOW_UNAUTHENTICATED=1."
    )


def format_sources(chunks: List[dict]) -> str:
    """A compact markdown list of what retrieval actually returned."""
    if not chunks:
        return "_No documentation matched this question._"
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        if "similarity_score" in chunk:
            score = f"L2 {chunk['similarity_score']:.3f}"
        else:
            score = f"BM25 {chunk.get('bm25_score', 0.0):.3f}"
        lines.append(
            f"{i}. **{meta.get('project', '?')}** — `{meta.get('relative_path', '?')}` "
            f"· {meta.get('article_title', 'untitled')} · {score}"
        )
    return "\n".join(lines)


# One per indexed corpus. These are the same queries node 04 verifies the
# indices with, so each is known to retrieve its own documentation rather than
# being a plausible-looking guess.
EXAMPLE_QUESTIONS = [
    "What is Backend.AI?",
    "How do I mount a virtual folder into a session?",
    "How do I add a new model to mlxcel?",
    "How do I monitor GPU utilization across multiple nodes?",
    "How do I run a command on multiple hosts at once?",
]


def build_interface(retriever: Retriever, settings: Settings) -> gr.Blocks:
    projects = retriever.projects

    async def respond(message: str, history, selected: List[str], mode: str):
        if not message.strip():
            yield "", "_Ask a question to see which documents were used._"
            return

        chunks = await retriever.retrieve(
            message, selected or projects, mode=mode, top_k=settings.global_top_k
        )
        sources = format_sources(chunks)

        # A fresh chat per turn, seeded from this session's history: a shared
        # RAGChat would leak one browser tab's conversation into another's, but
        # without the seed every follow-up is answered as an isolated question.
        # `history` already has the pending question appended, hence [:-1].
        chat = RAGChat(retriever, settings)
        chat.seed_history((m["role"], m["content"]) for m in (history or [])[:-1])
        context = format_context(chunks, settings.max_chars_per_chunk)
        answer = ""
        async for token in chat.answer_with_context(message, context):
            answer += token
            yield answer, sources

    with gr.Blocks(title="docs-rag", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Backend.AI documentation assistant\n"
            "Answers are grounded in the indexed open-source documentation. "
            f"Corpora loaded: **{', '.join(projects) or 'none'}**."
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=520, type="messages")
                question = gr.Textbox(
                    placeholder="e.g. How do I run one command across several hosts?",
                    label="Question",
                    lines=2,
                )
                with gr.Row():
                    ask = gr.Button("Ask", variant="primary")
                    clear = gr.Button("Clear")
                # Fills the box rather than sending: a click that fires a request
                # gives no chance to pick the corpora or the retrieval mode first.
                gr.Examples(
                    examples=EXAMPLE_QUESTIONS,
                    inputs=question,
                    label="Example questions",
                )
            with gr.Column(scale=2):
                project_picker = gr.CheckboxGroup(
                    choices=projects, value=projects, label="Search in"
                )
                mode_picker = gr.Radio(
                    choices=["hybrid", "semantic", "lexical"],
                    value="hybrid",
                    label="Retrieval mode",
                    info="hybrid fuses both; lexical is exact-token BM25 only",
                )
                sources_box = gr.Markdown(
                    "_Ask a question to see which documents were used._",
                    label="Retrieved documents",
                )

        async def on_ask(message: str, history: list, selected: List[str], mode: str):
            history = (history or []) + [{"role": "user", "content": message}]
            yield history, "", "_Retrieving…_"
            async for answer, sources in respond(message, history, selected, mode):
                yield (
                    history + [{"role": "assistant", "content": answer}],
                    "",
                    sources,
                )

        inputs = [question, chatbot, project_picker, mode_picker]
        outputs = [chatbot, question, sources_box]
        ask.click(on_ask, inputs=inputs, outputs=outputs)
        question.submit(on_ask, inputs=inputs, outputs=outputs)
        clear.click(
            lambda: ([], "", "_Ask a question to see which documents were used._"),
            outputs=outputs,
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(prog="docs-rag-ui")
    parser.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", "8000")))
    parser.add_argument("--root-path", default=os.environ.get("GRADIO_ROOT_PATH", ""))
    args = parser.parse_args()

    # Order matters: populate the environment, then let the credentials file
    # fill in anything .env did not, and only then build the cached Settings.
    load_layered_dotenv()
    credentials.apply_for_serving(credentials.service_credentials_candidates(), "serve-gradio")

    settings = get_settings()
    auth = resolve_auth(settings)

    indices_root = Path(
        os.environ.get("DOCSRAG_INDICES", Path.cwd() / "pipeline" / "data" / "03_indices")
    )
    retriever = Retriever(indices_root, settings)
    if not retriever.load():
        log.error(
            "no indices loaded from %s — the UI will start but answer nothing. "
            "Run the build-indices node first.", indices_root,
        )

    demo = build_interface(retriever, settings)
    app, _, _ = demo.launch(
        server_name=args.host,
        server_port=args.port,
        root_path=args.root_path or None,
        auth=auth,
        prevent_thread_lock=True,
        show_api=False,
    )

    # Gradio has no health endpoint of its own, and the model service needs one
    # that answers 200 without a login.
    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "projects": retriever.projects}

    demo.block_thread()


if __name__ == "__main__":
    main()
