"""OpenAI-compatible HTTP API over the documentation corpus.

Speaks enough of the Chat Completions protocol that any OpenAI client library
can point at it unchanged, with a few extra request fields for choosing which
corpus to search and how.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, List, Literal, Optional, Tuple

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from docs_rag import credentials
from docs_rag.rag import RAGChat
from docs_rag.retriever import RETRIEVAL_MODES, Retriever
from docs_rag.settings import Settings, get_settings, load_layered_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("server")


# --------------------------------------------------------------------------
# Protocol models
# --------------------------------------------------------------------------


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "docs-rag"
    messages: List[Message]
    stream: bool = False

    # Extensions. All optional; each falls back to the configured default.
    projects: Optional[List[str]] = Field(
        default=None,
        description="Which corpora to search. Defaults to every loaded index.",
    )
    retrieval_mode: Optional[Literal["hybrid", "semantic", "lexical"]] = None
    # Positive only: zero would silently mean "use the default", and a negative
    # value reaches the retriever as a Python negative slice.
    top_k: Optional[int] = Field(default=None, gt=0)


class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "docs-rag"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------

security = HTTPBearer(auto_error=False)


def require_auth_configured(settings: Settings) -> None:
    """Refuse to start unauthenticated unless that was asked for explicitly.

    Called from the serving entrypoint, not at import, so that importing this
    module stays free of side effects.
    """
    if settings.api_key:
        return
    if os.environ.get("ALLOW_UNAUTHENTICATED", "").strip().lower() in {"1", "true", "yes"}:
        log.warning(
            "API_KEY is empty and ALLOW_UNAUTHENTICATED is set — serving WITHOUT "
            "authentication. Anyone who can reach this port can query the corpus."
        )
        return
    raise SystemExit(
        "refusing to start: API_KEY is not set.\n"
        "  Set API_KEY in your .env, or let the stage-service pipeline node "
        "generate one (it prints the value in its task log).\n"
        "  To serve deliberately without a login, set ALLOW_UNAUTHENTICATED=1."
    )


def verify_token(
    token: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> None:
    """Check the bearer token. Fails closed independently of the entrypoint.

    `require_auth_configured()` already refuses to boot without a key, but it
    runs in `main()` — so it is skipped by anything that serves the ASGI object
    directly (`uvicorn docs_rag.server:app`, a gunicorn worker, an overridden
    container command). Re-checking here means an unconfigured server returns
    503 rather than quietly serving the corpus to everyone.
    """
    expected = app.state.settings.api_key
    if not expected:
        if not credentials.allow_unauthenticated():
            raise HTTPException(
                status_code=503,
                detail="server is misconfigured: no API_KEY. Start it via "
                       "pipeline/serve.sh, or set ALLOW_UNAUTHENTICATED=1 to serve openly.",
            )
        return
    # compare_digest, not !=: constant-time comparison costs nothing here and
    # keeps the check correct if the token is ever shortened or made guessable.
    if token is None or not secrets.compare_digest(token.credentials, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(application: FastAPI):
    settings = get_settings()
    indices_root = Path(
        os.environ.get("DOCSRAG_INDICES", Path.cwd() / "pipeline" / "data" / "03_indices")
    )
    retriever = Retriever(indices_root, settings)
    loaded = retriever.load()
    if not loaded:
        log.error(
            "no indices loaded from %s — every request will return an empty context. "
            "Run the build-indices node first.", indices_root,
        )
    application.state.settings = settings
    application.state.retriever = retriever
    yield


app = FastAPI(title="docs-rag", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # False, not True: this API authenticates with a bearer header and sets no
    # cookies, so credentialed CORS buys nothing — and pairing it with a
    # wildcard origin is the footgun that would bite if a session were ever added.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resolve_projects(request: ChatCompletionRequest) -> List[str]:
    """Which corpora this request should search.

    An explicit list naming nothing we have is a client bug worth a 400 — far
    better than silently searching everything and returning a confident answer
    from the wrong corpus.
    """
    loaded = app.state.retriever.projects
    if request.projects is None:
        return loaded
    unknown = [p for p in request.projects if p not in loaded]
    if unknown:
        # Every explicitly named project must exist. Dropping the unknown ones
        # and answering from the rest looks like success while quietly searching
        # somewhere the caller did not ask for.
        raise HTTPException(
            status_code=400,
            detail=f"Unknown project(s) {unknown}. Available: {loaded}",
        )
    return list(request.projects)


def _last_user_message(request: ChatCompletionRequest) -> str:
    for message in reversed(request.messages):
        if message.role == "user":
            return message.content
    raise HTTPException(status_code=400, detail="No user message in the request")


def _prior_turns(request: ChatCompletionRequest) -> List[Tuple[str, str]]:
    """Every turn before the question being asked.

    OpenAI clients resend the whole conversation on each request, so answering
    only the last user line drops the context a follow-up depends on — "how do I
    stop it on the first failure?" is unanswerable without the turn that named
    the tool.
    """
    for index in range(len(request.messages) - 1, -1, -1):
        if request.messages[index].role == "user":
            return [(m.role, m.content) for m in request.messages[:index]]
    return []


def _new_chat(request: Optional[ChatCompletionRequest] = None) -> RAGChat:
    chat = RAGChat(app.state.retriever, app.state.settings)
    if request is not None:
        chat.seed_history(_prior_turns(request))
    return chat


@app.get("/")
async def root() -> dict:
    """Unauthenticated liveness probe.

    Mirrors /health so a deployment configured with either path passes. Gradio
    answers 200 at / because it serves its login page there, and a definition
    written for one service then quietly fails against the other.
    """
    return await health()


@app.get("/health")
async def health() -> dict:
    """Readiness, not liveness: 503 until at least one index is loaded.

    Both model definitions treat 200 as ready. Answering 200 with no corpus
    would route traffic to a service that replies fluently from empty context,
    which is worse than a deployment that never goes healthy.
    """
    retriever = getattr(app.state, "retriever", None)
    projects = retriever.projects if retriever else []
    if not projects:
        raise HTTPException(
            status_code=503,
            detail="no indices loaded — check that stage-service staged them",
        )
    return {"status": "ok", "projects": projects}


@app.get("/v1/models", dependencies=[Depends(verify_token)])
async def list_models() -> ModelList:
    return ModelList(
        data=[ModelCard(id="docs-rag", created=int(time.time()))]
    )


@app.post("/v1/chat/completions", dependencies=[Depends(verify_token)])
async def chat_completions(request: ChatCompletionRequest):
    question = _last_user_message(request)
    projects = _resolve_projects(request)
    mode = request.retrieval_mode or "hybrid"
    if mode not in RETRIEVAL_MODES:
        raise HTTPException(status_code=400, detail=f"retrieval_mode must be one of {RETRIEVAL_MODES}")

    chat = _new_chat(request)
    created = int(time.time())
    response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    if request.stream:
        return EventSourceResponse(
            _stream(chat, question, projects, mode, request, response_id, created)
        )

    parts: List[str] = []
    async for token in chat.answer(question, projects, mode=mode, top_k=request.top_k):
        parts.append(token)
    answer = "".join(parts)

    return ChatCompletionResponse(
        id=response_id,
        created=created,
        model=request.model,
        choices=[Choice(message=Message(role="assistant", content=answer))],
        # Approximate: this server does not see the provider's token accounting.
        usage=Usage(
            prompt_tokens=len(question) // 4,
            completion_tokens=len(answer) // 4,
            total_tokens=(len(question) + len(answer)) // 4,
        ),
    )


async def _stream(
    chat: RAGChat,
    question: str,
    projects: List[str],
    mode: str,
    request: ChatCompletionRequest,
    response_id: str,
    created: int,
) -> AsyncGenerator[dict, None]:
    def frame(delta: dict, finish_reason: Optional[str] = None) -> dict:
        return {
            "data": json.dumps(
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {"index": 0, "delta": delta, "finish_reason": finish_reason}
                    ],
                }
            )
        }

    yield frame({"role": "assistant"})
    try:
        async for token in chat.answer(question, projects, mode=mode, top_k=request.top_k):
            yield frame({"content": token})
    except Exception as exc:  # noqa: BLE001 - the stream is already open; report inside it
        log.exception("generation failed")
        yield frame({"content": f"\n\n[error: {type(exc).__name__}: {exc}]"})
    yield frame({}, finish_reason="stop")
    yield {"data": "[DONE]"}


def main() -> None:
    parser = argparse.ArgumentParser(prog="docs-rag-server")
    parser.add_argument("--host", default=os.environ.get("FASTAPI_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("FASTAPI_PORT", "8000")))
    args = parser.parse_args()

    # Order matters: populate the environment, then let the credentials file
    # fill in anything .env did not, and only then build the cached Settings.
    load_layered_dotenv()
    credentials.apply_for_serving(credentials.service_credentials_candidates(), "serve-fastapi")

    require_auth_configured(get_settings())
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
