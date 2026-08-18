"""Retrieve, prompt, stream — the answering half of the system.

Deliberately small. It holds a retriever, a chat model, and a bounded message
history; everything else is in the prompt.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator, Iterable, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from docs_rag.retriever import Retriever, format_context
from docs_rag.settings import Settings

log = logging.getLogger("rag")

SYSTEM_PROMPT = """You answer questions about open-source Backend.AI projects using ONLY the documentation excerpts provided to you.

RULES

1. Use only information explicitly stated in the context below. Do not add
   knowledge from anywhere else, and do not guess.
2. If the context does not answer the question, say so plainly: "The provided
   documentation does not cover this." Then say what related information the
   context *does* contain, so the reader knows where to look next.
3. Never invent specific values. IP addresses, ports, file paths, hostnames,
   version numbers, environment variables, CLI flags and commands must appear
   verbatim in the context or not at all. A plausible-looking command that does
   not exist is worse than no answer.
4. Each excerpt is labelled with its project, source file and match score.
   Similarity is L2 distance, so lower is better; excerpts labelled "Lexical"
   matched by keyword rather than meaning. Prefer closer matches, and say so
   when every excerpt is a weak match.
5. Cite the source file for each substantive claim, e.g. (backendai:
   docs/install/index.rst).
6. The corpus spans several separate projects. Do not blend facts from
   different projects into one procedure unless the context says they interact.
"""


class RAGChat:
    """One conversation: retrieval, generation, and a bounded history."""

    def __init__(self, retriever: Retriever, settings: Settings, memory_turns: int = 6):
        self.retriever = retriever
        self.settings = settings
        # Two messages per turn. Bounded because the whole history is re-sent on
        # every request, so an unbounded one silently inflates cost and latency.
        self.max_messages = memory_turns * 2
        self.messages: List = []

        self.llm = ChatOpenAI(timeout=120, max_retries=3, **settings.llm_kwargs())
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("system", "Context:\n\n{context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def clear(self) -> None:
        self.messages = []

    def seed_history(self, turns: Iterable[Tuple[str, str]]) -> None:
        """Load a prior conversation from (role, content) pairs.

        Only user and assistant turns are kept. A caller-supplied system message
        is deliberately dropped: the grounding rules are this server's, and
        honouring an injected one would let a request opt out of them.
        """
        self.messages = []
        for role, content in turns:
            if not content:
                continue
            if role == "user":
                self.messages.append(HumanMessage(content=content))
            elif role == "assistant":
                self.messages.append(AIMessage(content=content))
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def _remember(self, question: str, answer: str) -> None:
        self.messages.append(HumanMessage(content=question))
        self.messages.append(AIMessage(content=answer))
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    async def answer(
        self,
        question: str,
        projects: Optional[List[str]] = None,
        mode: str = "hybrid",
        top_k: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Retrieve, then stream the answer token by token."""
        chunks = await self.retriever.retrieve(
            question,
            projects if projects is not None else self.retriever.projects,
            mode=mode,
            top_k=top_k,
        )
        context = format_context(chunks, self.settings.max_chars_per_chunk)
        async for token in self.answer_with_context(question, context):
            yield token

    async def answer_with_context(
        self, question: str, context: str
    ) -> AsyncGenerator[str, None]:
        """Stream an answer for a context block that was retrieved elsewhere.

        Kept separate so the evaluation harness can score generation against a
        context it controls, without re-running retrieval.
        """
        if not context.strip():
            context = "(no documentation excerpts matched this question)"

        collected: List[str] = []
        async for token in self.chain.astream(
            {
                "context": context,
                "question": question,
                "chat_history": self.messages,
            }
        ):
            collected.append(token)
            yield token

        # Store the assembled answer, not the last token. Appending the streamed
        # chunk here leaves the history holding a single trailing fragment.
        self._remember(question, "".join(collected))
