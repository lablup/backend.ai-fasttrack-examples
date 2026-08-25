"""Configuration — one settings object, one loading mechanism.

Every tunable is an environment variable with a default, and the environment is
populated from a layered `.env` search. There is no config YAML: a second source
of truth for the same values is how a chunk size ends up different between the
node that builds an index and the node that queries it.

Load order (first hit wins):

    1. os.environ                    values typed into the pipeline YAML's
                                     `envs` block, GUI secrets, a shell export
    2. /models/.env                  model storage, as a serving container
                                     mounts it
    3. $PIPELINE_MODEL_STORAGE/.env  the same vfolder as a batch task mounts
                                     it, by name under /home/work
    4. /pipeline/vfroot/.env         the auto-created pipeline vFolder
    5. ./.env                        local development

Layers 2 and 3 are the same folder seen from the two container kinds, which is
why one uploaded .env serves both. FastTrack does not deliver a deployment
node's `envs` to its container at all, so for the services a file is the only
channel — the stage-service node writes one for exactly that reason.

"Set" means *set to something*. An empty value, or an unresolved
`${{ secrets.NAME }}` placeholder, counts as unset and falls through to the next
layer. Without that rule the YAML could not be shipped with a blank
`OPENAI_API_KEY:` line for the user to fill in — the blank would win over the
.env file and nothing downstream would ever see a key.
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger("settings")

# FastTrack substitutes `${{ secrets.NAME }}` inside `envs` values just before
# the task starts. When no such secret exists the literal text is passed
# through, and a variable holding it is worse than an unset one — it looks set.
PLACEHOLDER = re.compile(r"^\$\{\{.*\}\}$")


def is_unset(value: Optional[str]) -> bool:
    """True when a value carries no information: absent, blank, or a placeholder."""
    if value is None:
        return True
    stripped = value.strip()
    return not stripped or bool(PLACEHOLDER.match(stripped))

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

# Model storage. Selected in the Create Pipeline dialog rather than named in the
# YAML, and auto-mounted here — read-write in a batch task, read-only in a
# serving container. The only mount both container kinds share.
MODEL_ROOT = Path(os.environ.get("PIPELINE_MODEL_ROOT", "/models"))

# Model storage is mounted at /models in a serving container, but a batch task
# mounts the same vfolder by name — /home/work/<name>. Both are candidates, so a
# .env uploaded to that folder is found from either side.
MODEL_STORAGE = Path(os.environ.get("PIPELINE_MODEL_STORAGE", "/models"))

DOTENV_CANDIDATES = (
    MODEL_ROOT / ".env",
    MODEL_STORAGE / ".env",
    Path(os.environ.get("PIPELINE_VFROOT", "/pipeline/vfroot")) / ".env",
    PROJECT_ROOT / ".env",
)


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse KEY=VALUE lines. Dependency-free and deliberately literal.

    Blank lines and `#` comments are skipped; surrounding quotes are stripped.
    Everything after the first `=` is the value — an inline `# comment` is NOT
    stripped, because a `#` is legal inside a password or a URL fragment and
    guessing wrong there silently corrupts a secret.
    """
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key:
            values[key] = value.strip().strip('"').strip("'")
    return values


def clear_unset_env(names: Iterable[str]) -> List[str]:
    """Remove variables that are present but carry no value.

    The pipeline YAML ships every setting as a blank line for the user to fill
    in, and a cluster that propagates `envs` will export those blanks. Leaving
    them in place would mean a blank in the YAML silently beats a real value in
    .env — and, for the numeric settings, that an empty string reaches an int
    field. Deleting them restores the intended meaning: blank means "not set".
    """
    removed: List[str] = []
    for name in names:
        for candidate in (name, name.upper(), name.lower()):
            if candidate in os.environ and is_unset(os.environ[candidate]):
                del os.environ[candidate]
                removed.append(candidate)
    return removed


def load_layered_dotenv(candidates: Optional[List[Path]] = None) -> List[Path]:
    """Populate os.environ from the first-found value in each candidate file.

    Runs `clear_unset_env` over the known settings first, so blanks and
    unresolved placeholders do not block the file layers. A variable that holds
    a real value is never overridden — an explicit export, a GUI secret, or a
    value typed into the pipeline YAML always wins over a file.

    Returns the files that were read, for logging.
    """
    clear_unset_env(SETTING_ENV_NAMES)

    loaded: List[Path] = []
    for path in candidates if candidates is not None else DOTENV_CANDIDATES:
        try:
            if not path.is_file():
                continue
            values = parse_dotenv(path.read_text(encoding="utf-8"))
        except OSError as exc:
            # An unreadable /models on a cluster that mounts it differently is
            # not fatal — the next layer, or a real env var, may still supply it.
            log.warning("could not read %s: %s", path, exc)
            continue
        for key, value in values.items():
            if is_unset(os.environ.get(key)) and not is_unset(value):
                os.environ[key] = value
        loaded.append(path)
    if loaded:
        log.info("loaded env from: %s", ", ".join(str(p) for p in loaded))
    return loaded


class Settings(BaseSettings):
    """Runtime configuration. Field name upper-cased is the env var name."""

    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    # --- LLM / embeddings ---
    openai_api_key: str = ""
    # Empty = api.openai.com. Set for vLLM, a Backend.AI model service, etc.
    openai_base_url: str = ""
    llm_model: str = "gpt-4.1"
    embedding_model: str = "text-embedding-3-small"
    # Qwen-style chain of thought. Only applied when openai_base_url is set;
    # OpenAI models reject the extra body field.
    enable_thinking: bool = False
    temperature: float = 0.2
    max_tokens: int = 4096

    # --- indexing ---
    chunk_size: int = 1000
    chunk_overlap: int = 100

    # --- retrieval ---
    # Chunks pulled per project before global pooling.
    k_per_project: int = 10
    # Size of the pooled context handed to the LLM.
    global_top_k: int = 15
    # Reciprocal Rank Fusion constant; higher flattens weighting across ranks.
    rrf_k: int = 60
    # L2 confidence cutoff. Applied to the semantic arm only, so an exact
    # lexical match on an error code or a CLI flag still survives. Also node
    # 04's canary threshold.
    max_l2: float = 1.5
    max_chars_per_chunk: int = 4000

    # --- service auth ---
    api_key: str = ""
    gradio_username: str = ""
    gradio_password: str = ""

    # --- evaluation ---
    eval_judge_model: str = "gpt-4.1"
    # 0 or unset = score every question in the fixture.
    eval_sample_size: int = 0

    def llm_kwargs(self) -> dict:
        """Constructor kwargs shared by every ChatOpenAI in the package."""
        kwargs: dict = {
            "model": self.llm_model,
            "api_key": self.openai_api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.openai_base_url:
            kwargs["base_url"] = self.openai_base_url
            # Qwen-family servers read this; OpenAI rejects unknown body keys,
            # which is why it is gated on a custom endpoint.
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
            }
        return kwargs


# Variables the blank-means-unset rule applies to: every Settings field, plus
# the few things read directly from the environment rather than through Settings.
SETTING_ENV_NAMES = tuple(name.upper() for name in Settings.model_fields) + (
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "ALLOW_UNAUTHENTICATED",
)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The process-wide settings, loading the layered .env on first use.

    Cached rather than a module-level global so that importing this module has
    no side effects — tests can point DOTENV_CANDIDATES elsewhere and clear the
    cache.
    """
    load_layered_dotenv()
    return Settings()
