"""Service access credentials for the two deployment nodes.

Both apps read auth from three environment variables:

    GRADIO_USERNAME / GRADIO_PASSWORD   Gradio login
    API_KEY                             FastAPI bearer token

An empty value must never mean "no auth required" — that is how a RAG service
ends up publicly readable without anyone noticing. This module makes the
credentials exist unconditionally and makes them visible.

Policy:
  * If the environment (i.e. your .env) specifies all three, they are used
    verbatim and nothing is generated — the operator is in control.
  * Otherwise fresh credentials are generated once per pipeline run by the
    stage-service node. They are not cached: every run issues a new password and
    token, which the operator reads from that run's log.

The stage-service node and the deployment nodes run in *different containers*,
so a file is the only channel between them. Model storage (/models) is the only
mount both container kinds can see, so that is where the file goes. The file is
transport, never a cache — the staging node does not read it to decide what this
run's credentials are.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import stat
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from docs_rag.settings import MODEL_ROOT

log = logging.getLogger("credentials")

CREDENTIALS_FILENAME = "service_credentials.json"


def service_credentials_path() -> Path:
    """Where the staging node writes, and the services read."""
    return MODEL_ROOT / "99_state" / CREDENTIALS_FILENAME

ENV_GRADIO_USER = "GRADIO_USERNAME"
ENV_GRADIO_PASS = "GRADIO_PASSWORD"
ENV_API_KEY = "API_KEY"

DEFAULT_USERNAME = "admin"


class ServiceCredentials(BaseModel):
    """Access credentials for the two deployed services."""

    gradio_username: str = Field(..., description="Gradio login username")
    gradio_password: str = Field(..., description="Gradio login password")
    api_key: str = Field(..., description="FastAPI bearer token")
    source: str = Field(..., description="Where these came from: env, generated or file")

    @classmethod
    def generate(cls) -> "ServiceCredentials":
        """Fresh credentials for one pipeline run.

        token_urlsafe(24) is ~192 bits of entropy. The username is taken from
        the environment when set, so an operator can fix it without owning the
        password.
        """
        return cls(
            gradio_username=os.environ.get(ENV_GRADIO_USER, "").strip() or DEFAULT_USERNAME,
            gradio_password=secrets.token_urlsafe(24),
            api_key=secrets.token_urlsafe(32),
            source="generated",
        )

    def to_env(self) -> dict[str, str]:
        return {
            ENV_GRADIO_USER: self.gradio_username,
            ENV_GRADIO_PASS: self.gradio_password,
            ENV_API_KEY: self.api_key,
        }

    def export(self) -> None:
        """Put these into os.environ so an exec'd app inherits them."""
        os.environ.update(self.to_env())

    def banner(self, title: str, *, note: str = "") -> str:
        """Human-readable block for the task log.

        Deliberately prints the values: these are the credentials the operator
        needs to reach their own service, and the task log is visible only to
        the pipeline owner. OPENAI_API_KEY and GITHUB_TOKEN are never printed
        anywhere — preflight masks those to length only.
        """
        line = "=" * 72
        rows = [
            "",
            line,
            f"  {title}",
            line,
            f"  Gradio login    username : {self.gradio_username}",
            f"                  password : {self.gradio_password}",
            f"  FastAPI bearer  API_KEY  : {self.api_key}",
            "",
            f"  source: {self.source}",
        ]
        if note:
            rows.append(f"  {note}")
        rows += [line, ""]
        return "\n".join(rows)


def from_env() -> Optional[ServiceCredentials]:
    """Credentials fully specified in the environment.

    All three must be present. A half-filled .env is a misconfiguration, not a
    partial win — returning None sends the caller to generation so the services
    never come up half-protected.
    """
    user = os.environ.get(ENV_GRADIO_USER, "").strip()
    password = os.environ.get(ENV_GRADIO_PASS, "").strip()
    api_key = os.environ.get(ENV_API_KEY, "").strip()
    if user and password and api_key:
        return ServiceCredentials(
            gradio_username=user,
            gradio_password=password,
            api_key=api_key,
            source="env",
        )
    return None


def load(path: Path) -> Optional[ServiceCredentials]:
    """Read a credentials file written by the staging node, or None."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("ignoring unreadable credentials file %s: %s", path, exc)
        return None
    try:
        return ServiceCredentials(**{**data, "source": "file"})
    except Exception as exc:  # noqa: BLE001 - any malformed file is just unusable
        log.warning("ignoring malformed credentials file %s: %s", path, exc)
        return None


def save(creds: ServiceCredentials, path: Path) -> None:
    """Write credentials to `path` with owner-only permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = creds.model_dump(exclude={"source"})
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    # Best-effort: some mounted filesystems ignore chmod.
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError as exc:
        log.warning("could not restrict permissions on %s: %s", path, exc)


def allow_unauthenticated() -> bool:
    """Explicit opt-out of the fail-closed auth check."""
    return os.environ.get("ALLOW_UNAUTHENTICATED", "").strip().lower() in {"1", "true", "yes"}


def resolve_for_run() -> ServiceCredentials:
    """Credentials for this pipeline run: environment if complete, else fresh."""
    env_creds = from_env()
    if env_creds is not None:
        log.info("using service credentials from the environment — not generating new ones")
        return env_creds

    log.warning(
        "no complete service credentials in the environment — generating new ones "
        "for this run. Set %s, %s and %s in your .env to keep them stable across runs.",
        ENV_GRADIO_USER, ENV_GRADIO_PASS, ENV_API_KEY,
    )
    return ServiceCredentials.generate()


def resolve_for_serving(credentials_file: Path) -> Optional[ServiceCredentials]:
    """Credentials for a deployment node: environment if complete, else the file."""
    env_creds = from_env()
    if env_creds is not None:
        log.info("using service credentials from the environment")
        return env_creds
    return load(credentials_file)


def apply_for_serving(credentials_file: Path, service: str) -> Optional[ServiceCredentials]:
    """Load, export and announce this service's credentials. Fail closed.

    Called before the app starts, so the banner heads the deployment log and the
    app inherits the three variables.

    Raises SystemExit when no credentials can be found, unless
    ALLOW_UNAUTHENTICATED=1 — serving the corpus to anyone who can reach the
    port is not a reasonable default.
    """
    creds = resolve_for_serving(credentials_file)

    if creds is None:
        if allow_unauthenticated():
            log.warning(
                "%s: NO CREDENTIALS FOUND and ALLOW_UNAUTHENTICATED is set — starting "
                "WITHOUT authentication. Anyone who can reach this port has full access.",
                service,
            )
            return None
        raise SystemExit(
            f"{service}: no service credentials found.\n"
            f"  Looked in: the environment ({ENV_GRADIO_USER}/{ENV_GRADIO_PASS}/"
            f"{ENV_API_KEY}, normally from .env)\n"
            f"         and: {credentials_file}\n"
            "  That file is written by the stage-service node; if it is missing, that "
            "node did not run, or model storage was not mounted where it expected.\n"
            "  Refusing to serve unauthenticated. Set the three variables in .env, or "
            "set ALLOW_UNAUTHENTICATED=1 to override."
        )

    creds.export()
    log.info("%s", creds.banner(f"SERVICE ACCESS CREDENTIALS ({service})"))
    return creds
