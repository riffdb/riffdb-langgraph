"""Strict, secret-free LangGraph checkpointer URI parsing."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit


class InvalidCheckpointerURI(ValueError):
    """Raised before transport work when a LangGraph RiffDB URI is unsafe or malformed."""


@dataclass(frozen=True, slots=True)
class CheckpointerConfig:
    """The non-secret configuration carried by a LangGraph checkpointer URI."""

    database: str

    @classmethod
    def from_uri(cls, uri: str) -> CheckpointerConfig:
        parsed = urlsplit(uri)
        if parsed.scheme != "riffdb":
            raise InvalidCheckpointerURI("checkpointer URI must use the riffdb scheme")
        if parsed.username is not None or parsed.password is not None:
            raise InvalidCheckpointerURI("checkpointer URI must not contain credentials")
        if parsed.port is not None:
            raise InvalidCheckpointerURI("checkpointer URI must not contain a port")
        if not parsed.hostname:
            raise InvalidCheckpointerURI("checkpointer URI must name one database")
        if parsed.path not in ("", "/"):
            raise InvalidCheckpointerURI("checkpointer URI must not contain a path")
        if parsed.query:
            raise InvalidCheckpointerURI("checkpointer URI must not contain query parameters")
        if parsed.fragment:
            raise InvalidCheckpointerURI("checkpointer URI must not contain a fragment")
        if parsed.hostname != parsed.netloc:
            raise InvalidCheckpointerURI("checkpointer URI contains unsupported authority syntax")
        return cls(database=parsed.hostname)
