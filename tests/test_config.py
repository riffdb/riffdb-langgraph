from __future__ import annotations

import pytest

from riffdb_langgraph.config import CheckpointerConfig, InvalidCheckpointerURI


def test_accepts_only_database_authority() -> None:
    assert CheckpointerConfig.from_uri("riffdb://langgraph-development") == CheckpointerConfig(
        database="langgraph-development"
    )


@pytest.mark.parametrize(
    "uri",
    [
        "postgresql://langgraph",
        "riffdb://",
        "riffdb://user@langgraph",
        "riffdb://user:secret@langgraph",
        "riffdb://langgraph:7443",
        "riffdb://langgraph/profile",
        "riffdb://langgraph?role=admin",
        "riffdb://langgraph#fragment",
    ],
)
def test_rejects_noncanonical_or_secret_bearing_uri(uri: str) -> None:
    with pytest.raises(InvalidCheckpointerURI):
        CheckpointerConfig.from_uri(uri)
