from __future__ import annotations

import pytest
from langgraph.checkpoint.conformance.capabilities import Capability, DetectedCapabilities

from riffdb_langgraph.saver import RiffDBCheckpointSaver


def test_all_conformance_capabilities_detected() -> None:
    saver = RiffDBCheckpointSaver("riffdb://langgraph-development")
    detected = DetectedCapabilities.from_instance(saver)
    assert detected.detected == set(Capability)


def test_saver_requires_secure_connection_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RIFFDB_LANGGRAPH_ENDPOINT", raising=False)
    monkeypatch.delenv("RIFFDB_ENDPOINT", raising=False)
    monkeypatch.delenv("RIFFDB_LANGGRAPH_CREDENTIAL_FILE", raising=False)
    monkeypatch.delenv("RIFFDB_CREDENTIAL_FILE", raising=False)
    saver = RiffDBCheckpointSaver("riffdb://langgraph-development")
    assert saver.database == "langgraph-development"
    assert saver.scope == "global"
    with pytest.raises(RuntimeError, match="Missing RiffDB LangGraph connection configuration"):
        saver.get_tuple({"configurable": {"thread_id": "t1"}})


def test_saver_validates_scope_bounds() -> None:
    with pytest.raises(ValueError, match=r"scope must be 1\.\.32 UTF-8 bytes"):
        RiffDBCheckpointSaver("riffdb://langgraph-development", scope="")
    with pytest.raises(ValueError, match=r"scope must be 1\.\.32 UTF-8 bytes"):
        RiffDBCheckpointSaver("riffdb://langgraph-development", scope="x" * 33)
