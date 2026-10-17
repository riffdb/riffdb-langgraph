"""Disposable real-driver conformance and StateGraph smoke runner used by ``riffdb dev --exec``."""

from __future__ import annotations

import asyncio
import json
import operator
import os
import sys
from collections.abc import AsyncGenerator
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.conformance import checkpointer_test, validate
from langgraph.checkpoint.serde.types import TASKS
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from riffdb_langgraph.saver import RiffDBCheckpointSaver


class _AgentState(TypedDict):
    messages: Annotated[list[str], operator.add]
    items: list[str]
    processed: Annotated[list[str], operator.add]


def _plan_node(state: _AgentState) -> dict[str, Any]:
    return {
        "messages": [f"planned:{len(state.get('items', []))}"],
    }


def _fanout_edges(state: _AgentState) -> list[Send]:
    return [Send("worker", {"item": item}) for item in state.get("items", [])]


def _worker_node(state: Any) -> dict[str, Any]:
    item = str(state["item"])
    return {
        "processed": [item.upper()],
        "messages": [f"worker:{item}"],
    }


def _finalize_node(state: _AgentState) -> dict[str, Any]:
    joined = ",".join(sorted(state.get("processed", [])))
    return {
        "messages": [f"done:{joined}"],
    }


def _build_graph(saver: RiffDBCheckpointSaver) -> Any:
    builder: StateGraph[Any, Any, Any, Any] = StateGraph(_AgentState)
    builder.add_node("plan", _plan_node)
    builder.add_node("worker", _worker_node)
    builder.add_node("finalize", _finalize_node)
    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", _fanout_edges, ["worker"])
    builder.add_edge("worker", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=saver)


def _run_sync_smoke(uri: str) -> dict[str, Any]:
    with RiffDBCheckpointSaver(uri, scope="sync-smoke") as saver:
        graph = _build_graph(saver)
        thread_id = "sync-thread-1"
        config: Any = {"configurable": {"thread_id": thread_id}}
        out = graph.invoke(
            {"messages": ["init"], "items": ["alpha", "beta"], "processed": []},
            config=config,
        )
        assert out["processed"] == ["ALPHA", "BETA"]
        assert out["messages"][0] == "init"
        assert out["messages"][-1] == "done:ALPHA,BETA"

        snapshot = graph.get_state(config)
        assert snapshot.values["processed"] == ["ALPHA", "BETA"]

        history = list(graph.get_state_history(config))
        assert len(history) >= 3

        # Fork/update state and verify time-travel
        graph.update_state(config, {"messages": ["manual-note"]})
        updated = graph.get_state(config)
        assert updated.values["messages"][-1] == "manual-note"

        # Copy thread and prune sync
        saver.copy_thread(thread_id, "sync-thread-1-copy")
        copied_tuple = saver.get_tuple({"configurable": {"thread_id": "sync-thread-1-copy"}})
        assert copied_tuple is not None
        assert copied_tuple.checkpoint["channel_values"]["messages"][-1] == "manual-note"

        saver.prune(["sync-thread-1-copy"], strategy="keep_latest")
        pruned_list = list(saver.list({"configurable": {"thread_id": "sync-thread-1-copy"}}))
        assert len(pruned_list) == 1

        saver.delete_thread("sync-thread-1-copy")
        assert saver.get_tuple({"configurable": {"thread_id": "sync-thread-1-copy"}}) is None

        # Verify v<4 pending_sends migration on get_tuple
        legacy_tid = "sync-legacy-v3"
        parent_cp = empty_checkpoint()
        parent_cp["v"] = 3
        parent_cfg = saver.put(
            {"configurable": {"thread_id": legacy_tid, "checkpoint_ns": ""}},
            parent_cp,
            {"source": "input", "step": 0},
            {},
        )
        saver.put_writes(parent_cfg, [(TASKS, "migrated-send-payload")], task_id="task-legacy")
        child_cp = empty_checkpoint()
        child_cp["v"] = 3
        child_cfg = saver.put(
            parent_cfg,
            child_cp,
            {"source": "loop", "step": 1},
            {},
        )
        migrated_tuple = saver.get_tuple(child_cfg)
        assert migrated_tuple is not None
        assert migrated_tuple.checkpoint["channel_values"].get(TASKS) == ["migrated-send-payload"]
        saver.delete_thread(legacy_tid)

        return {
            "sync_messages": out["messages"],
            "sync_processed": out["processed"],
            "sync_history_depth": len(history),
        }


async def _run_async_smoke(uri: str) -> dict[str, Any]:
    async with RiffDBCheckpointSaver(uri, scope="async-smoke") as saver:
        graph = _build_graph(saver)
        thread_id = "async-thread-1"
        config: Any = {"configurable": {"thread_id": thread_id}}
        out = await graph.ainvoke(
            {"messages": ["start"], "items": ["gamma", "delta"], "processed": []},
            config=config,
        )
        assert out["processed"] == ["GAMMA", "DELTA"]
        assert out["messages"][-1] == "done:DELTA,GAMMA"

        snapshot = await graph.aget_state(config)
        assert snapshot.values["processed"] == ["GAMMA", "DELTA"]

        history = [s async for s in graph.aget_state_history(config)]
        assert len(history) >= 3

        return {
            "async_messages": out["messages"],
            "async_processed": out["processed"],
            "async_history_depth": len(history),
        }


async def _run_conformance(uri: str) -> dict[str, Any]:
    suite_counter = 0

    @checkpointer_test(name="RiffDBCheckpointSaver")
    async def _factory() -> AsyncGenerator[RiffDBCheckpointSaver]:
        nonlocal suite_counter
        suite_counter += 1
        scope = f"conf-{suite_counter}"
        async with RiffDBCheckpointSaver(uri, scope=scope) as saver:
            yield saver

    report = await validate(_factory)
    if not report.passed_all():
        failures: dict[str, Any] = {}
        for cap, res in report.results.items():
            if not res.passed:
                failures[cap] = {
                    "detected": res.detected,
                    "passed": res.tests_passed,
                    "failed": res.tests_failed,
                    "failures": res.failures,
                }
        raise RuntimeError(f"LangGraph checkpointer conformance failed: {json.dumps(failures)}")

    summary: dict[str, dict[str, Any]] = {}
    total_passed = 0
    for cap, res in sorted(report.results.items()):
        assert res.detected, f"Expected capability {cap} to be detected"
        assert res.passed is True, f"Expected capability {cap} to pass"
        total_passed += res.tests_passed
        summary[cap] = {
            "detected": res.detected,
            "passed": res.passed,
            "tests_passed": res.tests_passed,
            "tests_failed": res.tests_failed,
        }
    return {
        "all_passed": report.passed_all(),
        "total_passed": total_passed,
        "capabilities": summary,
    }


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    endpoint = args[0] if len(args) >= 1 else os.environ.get("RIFFDB_ENDPOINT", "")
    cred_file = args[1] if len(args) >= 2 else os.environ.get("RIFFDB_CREDENTIAL_FILE", "")
    database = (
        args[2] if len(args) >= 3 else os.environ.get("RIFFDB_DATABASE", "langgraph-development")
    )
    if not endpoint or not cred_file:
        raise SystemExit(
            "usage: python -m riffdb_langgraph <endpoint> <credential_file> [database]"
        )

    os.environ["RIFFDB_LANGGRAPH_ENDPOINT"] = endpoint
    os.environ["RIFFDB_LANGGRAPH_CREDENTIAL_FILE"] = cred_file
    if os.environ.get("RIFFDB_TENANT") == "":
        os.environ.pop("RIFFDB_TENANT", None)

    uri = f"riffdb://{database}"
    sync_summary = _run_sync_smoke(uri)
    async_summary = asyncio.run(_run_async_smoke(uri))
    conformance_summary = asyncio.run(_run_conformance(uri))

    output = {
        "database": database,
        "conformance": conformance_summary,
        "sync_smoke": sync_summary,
        "async_smoke": async_summary,
    }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
