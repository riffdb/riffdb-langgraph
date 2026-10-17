from __future__ import annotations

import operator
from types import SimpleNamespace
from typing import Annotated, Any, TypedDict

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.serde.types import ERROR, INTERRUPT, TASKS
from langgraph.graph import END, START, StateGraph

from riffdb_langgraph.generated.client import (
    Checkpoint,
    CheckpointBlob,
    CheckpointWrite,
)
from riffdb_langgraph.saver import (
    RiffDBCheckpointSaver,
    _chunk_blobs,
    _chunk_writes,
    _make_blob_key,
)


class FakeLangGraphClient:
    def __init__(self) -> None:
        self.checkpoints: dict[tuple[str, str, str, str], Checkpoint] = {}
        self.blobs: dict[tuple[str, str, str, str, str], CheckpointBlob] = {}
        self.writes: dict[tuple[str, str, str, str, str, int], CheckpointWrite] = {}
        self.commit_seq = 0

    def with_read_after_commit(self, commit_seq: int | None) -> FakeLangGraphClient:
        _ = commit_seq
        return self

    def _next_commit(self) -> int:
        self.commit_seq += 1
        return self.commit_seq

    def put_checkpoint(self, command: Any) -> SimpleNamespace:
        key = (command.scope, command.thread_id, command.checkpoint_ns, command.checkpoint_id)
        prev = self.checkpoints.get(key)
        rev = (prev.revision + 1) if prev is not None else 1
        cp = Checkpoint(
            scope=command.scope,
            thread_id=command.thread_id,
            checkpoint_ns=command.checkpoint_ns,
            checkpoint_id=command.checkpoint_id,
            parent_checkpoint_id=command.parent_checkpoint_id,
            checkpoint_serde_type=command.checkpoint_serde_type,
            checkpoint_payload=command.checkpoint_payload,
            metadata_serde_type=command.metadata_serde_type,
            metadata_payload=command.metadata_payload,
            revision=rev,
        )
        self.checkpoints[key] = cp
        return SimpleNamespace(
            commit_sequence=self._next_commit(), outcome=SimpleNamespace(checkpoint=cp)
        )

    def put_checkpoint_with_blobs(self, command: Any) -> SimpleNamespace:
        for b in command.blobs:
            bkey = (
                command.scope,
                command.thread_id,
                command.checkpoint_ns,
                b.checkpoint_id,
                b.blob_key,
            )
            self.blobs[bkey] = b
        return self.put_checkpoint(command)

    def put_checkpoint_blobs(self, command: Any) -> SimpleNamespace:
        for b in command.blobs:
            bkey = (
                command.scope,
                command.thread_id,
                command.checkpoint_ns,
                b.checkpoint_id,
                b.blob_key,
            )
            self.blobs[bkey] = b
        return SimpleNamespace(commit_sequence=self._next_commit(), outcome=SimpleNamespace())

    def put_checkpoint_writes(self, command: Any) -> SimpleNamespace:
        for item in command.writes:
            wkey = (
                command.scope,
                command.thread_id,
                command.checkpoint_ns,
                command.checkpoint_id,
                item.task_id,
                item.write_idx,
            )
            existing = self.writes.get(wkey)
            if existing is None or not existing.initialized or item.allow_overwrite:
                self.writes[wkey] = CheckpointWrite(
                    scope=command.scope,
                    thread_id=command.thread_id,
                    checkpoint_ns=command.checkpoint_ns,
                    checkpoint_id=command.checkpoint_id,
                    task_id=item.task_id,
                    write_idx=item.write_idx,
                    channel=item.channel,
                    serde_type=item.serde_type,
                    payload=item.payload,
                    task_path=item.task_path,
                    initialized=True,
                )
        return SimpleNamespace(commit_sequence=self._next_commit(), outcome=SimpleNamespace())

    def delete_checkpoints(self, command: Any) -> SimpleNamespace:
        for k in command.keys:
            self.checkpoints.pop(
                (command.scope, k.thread_id, k.checkpoint_ns, k.checkpoint_id), None
            )
        return SimpleNamespace(commit_sequence=self._next_commit(), outcome=SimpleNamespace())

    def delete_checkpoint_blobs(self, command: Any) -> SimpleNamespace:
        for k in command.keys:
            self.blobs.pop(
                (command.scope, k.thread_id, k.checkpoint_ns, k.checkpoint_id, k.blob_key),
                None,
            )
        return SimpleNamespace(commit_sequence=self._next_commit(), outcome=SimpleNamespace())

    def delete_checkpoint_writes(self, command: Any) -> SimpleNamespace:
        for k in command.keys:
            self.writes.pop(
                (
                    command.scope,
                    k.thread_id,
                    k.checkpoint_ns,
                    k.checkpoint_id,
                    k.task_id,
                    k.write_idx,
                ),
                None,
            )
        return SimpleNamespace(commit_sequence=self._next_commit(), outcome=SimpleNamespace())

    def _writes_for(
        self, scope: str, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> list[Any]:
        rows = [
            w
            for w in self.writes.values()
            if (w.scope, w.thread_id, w.checkpoint_ns, w.checkpoint_id)
            == (scope, thread_id, checkpoint_ns, checkpoint_id)
        ]
        rows.sort(key=lambda w: (w.task_id, w.write_idx))
        return rows

    def get_checkpoint_by_id(self, params: Any, options: Any = None) -> SimpleNamespace:
        _ = options
        cp = self.checkpoints.get(
            (params.scope, params.thread_id, params.checkpoint_ns, params.checkpoint_id)
        )
        writes = (
            self._writes_for(
                params.scope, params.thread_id, params.checkpoint_ns, params.checkpoint_id
            )
            if cp is not None
            else []
        )
        return SimpleNamespace(
            value=SimpleNamespace(checkpoint=cp, writes=tuple(writes)),
            next_cursor=None,
        )

    def get_latest_checkpoint(self, params: Any, options: Any = None) -> SimpleNamespace:
        _ = options
        matching = [
            cp
            for cp in self.checkpoints.values()
            if (cp.scope, cp.thread_id, cp.checkpoint_ns)
            == (params.scope, params.thread_id, params.checkpoint_ns)
        ]
        matching.sort(key=lambda c: c.checkpoint_id, reverse=True)
        if not matching:
            return SimpleNamespace(value=SimpleNamespace(checkpoints=()), next_cursor=None)
        top = matching[0]
        row = SimpleNamespace(
            thread_id=top.thread_id,
            checkpoint_ns=top.checkpoint_ns,
            checkpoint_id=top.checkpoint_id,
            parent_checkpoint_id=top.parent_checkpoint_id,
            checkpoint_serde_type=top.checkpoint_serde_type,
            checkpoint_payload=top.checkpoint_payload,
            metadata_serde_type=top.metadata_serde_type,
            metadata_payload=top.metadata_payload,
            writes=tuple(
                self._writes_for(top.scope, top.thread_id, top.checkpoint_ns, top.checkpoint_id)
            ),
        )
        return SimpleNamespace(value=SimpleNamespace(checkpoints=(row,)), next_cursor=None)

    def get_checkpoint_blobs(self, params: Any, options: Any = None) -> SimpleNamespace:
        _ = options
        key_set = set(params.blob_keys)
        found = [
            b
            for b in self.blobs.values()
            if (b.scope, b.thread_id, b.checkpoint_ns)
            == (params.scope, params.thread_id, params.checkpoint_ns)
            and b.blob_key in key_set
        ]
        found.sort(key=lambda b: b.blob_key)
        return SimpleNamespace(value=SimpleNamespace(blobs=tuple(found)), next_cursor=None)

    def get_parent_pending_sends(self, params: Any, options: Any = None) -> SimpleNamespace:
        _ = options
        rows = [
            w
            for w in self.writes.values()
            if (w.scope, w.thread_id, w.checkpoint_ns, w.checkpoint_id, w.channel)
            == (
                params.scope,
                params.thread_id,
                params.checkpoint_ns,
                params.checkpoint_id,
                params.channel,
            )
        ]
        rows.sort(key=lambda w: (w.task_id, w.write_idx))
        return SimpleNamespace(value=SimpleNamespace(writes=tuple(rows)), next_cursor=None)

    def _build_cp_rows(self, cps: list[Checkpoint]) -> tuple[Any, ...]:
        return tuple(
            SimpleNamespace(
                thread_id=c.thread_id,
                checkpoint_ns=c.checkpoint_ns,
                checkpoint_id=c.checkpoint_id,
                parent_checkpoint_id=c.parent_checkpoint_id,
                checkpoint_serde_type=c.checkpoint_serde_type,
                checkpoint_payload=c.checkpoint_payload,
                metadata_serde_type=c.metadata_serde_type,
                metadata_payload=c.metadata_payload,
                writes=tuple(
                    self._writes_for(c.scope, c.thread_id, c.checkpoint_ns, c.checkpoint_id)
                ),
            )
            for c in cps
        )

    def list_checkpoints_by_thread_ns(self, params: Any, options: Any = None) -> SimpleNamespace:
        _ = options
        cps = [
            c
            for c in self.checkpoints.values()
            if (c.scope, c.thread_id, c.checkpoint_ns)
            == (params.scope, params.thread_id, params.checkpoint_ns)
        ]
        cps.sort(key=lambda c: c.checkpoint_id, reverse=True)
        return SimpleNamespace(
            value=SimpleNamespace(checkpoints=self._build_cp_rows(cps)),
            next_cursor=None,
        )

    def list_checkpoints_by_thread_ns_before(
        self, params: Any, options: Any = None
    ) -> SimpleNamespace:
        _ = options
        cps = [
            c
            for c in self.checkpoints.values()
            if (c.scope, c.thread_id, c.checkpoint_ns)
            == (params.scope, params.thread_id, params.checkpoint_ns)
            and c.checkpoint_id < params.before_checkpoint_id
        ]
        cps.sort(key=lambda c: c.checkpoint_id, reverse=True)
        return SimpleNamespace(
            value=SimpleNamespace(checkpoints=self._build_cp_rows(cps)),
            next_cursor=None,
        )

    def list_checkpoints_by_thread(self, params: Any, options: Any = None) -> SimpleNamespace:
        _ = options
        cps = [
            c
            for c in self.checkpoints.values()
            if (c.scope, c.thread_id) == (params.scope, params.thread_id)
        ]
        cps.sort(key=lambda c: (c.checkpoint_id, c.checkpoint_ns), reverse=True)
        return SimpleNamespace(
            value=SimpleNamespace(checkpoints=self._build_cp_rows(cps)),
            next_cursor=None,
        )

    def list_checkpoints_by_thread_pages(
        self, params: Any, *, max_pages: int = 1024, options: Any = None
    ) -> list[SimpleNamespace]:
        _ = max_pages
        return [self.list_checkpoints_by_thread(params, options)]

    def list_checkpoints_by_thread_before(
        self, params: Any, options: Any = None
    ) -> SimpleNamespace:
        _ = options
        cps = [
            c
            for c in self.checkpoints.values()
            if (c.scope, c.thread_id) == (params.scope, params.thread_id)
            and c.checkpoint_id < params.before_checkpoint_id
        ]
        cps.sort(key=lambda c: (c.checkpoint_id, c.checkpoint_ns), reverse=True)
        return SimpleNamespace(
            value=SimpleNamespace(checkpoints=self._build_cp_rows(cps)),
            next_cursor=None,
        )

    def list_checkpoints_by_scope(self, params: Any, options: Any = None) -> SimpleNamespace:
        _ = options
        cps = [c for c in self.checkpoints.values() if c.scope == params.scope]
        cps.sort(key=lambda c: (c.checkpoint_id, c.thread_id, c.checkpoint_ns), reverse=True)
        return SimpleNamespace(
            value=SimpleNamespace(checkpoints=self._build_cp_rows(cps)),
            next_cursor=None,
        )

    def list_checkpoints_by_scope_pages(
        self, params: Any, *, max_pages: int = 1024, options: Any = None
    ) -> list[SimpleNamespace]:
        _ = max_pages
        return [self.list_checkpoints_by_scope(params, options)]

    def list_checkpoints_by_scope_before(self, params: Any, options: Any = None) -> SimpleNamespace:
        _ = options
        cps = [
            c
            for c in self.checkpoints.values()
            if c.scope == params.scope and c.checkpoint_id < params.before_checkpoint_id
        ]
        cps.sort(key=lambda c: (c.checkpoint_id, c.thread_id, c.checkpoint_ns), reverse=True)
        return SimpleNamespace(
            value=SimpleNamespace(checkpoints=self._build_cp_rows(cps)),
            next_cursor=None,
        )

    def list_thread_checkpoint_keys_pages(
        self, params: Any, *, max_pages: int = 1024, options: Any = None
    ) -> list[SimpleNamespace]:
        _ = (max_pages, options)
        cps = [
            c
            for c in self.checkpoints.values()
            if (c.scope, c.thread_id) == (params.scope, params.thread_id)
        ]
        return [SimpleNamespace(value=SimpleNamespace(checkpoints=tuple(cps)))]

    def list_thread_blob_keys_pages(
        self, params: Any, *, max_pages: int = 1024, options: Any = None
    ) -> list[SimpleNamespace]:
        _ = (max_pages, options)
        blobs = [
            b
            for b in self.blobs.values()
            if (b.scope, b.thread_id) == (params.scope, params.thread_id)
        ]
        return [SimpleNamespace(value=SimpleNamespace(blobs=tuple(blobs)))]

    def list_thread_write_keys_pages(
        self, params: Any, *, max_pages: int = 1024, options: Any = None
    ) -> list[SimpleNamespace]:
        _ = (max_pages, options)
        writes = [
            w
            for w in self.writes.values()
            if (w.scope, w.thread_id) == (params.scope, params.thread_id)
        ]
        return [SimpleNamespace(value=SimpleNamespace(writes=tuple(writes)))]


def _make_saver() -> tuple[RiffDBCheckpointSaver, FakeLangGraphClient]:
    saver = RiffDBCheckpointSaver("riffdb://langgraph-development")
    fake = FakeLangGraphClient()
    saver._sync_client = fake  # type: ignore[assignment]
    return saver, fake


def test_blob_key_and_batch_chunking() -> None:
    assert _make_blob_key("messages", "0001.123") == "8:messages:0001.123"
    blobs = [
        CheckpointBlob(
            scope="global",
            thread_id="t1",
            checkpoint_ns="",
            checkpoint_id="",
            blob_key=f"2:ch:{i}",
            channel="ch",
            blob_version=str(i),
            serde_type="json",
            payload=b"x" * 32_768,
        )
        for i in range(80)
    ]
    chunks = _chunk_blobs(blobs)
    assert len(chunks) >= 2
    assert sum(len(c) for c in chunks) == 80
    assert _chunk_writes([]) == []


def test_put_get_list_writes_and_lifecycle() -> None:
    saver, _ = _make_saver()
    tid = "thread-1"
    cfg: Any = {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}

    cp1 = empty_checkpoint()
    cp1["channel_values"] = {"a": "hello", "b": [1, 2, 3]}
    cp1["channel_versions"] = {"a": "1", "b": "1"}
    stored1 = saver.put(
        cfg, cp1, {"source": "input", "step": 0, "run_id": "run-1"}, {"a": "1", "b": "1"}
    )

    tup1 = saver.get_tuple(stored1)
    assert tup1 is not None
    assert tup1.checkpoint["channel_values"] == {"a": "hello", "b": [1, 2, 3]}
    assert tup1.metadata["step"] == 0

    # Idempotent regular writes vs overwritable special writes
    saver.put_writes(stored1, [("a", "first")], task_id="t1", task_path="0")
    saver.put_writes(stored1, [("a", "second")], task_id="t1", task_path="0")
    saver.put_writes(stored1, [(ERROR, "err-1")], task_id="t1", task_path="0")
    saver.put_writes(stored1, [(ERROR, "err-2"), (INTERRUPT, "int-1")], task_id="t1", task_path="0")

    tup1_writes = saver.get_tuple(stored1)
    assert tup1_writes is not None
    assert tup1_writes.pending_writes == [
        ("t1", INTERRUPT, "int-1"),
        ("t1", ERROR, "err-2"),
        ("t1", "a", "first"),
    ]

    # Second checkpoint on same thread
    cp2 = empty_checkpoint()
    cp2["channel_values"] = {"a": "world", "b": [1, 2, 3]}
    cp2["channel_versions"] = {"a": "2", "b": "1"}
    stored2 = saver.put(stored1, cp2, {"source": "loop", "step": 1, "run_id": "run-2"}, {"a": "2"})

    latest = saver.get_tuple(cfg)
    assert latest is not None
    assert latest.checkpoint["id"] == cp2["id"]
    assert latest.checkpoint["channel_values"] == {"a": "world", "b": [1, 2, 3]}
    assert latest.parent_config == stored1

    # List filters
    loop_only = list(saver.list(cfg, filter={"source": "loop"}))
    assert len(loop_only) == 1
    assert loop_only[0].checkpoint["id"] == cp2["id"]

    before_cp2 = list(saver.list(cfg, before=stored2))
    assert len(before_cp2) == 1
    assert before_cp2[0].checkpoint["id"] == cp1["id"]

    # Copy thread
    saver.copy_thread(tid, "thread-copy")
    copied = saver.get_tuple({"configurable": {"thread_id": "thread-copy"}})
    assert copied is not None
    assert copied.checkpoint["channel_values"] == {"a": "world", "b": [1, 2, 3]}

    # Prune keep_latest
    saver.prune(["thread-copy"], strategy="keep_latest")
    remaining = list(saver.list({"configurable": {"thread_id": "thread-copy"}}))
    assert len(remaining) == 1
    assert remaining[0].checkpoint["id"] == cp2["id"]

    # Invalid prune strategy
    with pytest.raises(ValueError, match="unsupported prune strategy"):
        saver.prune(["thread-copy"], strategy="invalid")

    # Delete for runs
    saver.delete_for_runs(["run-1"])
    after_run_del = list(saver.list(cfg))
    assert [t.checkpoint["id"] for t in after_run_del] == [cp2["id"]]

    # Delete thread
    saver.delete_thread("thread-copy")
    assert saver.get_tuple({"configurable": {"thread_id": "thread-copy"}}) is None


def test_legacy_pending_sends_migration_and_stategraph() -> None:
    saver, _ = _make_saver()

    # Legacy v<4 pending_sends migration
    parent_cp = empty_checkpoint()
    parent_cp["v"] = 3
    p_cfg = saver.put(
        {"configurable": {"thread_id": "legacy", "checkpoint_ns": ""}},
        parent_cp,
        {"source": "input", "step": 0},
        {},
    )
    saver.put_writes(p_cfg, [(TASKS, "send-payload-1")], task_id="task-1")
    child_cp = empty_checkpoint()
    child_cp["v"] = 3
    c_cfg = saver.put(p_cfg, child_cp, {"source": "loop", "step": 1}, {})
    tup = saver.get_tuple(c_cfg)
    assert tup is not None
    assert tup.checkpoint["channel_values"][TASKS] == ["send-payload-1"]

    # StateGraph end-to-end execution
    class State(TypedDict):
        log: Annotated[list[str], operator.add]

    builder: StateGraph[Any, Any, Any, Any] = StateGraph(State)
    builder.add_node("step_a", lambda s: {"log": [f"a:{len(s['log'])}"]})
    builder.add_node("step_b", lambda s: {"log": [f"b:{len(s['log'])}"]})
    builder.add_edge(START, "step_a")
    builder.add_edge("step_a", "step_b")
    builder.add_edge("step_b", END)
    graph = builder.compile(checkpointer=saver)

    res = graph.invoke({"log": ["init"]}, config={"configurable": {"thread_id": "graph-thread"}})
    assert res["log"] == ["init", "a:1", "b:2"]
