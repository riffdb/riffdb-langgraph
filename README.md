# riffdb-langgraph

`riffdb-langgraph` is a compiled RiffDB checkpoint-saver adapter
(`RiffDBCheckpointSaver`) for LangGraph (`langgraph>=1.1.0`,
`langgraph-checkpoint>=4.0.0`). It is an out-of-tree integration: no
LangGraph-specific code or vocabulary belongs in the RiffDB repository.

The adapter implements both synchronous and asynchronous `BaseCheckpointSaver`
surfaces and passes all eight capability suites (81 tests) in the upstream
`langgraph-checkpoint-conformance` package:

- `put` (`put`, `aput`)
- `put_writes` (`put_writes`, `aput_writes`)
- `get_tuple` (`get_tuple`, `aget_tuple`)
- `list` (`list`, `alist`)
- `delete_thread` (`delete_thread`, `adelete_thread`)
- `delete_for_runs` (`delete_for_runs`, `adelete_for_runs`)
- `copy_thread` (`copy_thread`, `acopy_thread`)
- `prune` (`prune`, `aprune`)

## Usage

```python
from langgraph.graph import START, END, StateGraph
from riffdb_langgraph import RiffDBCheckpointSaver

with RiffDBCheckpointSaver("riffdb://langgraph-development") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    result = graph.invoke(
        {"messages": ["hello"]},
        config={"configurable": {"thread_id": "thread-1"}},
    )
```

RiffDB TLS and credential material is supplied through protected environment
configuration (`RIFFDB_LANGGRAPH_ENDPOINT` and
`RIFFDB_LANGGRAPH_CREDENTIAL_FILE`) described in [DESIGN.md](DESIGN.md); it is
never embedded in the checkpointer URI.

## Development and Verification

```bash
make check
make conformance
```

`make conformance` starts a disposable `riffdb dev` instance, runs sync and
async `StateGraph` fan-out smoke tests, and executes the full
`langgraph-checkpoint-conformance` validation suite against the live server.

## RiffDB compatibility CI

This adapter owns its RiffDB compatibility checks. Pull requests compile the
application sources and verify existing locks and generated artifacts against
the full commit in `.github/riffdb-revision`. Scheduled checks use RiffDB
`main`; manual dispatch accepts a full commit or `main`. Failures are reported
here and do not block RiffDB itself.

To upgrade, update the pin, migrate sources as needed, review the result of
`riffdb application lock --write`, and verify with `riffdb application check`.
Commit the pin and migration together in this repository.
