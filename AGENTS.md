# AGENTS.md — RiffDB LangGraph adapter

## Mission

Implement a standalone Python checkpoint-saver adapter for LangGraph using a
compiled RiffDB application. All LangGraph-specific behavior belongs in this
repository, never in RiffDB.

## Authoritative inputs

- `langgraph` (`>=1.1.0`), `langgraph-checkpoint` (`>=4.0.0`), and
  `langgraph-checkpoint-conformance` (`>=0.1.0`), especially
  `langgraph.checkpoint.base.BaseCheckpointSaver` and the upstream conformance
  suite (`langgraph.checkpoint.conformance.validate`).
- This repository's `DESIGN.md`.
- The accepted RiffDB specification and ADRs at the revision recorded in the
  generated application lock and `.github/riffdb-revision`.

If these inputs conflict, stop and request maintainer review. Do not weaken a
LangGraph checkpointer semantic or a RiffDB safety guarantee to make a test
pass.

## Architecture boundaries

1. LangGraph checkpoint mutations use generated, compiled RiffDB commands only.
2. Reads use generated named RiffQL queries only. Do not add generic CRUD,
   dynamically assembled RiffQL, or unbounded scans.
3. Checkpoint metadata, channel version blobs, and pending task writes belong to
   a single `Checkpoints` aggregate partitioned by `scope` with `conflict_key
   (scope, thread_id)`.
4. `put_writes` / `aput_writes` preserve upstream idempotency rules:
   non-negative write indices (`write_idx >= 0`) are write-once per
   `(thread_id, checkpoint_ns, checkpoint_id, task_id, write_idx)`, while
   special negative indices from `WRITES_IDX_MAP` (`ERROR`, `SCHEDULED`,
   `INTERRUPT`, `RESUME`) overwrite in place.
5. `get_tuple` and `list` hydrate channel values from `CheckpointBlob` rows
   referenced by `channel_versions`, order `pending_writes` via upstream
   `writes_sort_key`, and migrate `v < 4` parent `TASKS` writes into
   `channel_values[TASKS]`.
6. Connection URIs (`riffdb://<database>`) carry only the database alias.
   Credentials and TLS configuration come from protected environment variables,
   never from the URI.

## Completion checks

Before calling the adapter usable:

```bash
make check
make conformance
```

Unit fakes may supplement upstream tests but never replace real RiffDB loopback
and `langgraph-checkpoint-conformance` coverage.
