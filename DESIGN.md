# Design

## Compatibility target

`riffdb-langgraph` implements `langgraph.checkpoint.base.BaseCheckpointSaver[str]`
for `langgraph>=1.1.0` and `langgraph-checkpoint>=4.0.0`, validated against all
eight capabilities of `langgraph-checkpoint-conformance`:

| Capability | Methods | Behavior |
|---|---|---|
| `put` | `put`, `aput` | Atomic checkpoint envelope + channel blob persistence, metadata normalization via `get_checkpoint_metadata` |
| `put_writes` | `put_writes`, `aput_writes` | Pending task writes with idempotent regular indices (`write_idx >= 0`) and overwritable special indices (`WRITES_IDX_MAP`) |
| `get_tuple` | `get_tuple`, `aget_tuple` | Latest or specific `checkpoint_id` lookup, channel blob hydration, `pending_writes` ordering via `writes_sort_key`, and `v < 4` `TASKS` migration |
| `list` | `list`, `alist` | Reverse-chronological (`checkpoint_id DESC`) checkpoint listing with `thread_id`, `checkpoint_ns`, `before`, `limit`, and metadata filtering |
| `delete_thread` | `delete_thread`, `adelete_thread` | Removal of all checkpoints, channel blobs, and pending writes across all namespaces for a `thread_id` |
| `delete_for_runs` | `delete_for_runs`, `adelete_for_runs` | Removal of checkpoints (and their pending writes) whose metadata `run_id` matches the target run IDs |
| `copy_thread` | `copy_thread`, `acopy_thread` | Cloning all checkpoints, channel blobs, and pending writes across all namespaces from a source `thread_id` to a target `thread_id` |
| `prune` | `prune`, `aprune` | Pruning checkpoints per `(thread_id, checkpoint_ns)` using `keep_latest` or `delete` strategies |

The Python floor is 3.13, matching the RiffDB Python application runtime.

## Contract and aggregate model

The contract (`riffdb/contract.riff`) defines a single aggregate `Checkpoints`
partitioned by `scope` with `conflict_key (scope, thread_id)`:

- **`Checkpoint`** (root): keyed by `(scope, thread_id, checkpoint_ns, checkpoint_id)`.
  Stores `parent_checkpoint_id`, typed serde payloads for the stripped checkpoint
  envelope (`channel_values` removed) and normalized `CheckpointMetadata`, and a
  monotonic `revision`. Indexed with `binary_utf8_v1` text keys for reverse
  lexicographic `checkpoint_id` range queries (`< $before_checkpoint_id`).
- **`CheckpointBlob`** (child): keyed by
  `(scope, thread_id, checkpoint_ns, checkpoint_id, blob_key)` with
  `checkpoint_id = ""` and `blob_key = "{len(channel)}:{channel}:{version}"`.
  Deduplicates serialized channel values across checkpoints within a
  `(thread_id, checkpoint_ns)`. Empty channel versions are recorded with
  `serde_type = "empty"` so removed/unwritten channels are omitted on hydration.
- **`CheckpointWrite`** (child): keyed by
  `(scope, thread_id, checkpoint_ns, checkpoint_id, task_id, write_idx)`.
  Stores pending task writes with `initialized: bool` so `PutCheckpointWrites`
  enforces write-once semantics when `write_idx >= 0` (`allow_overwrite == false`)
  and overwrites when `write_idx < 0` (`allow_overwrite == true`).

## Configuration

`RiffDBCheckpointSaver` accepts a canonical URI `riffdb://<database>`. It
contains no credential or TLS material. Connection authority comes from
protected process configuration:

- `RIFFDB_LANGGRAPH_ENDPOINT` (or fallback `RIFFDB_ENDPOINT`)
- `RIFFDB_LANGGRAPH_CREDENTIAL_FILE` (or fallback `RIFFDB_CREDENTIAL_FILE`)
- `RIFFDB_LANGGRAPH_TRUST_ROOT` (optional, for verified TLS)
- `RIFFDB_LANGGRAPH_SERVER_NAME` (optional, for verified TLS)

The credential file is loaded via `BearerCredential.from_protected_file`.
Unknown URI schemes, userinfo, ports, paths, query parameters, and fragments
raise `InvalidCheckpointerURI` before any network activity.

## Session consistency

`RiffDBCheckpointSaver` tracks the highest `commit_sequence` returned by any
mutation command and attaches `read_after_commit` to subsequent queries on the
same saver instance, guaranteeing monotonic read-your-writes consistency across
both sync and async execution paths.
