"""LangGraph checkpoint-saver implementation backed by a compiled RiffDB profile."""

from __future__ import annotations

import os
import random
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from types import TracebackType
from typing import Any, Protocol, cast
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    DeltaChannelHistory,
    PendingWrite,
    get_checkpoint_id,
    get_checkpoint_metadata,
    writes_sort_key,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.types import TASKS
from riffdb_application import (
    AsyncApplicationTransport,
    AttemptBudget,
    BearerCredential,
    CallMetadata,
    DatabaseAlias,
    QueryOptions,
    SyncApplicationTransport,
    VerifiedTlsConfig,
)
from riffdb_application._binding import canonical_value_encoded_length, encode_value

from riffdb_langgraph.config import CheckpointerConfig
from riffdb_langgraph.generated.client import (
    AsyncRiffDBLangGraphClient,
    CheckpointBlob,
    CheckpointBlobKeyInput,
    CheckpointKeyInput,
    CheckpointWriteInput,
    CheckpointWriteKeyInput,
    DeleteCheckpointBlobsInput,
    DeleteCheckpointsInput,
    DeleteCheckpointWritesInput,
    GetCheckpointBlobsParams,
    GetCheckpointByIdParams,
    GetLatestCheckpointParams,
    GetParentPendingSendsParams,
    ListCheckpointsByScopeBeforeParams,
    ListCheckpointsByScopeParams,
    ListCheckpointsByThreadBeforeParams,
    ListCheckpointsByThreadNsBeforeParams,
    ListCheckpointsByThreadNsParams,
    ListCheckpointsByThreadParams,
    ListThreadBlobKeysParams,
    ListThreadCheckpointKeysParams,
    ListThreadWriteKeysParams,
    PutCheckpointBlobsInput,
    PutCheckpointInput,
    PutCheckpointWithBlobsInput,
    PutCheckpointWritesInput,
    RiffDBLangGraphClient,
)

_DEFAULT_SCOPE = "global"
_EMPTY_SERDE_TYPE = "empty"
_BLOB_BATCH_MAX_COUNT = 128
_BLOB_BATCH_MAX_BYTES = 2_097_152
_WRITE_BATCH_MAX_COUNT = 256
_WRITE_BATCH_MAX_BYTES = 2_097_152
_DELETE_BATCH_MAX_COUNT = 256
_GET_BLOBS_CHUNK_SIZE = 12
_LIST_PAGE_SIZE = 4
_KEY_LIST_PAGE_SIZE = 256
_MAX_PAGES = 1024


class _CheckpointRowProtocol(Protocol):
    @property
    def thread_id(self) -> str: ...
    @property
    def checkpoint_ns(self) -> str: ...
    @property
    def checkpoint_id(self) -> str: ...
    @property
    def parent_checkpoint_id(self) -> str | None: ...
    @property
    def checkpoint_serde_type(self) -> str: ...
    @property
    def checkpoint_payload(self) -> bytes: ...
    @property
    def metadata_serde_type(self) -> str: ...
    @property
    def metadata_payload(self) -> bytes: ...


class _WriteRowProtocol(Protocol):
    @property
    def task_id(self) -> str: ...
    @property
    def write_idx(self) -> int: ...
    @property
    def channel(self) -> str: ...
    @property
    def serde_type(self) -> str: ...
    @property
    def payload(self) -> bytes: ...
    @property
    def task_path(self) -> str: ...


def _make_blob_key(channel: str, version: str | int | float) -> str:
    ver_str = str(version)
    return f"{len(channel)}:{channel}:{ver_str}"


def _chunk_blobs(blobs: Sequence[CheckpointBlob]) -> list[tuple[CheckpointBlob, ...]]:
    chunks: list[tuple[CheckpointBlob, ...]] = []
    current: list[CheckpointBlob] = []
    current_bytes = 0
    for item in blobs:
        item_bytes = canonical_value_encoded_length(encode_value(item, CheckpointBlob))
        if current and (
            len(current) >= _BLOB_BATCH_MAX_COUNT
            or current_bytes + item_bytes > _BLOB_BATCH_MAX_BYTES
        ):
            chunks.append(tuple(current))
            current = [item]
            current_bytes = item_bytes
        else:
            current.append(item)
            current_bytes += item_bytes
    if current:
        chunks.append(tuple(current))
    return chunks


def _chunk_writes(writes: Sequence[CheckpointWriteInput]) -> list[tuple[CheckpointWriteInput, ...]]:
    chunks: list[tuple[CheckpointWriteInput, ...]] = []
    current: list[CheckpointWriteInput] = []
    current_bytes = 0
    for item in writes:
        item_bytes = canonical_value_encoded_length(encode_value(item, CheckpointWriteInput))
        if current and (
            len(current) >= _WRITE_BATCH_MAX_COUNT
            or current_bytes + item_bytes > _WRITE_BATCH_MAX_BYTES
        ):
            chunks.append(tuple(current))
            current = [item]
            current_bytes = item_bytes
        else:
            current.append(item)
            current_bytes += item_bytes
    if current:
        chunks.append(tuple(current))
    return chunks


class RiffDBCheckpointSaver(
    BaseCheckpointSaver[str],
    AbstractContextManager["RiffDBCheckpointSaver"],
    AbstractAsyncContextManager["RiffDBCheckpointSaver"],
):
    """Compiled RiffDB checkpoint saver for LangGraph."""

    def __init__(
        self,
        uri: str = "riffdb://langgraph-development",
        *,
        serde: SerializerProtocol | None = None,
        scope: str = _DEFAULT_SCOPE,
        sync_transport: SyncApplicationTransport | None = None,
        async_transport: AsyncApplicationTransport | None = None,
    ) -> None:
        super().__init__(serde=serde)
        self._config = CheckpointerConfig.from_uri(uri)
        if not 1 <= len(scope.encode("utf-8")) <= 32:
            raise ValueError("scope must be 1..32 UTF-8 bytes")
        self._scope = scope
        self._sync_transport = sync_transport
        self._async_transport = async_transport
        self._owns_sync_transport = sync_transport is None
        self._owns_async_transport = async_transport is None
        self._sync_client: RiffDBLangGraphClient | None = (
            RiffDBLangGraphClient(sync_transport, AttemptBudget(3))
            if sync_transport is not None
            else None
        )
        self._async_client: AsyncRiffDBLangGraphClient | None = (
            AsyncRiffDBLangGraphClient(async_transport, AttemptBudget(3))
            if async_transport is not None
            else None
        )
        self._last_commit: int | None = None

    @property
    def database(self) -> str:
        return self._config.database

    @property
    def scope(self) -> str:
        return self._scope

    def _record_commit(self, commit_seq: int | None) -> None:
        if commit_seq is not None and commit_seq > 0:
            if self._last_commit is None or commit_seq > self._last_commit:
                self._last_commit = commit_seq

    def _query_options(self, cursor: str | None = None) -> QueryOptions:
        return QueryOptions(cursor=cursor, read_after_commit=self._last_commit)

    def _resolve_env_connection(self) -> tuple[str, CallMetadata, VerifiedTlsConfig | None]:
        endpoint = os.environ.get("RIFFDB_LANGGRAPH_ENDPOINT") or os.environ.get("RIFFDB_ENDPOINT")
        cred_file = os.environ.get("RIFFDB_LANGGRAPH_CREDENTIAL_FILE") or os.environ.get(
            "RIFFDB_CREDENTIAL_FILE"
        )
        if not endpoint or not cred_file:
            raise RuntimeError(
                "Missing RiffDB LangGraph connection configuration: set "
                "RIFFDB_LANGGRAPH_ENDPOINT and RIFFDB_LANGGRAPH_CREDENTIAL_FILE"
            )
        if os.environ.get("RIFFDB_TENANT") == "":
            os.environ.pop("RIFFDB_TENANT", None)
        credential = BearerCredential.from_protected_file(cred_file)
        metadata = CallMetadata.authenticated(credential).with_database(
            DatabaseAlias(self.database)
        )
        trust_root = os.environ.get("RIFFDB_LANGGRAPH_TRUST_ROOT")
        server_name = os.environ.get("RIFFDB_LANGGRAPH_SERVER_NAME")
        if trust_root and server_name:
            return (
                endpoint,
                metadata,
                VerifiedTlsConfig(
                    endpoint=endpoint,
                    trust_root=trust_root,
                    server_name=server_name,
                ),
            )
        return endpoint, metadata, None

    def _sync_app(self) -> RiffDBLangGraphClient:
        if self._sync_client is None:
            endpoint, metadata, tls = self._resolve_env_connection()
            if tls is not None:
                self._sync_transport = SyncApplicationTransport.connect_verified_tls(tls, metadata)
            else:
                self._sync_transport = SyncApplicationTransport.connect_uri(endpoint, metadata)
            self._owns_sync_transport = True
            self._sync_client = RiffDBLangGraphClient(self._sync_transport, AttemptBudget(3))
        if self._last_commit is not None:
            self._sync_client = self._sync_client.with_read_after_commit(self._last_commit)
        return self._sync_client

    async def _async_app(self) -> AsyncRiffDBLangGraphClient:
        if self._async_client is None:
            endpoint, metadata, tls = self._resolve_env_connection()
            if tls is not None:
                self._async_transport = await AsyncApplicationTransport.connect_verified_tls(
                    tls, metadata
                )
            else:
                self._async_transport = await AsyncApplicationTransport.connect_uri(
                    endpoint, metadata
                )
            self._owns_async_transport = True
            self._async_client = AsyncRiffDBLangGraphClient(self._async_transport, AttemptBudget(3))
        if self._last_commit is not None:
            self._async_client = self._async_client.with_read_after_commit(self._last_commit)
        return self._async_client

    def close(self) -> None:
        if self._owns_sync_transport and self._sync_transport is not None:
            self._sync_transport.close()
            self._sync_transport = None
            self._sync_client = None

    async def aclose(self) -> None:
        if self._owns_async_transport and self._async_transport is not None:
            await self._async_transport.close()
            self._async_transport = None
            self._async_client = None
        self.close()

    def __enter__(self) -> RiffDBCheckpointSaver:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self.close()
        return None

    async def __aenter__(self) -> RiffDBCheckpointSaver:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        await self.aclose()
        return None

    def get_next_version(self, current: str | None, channel: None) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(str(current).split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"

    def _load_blobs_sync(
        self,
        client: RiffDBLangGraphClient,
        thread_id: str,
        checkpoint_ns: str,
        versions: ChannelVersions,
    ) -> dict[str, Any]:
        if not versions:
            return {}
        key_to_channel: dict[str, str] = {}
        for ch, ver in versions.items():
            key_to_channel[_make_blob_key(ch, ver)] = ch
        all_keys = sorted(key_to_channel.keys())
        result: dict[str, Any] = {}
        for i in range(0, len(all_keys), _GET_BLOBS_CHUNK_SIZE):
            chunk = tuple(all_keys[i : i + _GET_BLOBS_CHUNK_SIZE])
            res = client.get_checkpoint_blobs(
                GetCheckpointBlobsParams(
                    scope=self._scope,
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    blob_keys=chunk,
                ),
                self._query_options(),
            )
            for b in res.value.blobs:
                if b.serde_type == _EMPTY_SERDE_TYPE:
                    continue
                ch_name = key_to_channel.get(b.blob_key, b.channel)
                result[ch_name] = self.serde.loads_typed((b.serde_type, b.payload))
        return result

    async def _load_blobs_async(
        self,
        client: AsyncRiffDBLangGraphClient,
        thread_id: str,
        checkpoint_ns: str,
        versions: ChannelVersions,
    ) -> dict[str, Any]:
        if not versions:
            return {}
        key_to_channel: dict[str, str] = {}
        for ch, ver in versions.items():
            key_to_channel[_make_blob_key(ch, ver)] = ch
        all_keys = sorted(key_to_channel.keys())
        result: dict[str, Any] = {}
        for i in range(0, len(all_keys), _GET_BLOBS_CHUNK_SIZE):
            chunk = tuple(all_keys[i : i + _GET_BLOBS_CHUNK_SIZE])
            res = await client.get_checkpoint_blobs(
                GetCheckpointBlobsParams(
                    scope=self._scope,
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    blob_keys=chunk,
                ),
                self._query_options(),
            )
            for b in res.value.blobs:
                if b.serde_type == _EMPTY_SERDE_TYPE:
                    continue
                ch_name = key_to_channel.get(b.blob_key, b.channel)
                result[ch_name] = self.serde.loads_typed((b.serde_type, b.payload))
        return result

    def _decode_writes(self, writes: Sequence[_WriteRowProtocol]) -> list[PendingWrite]:
        sorted_rows = sorted(
            writes,
            key=lambda w: writes_sort_key(w.task_path, w.task_id, w.write_idx),
        )
        return [
            (w.task_id, w.channel, self.serde.loads_typed((w.serde_type, w.payload)))
            for w in sorted_rows
        ]

    def _migrate_pending_sends_sync(
        self,
        client: RiffDBLangGraphClient,
        checkpoint_dict: dict[str, Any],
        thread_id: str,
        checkpoint_ns: str,
        parent_checkpoint_id: str | None,
    ) -> None:
        if checkpoint_dict.get("v", 1) < 4 and parent_checkpoint_id is not None:
            res = client.get_parent_pending_sends(
                GetParentPendingSendsParams(
                    scope=self._scope,
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=parent_checkpoint_id,
                    channel=TASKS,
                ),
                self._query_options(),
            )
            if res.value.writes:
                sorted_sends = sorted(
                    res.value.writes,
                    key=lambda w: writes_sort_key(w.task_path, w.task_id, w.write_idx),
                )
                sends = [self.serde.loads_typed((w.serde_type, w.payload)) for w in sorted_sends]
                channel_values = checkpoint_dict.setdefault("channel_values", {})
                channel_values[TASKS] = sends
                channel_versions = checkpoint_dict.setdefault("channel_versions", {})
                if TASKS not in channel_versions:
                    channel_versions[TASKS] = (
                        max(channel_versions.values())
                        if channel_versions
                        else self.get_next_version(None, None)
                    )

    async def _migrate_pending_sends_async(
        self,
        client: AsyncRiffDBLangGraphClient,
        checkpoint_dict: dict[str, Any],
        thread_id: str,
        checkpoint_ns: str,
        parent_checkpoint_id: str | None,
    ) -> None:
        if checkpoint_dict.get("v", 1) < 4 and parent_checkpoint_id is not None:
            res = await client.get_parent_pending_sends(
                GetParentPendingSendsParams(
                    scope=self._scope,
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=parent_checkpoint_id,
                    channel=TASKS,
                ),
                self._query_options(),
            )
            if res.value.writes:
                sorted_sends = sorted(
                    res.value.writes,
                    key=lambda w: writes_sort_key(w.task_path, w.task_id, w.write_idx),
                )
                sends = [self.serde.loads_typed((w.serde_type, w.payload)) for w in sorted_sends]
                channel_values = checkpoint_dict.setdefault("channel_values", {})
                channel_values[TASKS] = sends
                channel_versions = checkpoint_dict.setdefault("channel_versions", {})
                if TASKS not in channel_versions:
                    channel_versions[TASKS] = (
                        max(channel_versions.values())
                        if channel_versions
                        else self.get_next_version(None, None)
                    )

    def _hydrate_tuple_sync(
        self,
        client: RiffDBLangGraphClient,
        cp_row: _CheckpointRowProtocol,
        writes: Sequence[_WriteRowProtocol],
        *,
        requested_config: RunnableConfig | None = None,
        preloaded_metadata: CheckpointMetadata | None = None,
    ) -> CheckpointTuple:
        checkpoint_raw: dict[str, Any] = self.serde.loads_typed(
            (cp_row.checkpoint_serde_type, cp_row.checkpoint_payload)
        )
        channel_versions: ChannelVersions = checkpoint_raw.get("channel_versions", {})
        channel_values = self._load_blobs_sync(
            client, cp_row.thread_id, cp_row.checkpoint_ns, channel_versions
        )
        full_checkpoint: dict[str, Any] = {
            **checkpoint_raw,
            "channel_values": channel_values,
        }
        self._migrate_pending_sends_sync(
            client,
            full_checkpoint,
            cp_row.thread_id,
            cp_row.checkpoint_ns,
            cp_row.parent_checkpoint_id,
        )
        metadata: CheckpointMetadata = (
            preloaded_metadata
            if preloaded_metadata is not None
            else self.serde.loads_typed((cp_row.metadata_serde_type, cp_row.metadata_payload))
        )
        tuple_config: RunnableConfig = (
            requested_config
            if requested_config is not None
            else {
                "configurable": {
                    "thread_id": cp_row.thread_id,
                    "checkpoint_ns": cp_row.checkpoint_ns,
                    "checkpoint_id": cp_row.checkpoint_id,
                }
            }
        )
        parent_config: RunnableConfig | None = (
            {
                "configurable": {
                    "thread_id": cp_row.thread_id,
                    "checkpoint_ns": cp_row.checkpoint_ns,
                    "checkpoint_id": cp_row.parent_checkpoint_id,
                }
            }
            if cp_row.parent_checkpoint_id
            else None
        )
        return CheckpointTuple(
            config=tuple_config,
            checkpoint=cast(Checkpoint, full_checkpoint),
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=self._decode_writes(writes),
        )

    async def _hydrate_tuple_async(
        self,
        client: AsyncRiffDBLangGraphClient,
        cp_row: _CheckpointRowProtocol,
        writes: Sequence[_WriteRowProtocol],
        *,
        requested_config: RunnableConfig | None = None,
        preloaded_metadata: CheckpointMetadata | None = None,
    ) -> CheckpointTuple:
        checkpoint_raw: dict[str, Any] = self.serde.loads_typed(
            (cp_row.checkpoint_serde_type, cp_row.checkpoint_payload)
        )
        channel_versions: ChannelVersions = checkpoint_raw.get("channel_versions", {})
        channel_values = await self._load_blobs_async(
            client, cp_row.thread_id, cp_row.checkpoint_ns, channel_versions
        )
        full_checkpoint: dict[str, Any] = {
            **checkpoint_raw,
            "channel_values": channel_values,
        }
        await self._migrate_pending_sends_async(
            client,
            full_checkpoint,
            cp_row.thread_id,
            cp_row.checkpoint_ns,
            cp_row.parent_checkpoint_id,
        )
        metadata: CheckpointMetadata = (
            preloaded_metadata
            if preloaded_metadata is not None
            else self.serde.loads_typed((cp_row.metadata_serde_type, cp_row.metadata_payload))
        )
        tuple_config: RunnableConfig = (
            requested_config
            if requested_config is not None
            else {
                "configurable": {
                    "thread_id": cp_row.thread_id,
                    "checkpoint_ns": cp_row.checkpoint_ns,
                    "checkpoint_id": cp_row.checkpoint_id,
                }
            }
        )
        parent_config: RunnableConfig | None = (
            {
                "configurable": {
                    "thread_id": cp_row.thread_id,
                    "checkpoint_ns": cp_row.checkpoint_ns,
                    "checkpoint_id": cp_row.parent_checkpoint_id,
                }
            }
            if cp_row.parent_checkpoint_id
            else None
        )
        return CheckpointTuple(
            config=tuple_config,
            checkpoint=cast(Checkpoint, full_checkpoint),
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=self._decode_writes(writes),
        )

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        configurable = config.get("configurable", {})
        thread_id = str(configurable["thread_id"])
        checkpoint_ns = str(configurable.get("checkpoint_ns", ""))
        client = self._sync_app()
        checkpoint_id = get_checkpoint_id(config)
        if checkpoint_id:
            res = client.get_checkpoint_by_id(
                GetCheckpointByIdParams(
                    scope=self._scope,
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint_id,
                ),
                self._query_options(),
            )
            if res.value.checkpoint is None:
                return None
            return self._hydrate_tuple_sync(
                client,
                res.value.checkpoint,
                res.value.writes,
                requested_config=config,
            )
        latest = client.get_latest_checkpoint(
            GetLatestCheckpointParams(
                scope=self._scope,
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
            ),
            self._query_options(),
        )
        if not latest.value.checkpoints:
            return None
        cp_row = latest.value.checkpoints[0]
        return self._hydrate_tuple_sync(client, cp_row, cp_row.writes)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        configurable = config.get("configurable", {})
        thread_id = str(configurable["thread_id"])
        checkpoint_ns = str(configurable.get("checkpoint_ns", ""))
        client = await self._async_app()
        checkpoint_id = get_checkpoint_id(config)
        if checkpoint_id:
            res = await client.get_checkpoint_by_id(
                GetCheckpointByIdParams(
                    scope=self._scope,
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint_id,
                ),
                self._query_options(),
            )
            if res.value.checkpoint is None:
                return None
            return await self._hydrate_tuple_async(
                client,
                res.value.checkpoint,
                res.value.writes,
                requested_config=config,
            )
        latest = await client.get_latest_checkpoint(
            GetLatestCheckpointParams(
                scope=self._scope,
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
            ),
            self._query_options(),
        )
        if not latest.value.checkpoints:
            return None
        cp_row = latest.value.checkpoints[0]
        return await self._hydrate_tuple_async(client, cp_row, cp_row.writes)

    @staticmethod
    def _matches_filter(metadata: CheckpointMetadata, filter_dict: dict[str, Any] | None) -> bool:
        if not filter_dict:
            return True
        return all(metadata.get(k) == v for k, v in filter_dict.items())

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        if limit is not None and limit <= 0:
            return
        client = self._sync_app()
        configurable = config.get("configurable", {}) if config else {}
        thread_id: str | None = (
            str(configurable["thread_id"]) if "thread_id" in configurable else None
        )
        checkpoint_ns: str | None = (
            str(configurable["checkpoint_ns"])
            if "checkpoint_ns" in configurable and configurable["checkpoint_ns"] is not None
            else None
        )
        config_checkpoint_id = get_checkpoint_id(config) if config else None
        before_checkpoint_id = get_checkpoint_id(before) if before else None

        remaining = limit
        if thread_id is not None and checkpoint_ns is not None and config_checkpoint_id is not None:
            if before_checkpoint_id is not None and config_checkpoint_id >= before_checkpoint_id:
                return
            res = client.get_checkpoint_by_id(
                GetCheckpointByIdParams(
                    scope=self._scope,
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=config_checkpoint_id,
                ),
                self._query_options(),
            )
            if res.value.checkpoint is None:
                return
            md: CheckpointMetadata = self.serde.loads_typed(
                (res.value.checkpoint.metadata_serde_type, res.value.checkpoint.metadata_payload)
            )
            if self._matches_filter(md, filter):
                yield self._hydrate_tuple_sync(
                    client,
                    res.value.checkpoint,
                    res.value.writes,
                    preloaded_metadata=md,
                )
            return

        cursor: str | None = None
        for _ in range(_MAX_PAGES):
            if thread_id is not None and checkpoint_ns is not None:
                if before_checkpoint_id is not None:
                    page: Any = client.list_checkpoints_by_thread_ns_before(
                        ListCheckpointsByThreadNsBeforeParams(
                            scope=self._scope,
                            thread_id=thread_id,
                            checkpoint_ns=checkpoint_ns,
                            before_checkpoint_id=before_checkpoint_id,
                            limit=_LIST_PAGE_SIZE,
                        ),
                        self._query_options(cursor),
                    )
                else:
                    page = client.list_checkpoints_by_thread_ns(
                        ListCheckpointsByThreadNsParams(
                            scope=self._scope,
                            thread_id=thread_id,
                            checkpoint_ns=checkpoint_ns,
                            limit=_LIST_PAGE_SIZE,
                        ),
                        self._query_options(cursor),
                    )
            elif thread_id is not None:
                if before_checkpoint_id is not None:
                    page = client.list_checkpoints_by_thread_before(
                        ListCheckpointsByThreadBeforeParams(
                            scope=self._scope,
                            thread_id=thread_id,
                            before_checkpoint_id=before_checkpoint_id,
                            limit=_LIST_PAGE_SIZE,
                        ),
                        self._query_options(cursor),
                    )
                else:
                    page = client.list_checkpoints_by_thread(
                        ListCheckpointsByThreadParams(
                            scope=self._scope,
                            thread_id=thread_id,
                            limit=_LIST_PAGE_SIZE,
                        ),
                        self._query_options(cursor),
                    )
            else:
                if before_checkpoint_id is not None:
                    page = client.list_checkpoints_by_scope_before(
                        ListCheckpointsByScopeBeforeParams(
                            scope=self._scope,
                            before_checkpoint_id=before_checkpoint_id,
                            limit=_LIST_PAGE_SIZE,
                        ),
                        self._query_options(cursor),
                    )
                else:
                    page = client.list_checkpoints_by_scope(
                        ListCheckpointsByScopeParams(
                            scope=self._scope,
                            limit=_LIST_PAGE_SIZE,
                        ),
                        self._query_options(cursor),
                    )

            for cp_row in page.value.checkpoints:
                if checkpoint_ns is not None and cp_row.checkpoint_ns != checkpoint_ns:
                    continue
                if (
                    config_checkpoint_id is not None
                    and cp_row.checkpoint_id != config_checkpoint_id
                ):
                    continue
                md = self.serde.loads_typed((cp_row.metadata_serde_type, cp_row.metadata_payload))
                if not self._matches_filter(md, filter):
                    continue
                yield self._hydrate_tuple_sync(client, cp_row, cp_row.writes, preloaded_metadata=md)
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        return

            if not page.next_cursor:
                break
            cursor = page.next_cursor

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        if limit is not None and limit <= 0:
            return
        client = await self._async_app()
        configurable = config.get("configurable", {}) if config else {}
        thread_id: str | None = (
            str(configurable["thread_id"]) if "thread_id" in configurable else None
        )
        checkpoint_ns: str | None = (
            str(configurable["checkpoint_ns"])
            if "checkpoint_ns" in configurable and configurable["checkpoint_ns"] is not None
            else None
        )
        config_checkpoint_id = get_checkpoint_id(config) if config else None
        before_checkpoint_id = get_checkpoint_id(before) if before else None

        remaining = limit
        if thread_id is not None and checkpoint_ns is not None and config_checkpoint_id is not None:
            if before_checkpoint_id is not None and config_checkpoint_id >= before_checkpoint_id:
                return
            res = await client.get_checkpoint_by_id(
                GetCheckpointByIdParams(
                    scope=self._scope,
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=config_checkpoint_id,
                ),
                self._query_options(),
            )
            if res.value.checkpoint is None:
                return
            md: CheckpointMetadata = self.serde.loads_typed(
                (res.value.checkpoint.metadata_serde_type, res.value.checkpoint.metadata_payload)
            )
            if self._matches_filter(md, filter):
                yield await self._hydrate_tuple_async(
                    client,
                    res.value.checkpoint,
                    res.value.writes,
                    preloaded_metadata=md,
                )
            return

        cursor: str | None = None
        for _ in range(_MAX_PAGES):
            if thread_id is not None and checkpoint_ns is not None:
                if before_checkpoint_id is not None:
                    page: Any = await client.list_checkpoints_by_thread_ns_before(
                        ListCheckpointsByThreadNsBeforeParams(
                            scope=self._scope,
                            thread_id=thread_id,
                            checkpoint_ns=checkpoint_ns,
                            before_checkpoint_id=before_checkpoint_id,
                            limit=_LIST_PAGE_SIZE,
                        ),
                        self._query_options(cursor),
                    )
                else:
                    page = await client.list_checkpoints_by_thread_ns(
                        ListCheckpointsByThreadNsParams(
                            scope=self._scope,
                            thread_id=thread_id,
                            checkpoint_ns=checkpoint_ns,
                            limit=_LIST_PAGE_SIZE,
                        ),
                        self._query_options(cursor),
                    )
            elif thread_id is not None:
                if before_checkpoint_id is not None:
                    page = await client.list_checkpoints_by_thread_before(
                        ListCheckpointsByThreadBeforeParams(
                            scope=self._scope,
                            thread_id=thread_id,
                            before_checkpoint_id=before_checkpoint_id,
                            limit=_LIST_PAGE_SIZE,
                        ),
                        self._query_options(cursor),
                    )
                else:
                    page = await client.list_checkpoints_by_thread(
                        ListCheckpointsByThreadParams(
                            scope=self._scope,
                            thread_id=thread_id,
                            limit=_LIST_PAGE_SIZE,
                        ),
                        self._query_options(cursor),
                    )
            else:
                if before_checkpoint_id is not None:
                    page = await client.list_checkpoints_by_scope_before(
                        ListCheckpointsByScopeBeforeParams(
                            scope=self._scope,
                            before_checkpoint_id=before_checkpoint_id,
                            limit=_LIST_PAGE_SIZE,
                        ),
                        self._query_options(cursor),
                    )
                else:
                    page = await client.list_checkpoints_by_scope(
                        ListCheckpointsByScopeParams(
                            scope=self._scope,
                            limit=_LIST_PAGE_SIZE,
                        ),
                        self._query_options(cursor),
                    )

            for cp_row in page.value.checkpoints:
                if checkpoint_ns is not None and cp_row.checkpoint_ns != checkpoint_ns:
                    continue
                if (
                    config_checkpoint_id is not None
                    and cp_row.checkpoint_id != config_checkpoint_id
                ):
                    continue
                md = self.serde.loads_typed((cp_row.metadata_serde_type, cp_row.metadata_payload))
                if not self._matches_filter(md, filter):
                    continue
                yield await self._hydrate_tuple_async(
                    client, cp_row, cp_row.writes, preloaded_metadata=md
                )
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        return

            if not page.next_cursor:
                break
            cursor = page.next_cursor

    def _prepare_put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> tuple[
        str,
        str,
        str,
        str | None,
        str,
        bytes,
        str,
        bytes,
        Sequence[CheckpointBlob],
    ]:
        c = checkpoint.copy()
        configurable = config["configurable"]
        thread_id = str(configurable["thread_id"])
        checkpoint_ns = str(configurable.get("checkpoint_ns", ""))
        checkpoint_id = str(c["id"])
        parent_checkpoint_id: str | None = (
            str(configurable["checkpoint_id"])
            if configurable.get("checkpoint_id") is not None
            else None
        )
        values: dict[str, Any] = c.pop("channel_values", {})  # type: ignore[misc]
        blobs: list[CheckpointBlob] = []
        for ch, ver in new_versions.items():
            if ch in values:
                serde_type, payload = self.serde.dumps_typed(values[ch])
            else:
                serde_type, payload = (_EMPTY_SERDE_TYPE, b"")
            blobs.append(
                CheckpointBlob(
                    scope=self._scope,
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id="",
                    blob_key=_make_blob_key(ch, ver),
                    channel=ch,
                    blob_version=str(ver),
                    serde_type=serde_type,
                    payload=payload,
                )
            )
        cp_serde_type, cp_payload = self.serde.dumps_typed(c)
        resolved_metadata = get_checkpoint_metadata(config, metadata)
        md_serde_type, md_payload = self.serde.dumps_typed(resolved_metadata)
        return (
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            parent_checkpoint_id,
            cp_serde_type,
            cp_payload,
            md_serde_type,
            md_payload,
            blobs,
        )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        (
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            parent_checkpoint_id,
            cp_serde_type,
            cp_payload,
            md_serde_type,
            md_payload,
            blobs,
        ) = self._prepare_put(config, checkpoint, metadata, new_versions)
        client = self._sync_app()
        if not blobs:
            res = client.put_checkpoint(
                PutCheckpointInput(
                    request_id=str(uuid4()),
                    scope=self._scope,
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint_id,
                    parent_checkpoint_id=parent_checkpoint_id,
                    checkpoint_serde_type=cp_serde_type,
                    checkpoint_payload=cp_payload,
                    metadata_serde_type=md_serde_type,
                    metadata_payload=md_payload,
                )
            )
            self._record_commit(res.commit_sequence)
        else:
            chunks = _chunk_blobs(blobs)
            for extra_chunk in chunks[1:]:
                blob_res = client.put_checkpoint_blobs(
                    PutCheckpointBlobsInput(
                        request_id=str(uuid4()),
                        scope=self._scope,
                        thread_id=thread_id,
                        checkpoint_ns=checkpoint_ns,
                        blobs=extra_chunk,
                    )
                )
                self._record_commit(blob_res.commit_sequence)
            res_with_blobs = client.put_checkpoint_with_blobs(
                PutCheckpointWithBlobsInput(
                    request_id=str(uuid4()),
                    scope=self._scope,
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint_id,
                    parent_checkpoint_id=parent_checkpoint_id,
                    checkpoint_serde_type=cp_serde_type,
                    checkpoint_payload=cp_payload,
                    metadata_serde_type=md_serde_type,
                    metadata_payload=md_payload,
                    blobs=chunks[0],
                )
            )
            self._record_commit(res_with_blobs.commit_sequence)
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        (
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            parent_checkpoint_id,
            cp_serde_type,
            cp_payload,
            md_serde_type,
            md_payload,
            blobs,
        ) = self._prepare_put(config, checkpoint, metadata, new_versions)
        client = await self._async_app()
        if not blobs:
            res = await client.put_checkpoint(
                PutCheckpointInput(
                    request_id=str(uuid4()),
                    scope=self._scope,
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint_id,
                    parent_checkpoint_id=parent_checkpoint_id,
                    checkpoint_serde_type=cp_serde_type,
                    checkpoint_payload=cp_payload,
                    metadata_serde_type=md_serde_type,
                    metadata_payload=md_payload,
                )
            )
            self._record_commit(res.commit_sequence)
        else:
            chunks = _chunk_blobs(blobs)
            for extra_chunk in chunks[1:]:
                blob_res = await client.put_checkpoint_blobs(
                    PutCheckpointBlobsInput(
                        request_id=str(uuid4()),
                        scope=self._scope,
                        thread_id=thread_id,
                        checkpoint_ns=checkpoint_ns,
                        blobs=extra_chunk,
                    )
                )
                self._record_commit(blob_res.commit_sequence)
            res_with_blobs = await client.put_checkpoint_with_blobs(
                PutCheckpointWithBlobsInput(
                    request_id=str(uuid4()),
                    scope=self._scope,
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint_id,
                    parent_checkpoint_id=parent_checkpoint_id,
                    checkpoint_serde_type=cp_serde_type,
                    checkpoint_payload=cp_payload,
                    metadata_serde_type=md_serde_type,
                    metadata_payload=md_payload,
                    blobs=chunks[0],
                )
            )
            self._record_commit(res_with_blobs.commit_sequence)
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def _prepare_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str,
    ) -> tuple[str, str, str, Sequence[CheckpointWriteInput]]:
        configurable = config["configurable"]
        thread_id = str(configurable["thread_id"])
        checkpoint_ns = str(configurable.get("checkpoint_ns", ""))
        checkpoint_id = str(configurable["checkpoint_id"])
        items: list[CheckpointWriteInput] = []
        seen_indices: set[int] = set()
        for idx, (channel, value) in enumerate(writes):
            write_idx = WRITES_IDX_MAP.get(channel, idx)
            allow_overwrite = write_idx < 0
            if not allow_overwrite and write_idx in seen_indices:
                continue
            seen_indices.add(write_idx)
            serde_type, payload = self.serde.dumps_typed(value)
            items.append(
                CheckpointWriteInput(
                    scope=self._scope,
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint_id,
                    task_id=task_id,
                    write_idx=write_idx,
                    channel=channel,
                    serde_type=serde_type,
                    payload=payload,
                    task_path=task_path,
                    allow_overwrite=allow_overwrite,
                )
            )
        return thread_id, checkpoint_ns, checkpoint_id, items

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        if not writes:
            return
        thread_id, checkpoint_ns, checkpoint_id, items = self._prepare_writes(
            config, writes, task_id, task_path
        )
        if not items:
            return
        client = self._sync_app()
        for chunk in _chunk_writes(items):
            res = client.put_checkpoint_writes(
                PutCheckpointWritesInput(
                    request_id=str(uuid4()),
                    scope=self._scope,
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint_id,
                    writes=chunk,
                )
            )
            self._record_commit(res.commit_sequence)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        if not writes:
            return
        thread_id, checkpoint_ns, checkpoint_id, items = self._prepare_writes(
            config, writes, task_id, task_path
        )
        if not items:
            return
        client = await self._async_app()
        for chunk in _chunk_writes(items):
            res = await client.put_checkpoint_writes(
                PutCheckpointWritesInput(
                    request_id=str(uuid4()),
                    scope=self._scope,
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint_id,
                    writes=chunk,
                )
            )
            self._record_commit(res.commit_sequence)

    def _delete_keys_sync(
        self,
        client: RiffDBLangGraphClient,
        cp_keys: Sequence[CheckpointKeyInput],
        blob_keys: Sequence[CheckpointBlobKeyInput],
        write_keys: Sequence[CheckpointWriteKeyInput],
    ) -> None:
        for i in range(0, len(write_keys), _DELETE_BATCH_MAX_COUNT):
            w_res = client.delete_checkpoint_writes(
                DeleteCheckpointWritesInput(
                    request_id=str(uuid4()),
                    scope=self._scope,
                    keys=tuple(write_keys[i : i + _DELETE_BATCH_MAX_COUNT]),
                )
            )
            self._record_commit(w_res.commit_sequence)
        for i in range(0, len(blob_keys), _DELETE_BATCH_MAX_COUNT):
            b_res = client.delete_checkpoint_blobs(
                DeleteCheckpointBlobsInput(
                    request_id=str(uuid4()),
                    scope=self._scope,
                    keys=tuple(blob_keys[i : i + _DELETE_BATCH_MAX_COUNT]),
                )
            )
            self._record_commit(b_res.commit_sequence)
        for i in range(0, len(cp_keys), _DELETE_BATCH_MAX_COUNT):
            c_res = client.delete_checkpoints(
                DeleteCheckpointsInput(
                    request_id=str(uuid4()),
                    scope=self._scope,
                    keys=tuple(cp_keys[i : i + _DELETE_BATCH_MAX_COUNT]),
                )
            )
            self._record_commit(c_res.commit_sequence)

    async def _delete_keys_async(
        self,
        client: AsyncRiffDBLangGraphClient,
        cp_keys: Sequence[CheckpointKeyInput],
        blob_keys: Sequence[CheckpointBlobKeyInput],
        write_keys: Sequence[CheckpointWriteKeyInput],
    ) -> None:
        for i in range(0, len(write_keys), _DELETE_BATCH_MAX_COUNT):
            w_res = await client.delete_checkpoint_writes(
                DeleteCheckpointWritesInput(
                    request_id=str(uuid4()),
                    scope=self._scope,
                    keys=tuple(write_keys[i : i + _DELETE_BATCH_MAX_COUNT]),
                )
            )
            self._record_commit(w_res.commit_sequence)
        for i in range(0, len(blob_keys), _DELETE_BATCH_MAX_COUNT):
            b_res = await client.delete_checkpoint_blobs(
                DeleteCheckpointBlobsInput(
                    request_id=str(uuid4()),
                    scope=self._scope,
                    keys=tuple(blob_keys[i : i + _DELETE_BATCH_MAX_COUNT]),
                )
            )
            self._record_commit(b_res.commit_sequence)
        for i in range(0, len(cp_keys), _DELETE_BATCH_MAX_COUNT):
            c_res = await client.delete_checkpoints(
                DeleteCheckpointsInput(
                    request_id=str(uuid4()),
                    scope=self._scope,
                    keys=tuple(cp_keys[i : i + _DELETE_BATCH_MAX_COUNT]),
                )
            )
            self._record_commit(c_res.commit_sequence)

    def _collect_thread_keys_sync(
        self, client: RiffDBLangGraphClient, thread_id: str
    ) -> tuple[
        Sequence[CheckpointKeyInput],
        Sequence[CheckpointBlobKeyInput],
        Sequence[CheckpointWriteKeyInput],
    ]:
        cp_pages = client.list_thread_checkpoint_keys_pages(
            ListThreadCheckpointKeysParams(
                scope=self._scope, thread_id=thread_id, limit=_KEY_LIST_PAGE_SIZE
            ),
            max_pages=_MAX_PAGES,
            options=self._query_options(),
        )
        cp_keys = [
            CheckpointKeyInput(
                scope=self._scope,
                thread_id=row.thread_id,
                checkpoint_ns=row.checkpoint_ns,
                checkpoint_id=row.checkpoint_id,
            )
            for page in cp_pages
            for row in page.value.checkpoints
        ]

        blob_pages = client.list_thread_blob_keys_pages(
            ListThreadBlobKeysParams(
                scope=self._scope, thread_id=thread_id, limit=_KEY_LIST_PAGE_SIZE
            ),
            max_pages=_MAX_PAGES,
            options=self._query_options(),
        )
        blob_keys = [
            CheckpointBlobKeyInput(
                scope=self._scope,
                thread_id=row.thread_id,
                checkpoint_ns=row.checkpoint_ns,
                checkpoint_id=row.checkpoint_id,
                blob_key=row.blob_key,
            )
            for page in blob_pages
            for row in page.value.blobs
        ]

        write_pages = client.list_thread_write_keys_pages(
            ListThreadWriteKeysParams(
                scope=self._scope, thread_id=thread_id, limit=_KEY_LIST_PAGE_SIZE
            ),
            max_pages=_MAX_PAGES,
            options=self._query_options(),
        )
        write_keys = [
            CheckpointWriteKeyInput(
                scope=self._scope,
                thread_id=row.thread_id,
                checkpoint_ns=row.checkpoint_ns,
                checkpoint_id=row.checkpoint_id,
                task_id=row.task_id,
                write_idx=row.write_idx,
            )
            for page in write_pages
            for row in page.value.writes
        ]
        return cp_keys, blob_keys, write_keys

    async def _collect_thread_keys_async(
        self, client: AsyncRiffDBLangGraphClient, thread_id: str
    ) -> tuple[
        Sequence[CheckpointKeyInput],
        Sequence[CheckpointBlobKeyInput],
        Sequence[CheckpointWriteKeyInput],
    ]:
        cp_pages = await client.list_thread_checkpoint_keys_pages(
            ListThreadCheckpointKeysParams(
                scope=self._scope, thread_id=thread_id, limit=_KEY_LIST_PAGE_SIZE
            ),
            max_pages=_MAX_PAGES,
            options=self._query_options(),
        )
        cp_keys = [
            CheckpointKeyInput(
                scope=self._scope,
                thread_id=row.thread_id,
                checkpoint_ns=row.checkpoint_ns,
                checkpoint_id=row.checkpoint_id,
            )
            for page in cp_pages
            for row in page.value.checkpoints
        ]

        blob_pages = await client.list_thread_blob_keys_pages(
            ListThreadBlobKeysParams(
                scope=self._scope, thread_id=thread_id, limit=_KEY_LIST_PAGE_SIZE
            ),
            max_pages=_MAX_PAGES,
            options=self._query_options(),
        )
        blob_keys = [
            CheckpointBlobKeyInput(
                scope=self._scope,
                thread_id=row.thread_id,
                checkpoint_ns=row.checkpoint_ns,
                checkpoint_id=row.checkpoint_id,
                blob_key=row.blob_key,
            )
            for page in blob_pages
            for row in page.value.blobs
        ]

        write_pages = await client.list_thread_write_keys_pages(
            ListThreadWriteKeysParams(
                scope=self._scope, thread_id=thread_id, limit=_KEY_LIST_PAGE_SIZE
            ),
            max_pages=_MAX_PAGES,
            options=self._query_options(),
        )
        write_keys = [
            CheckpointWriteKeyInput(
                scope=self._scope,
                thread_id=row.thread_id,
                checkpoint_ns=row.checkpoint_ns,
                checkpoint_id=row.checkpoint_id,
                task_id=row.task_id,
                write_idx=row.write_idx,
            )
            for page in write_pages
            for row in page.value.writes
        ]
        return cp_keys, blob_keys, write_keys

    def delete_thread(self, thread_id: str) -> None:
        client = self._sync_app()
        cp_keys, blob_keys, write_keys = self._collect_thread_keys_sync(client, thread_id)
        self._delete_keys_sync(client, cp_keys, blob_keys, write_keys)

    async def adelete_thread(self, thread_id: str) -> None:
        client = await self._async_app()
        cp_keys, blob_keys, write_keys = await self._collect_thread_keys_async(client, thread_id)
        await self._delete_keys_async(client, cp_keys, blob_keys, write_keys)

    def delete_for_runs(self, run_ids: Sequence[str]) -> None:
        run_id_set = {str(r) for r in run_ids if r}
        if not run_id_set:
            return
        client = self._sync_app()
        pages = client.list_checkpoints_by_scope_pages(
            ListCheckpointsByScopeParams(scope=self._scope, limit=_LIST_PAGE_SIZE),
            max_pages=_MAX_PAGES,
            options=self._query_options(),
        )
        affected_threads: set[str] = set()
        cp_keys_by_thread: dict[str, list[CheckpointKeyInput]] = {}
        target_cps: set[tuple[str, str, str]] = set()
        for page in pages:
            for cp_row in page.value.checkpoints:
                md: CheckpointMetadata = self.serde.loads_typed(
                    (cp_row.metadata_serde_type, cp_row.metadata_payload)
                )
                if md.get("run_id") in run_id_set:
                    affected_threads.add(cp_row.thread_id)
                    target_cps.add((cp_row.thread_id, cp_row.checkpoint_ns, cp_row.checkpoint_id))
                    cp_keys_by_thread.setdefault(cp_row.thread_id, []).append(
                        CheckpointKeyInput(
                            scope=self._scope,
                            thread_id=cp_row.thread_id,
                            checkpoint_ns=cp_row.checkpoint_ns,
                            checkpoint_id=cp_row.checkpoint_id,
                        )
                    )
        for tid in sorted(affected_threads):
            _, _, all_write_keys = self._collect_thread_keys_sync(client, tid)
            matching_writes = [
                wk
                for wk in all_write_keys
                if (wk.thread_id, wk.checkpoint_ns, wk.checkpoint_id) in target_cps
            ]
            self._delete_keys_sync(client, cp_keys_by_thread.get(tid, []), [], matching_writes)

    async def adelete_for_runs(self, run_ids: Sequence[str]) -> None:
        run_id_set = {str(r) for r in run_ids if r}
        if not run_id_set:
            return
        client = await self._async_app()
        pages = await client.list_checkpoints_by_scope_pages(
            ListCheckpointsByScopeParams(scope=self._scope, limit=_LIST_PAGE_SIZE),
            max_pages=_MAX_PAGES,
            options=self._query_options(),
        )
        affected_threads: set[str] = set()
        cp_keys_by_thread: dict[str, list[CheckpointKeyInput]] = {}
        target_cps: set[tuple[str, str, str]] = set()
        for page in pages:
            for cp_row in page.value.checkpoints:
                md: CheckpointMetadata = self.serde.loads_typed(
                    (cp_row.metadata_serde_type, cp_row.metadata_payload)
                )
                if md.get("run_id") in run_id_set:
                    affected_threads.add(cp_row.thread_id)
                    target_cps.add((cp_row.thread_id, cp_row.checkpoint_ns, cp_row.checkpoint_id))
                    cp_keys_by_thread.setdefault(cp_row.thread_id, []).append(
                        CheckpointKeyInput(
                            scope=self._scope,
                            thread_id=cp_row.thread_id,
                            checkpoint_ns=cp_row.checkpoint_ns,
                            checkpoint_id=cp_row.checkpoint_id,
                        )
                    )
        for tid in sorted(affected_threads):
            _, _, all_write_keys = await self._collect_thread_keys_async(client, tid)
            matching_writes = [
                wk
                for wk in all_write_keys
                if (wk.thread_id, wk.checkpoint_ns, wk.checkpoint_id) in target_cps
            ]
            await self._delete_keys_async(
                client, cp_keys_by_thread.get(tid, []), [], matching_writes
            )

    def copy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        if source_thread_id == target_thread_id:
            return
        client = self._sync_app()
        pages = client.list_checkpoints_by_thread_pages(
            ListCheckpointsByThreadParams(
                scope=self._scope, thread_id=source_thread_id, limit=_LIST_PAGE_SIZE
            ),
            max_pages=_MAX_PAGES,
            options=self._query_options(),
        )
        cp_rows = [cp for page in pages for cp in page.value.checkpoints]
        if not cp_rows:
            return

        _, src_blob_keys, _ = self._collect_thread_keys_sync(client, source_thread_id)
        keys_by_ns: dict[str, list[str]] = {}
        for bk in src_blob_keys:
            keys_by_ns.setdefault(bk.checkpoint_ns, []).append(bk.blob_key)
        for ns, bks in keys_by_ns.items():
            unique_bks = sorted(set(bks))
            target_blobs: list[CheckpointBlob] = []
            for i in range(0, len(unique_bks), _GET_BLOBS_CHUNK_SIZE):
                chunk = tuple(unique_bks[i : i + _GET_BLOBS_CHUNK_SIZE])
                b_res = client.get_checkpoint_blobs(
                    GetCheckpointBlobsParams(
                        scope=self._scope,
                        thread_id=source_thread_id,
                        checkpoint_ns=ns,
                        blob_keys=chunk,
                    ),
                    self._query_options(),
                )
                for b in b_res.value.blobs:
                    target_blobs.append(
                        CheckpointBlob(
                            scope=self._scope,
                            thread_id=target_thread_id,
                            checkpoint_ns=ns,
                            checkpoint_id="",
                            blob_key=b.blob_key,
                            channel=b.channel,
                            blob_version=b.blob_version,
                            serde_type=b.serde_type,
                            payload=b.payload,
                        )
                    )
            for b_chunk in _chunk_blobs(target_blobs):
                b_put = client.put_checkpoint_blobs(
                    PutCheckpointBlobsInput(
                        request_id=str(uuid4()),
                        scope=self._scope,
                        thread_id=target_thread_id,
                        checkpoint_ns=ns,
                        blobs=b_chunk,
                    )
                )
                self._record_commit(b_put.commit_sequence)

        for cp_row in reversed(cp_rows):
            res = client.put_checkpoint(
                PutCheckpointInput(
                    request_id=str(uuid4()),
                    scope=self._scope,
                    thread_id=target_thread_id,
                    checkpoint_ns=cp_row.checkpoint_ns,
                    checkpoint_id=cp_row.checkpoint_id,
                    parent_checkpoint_id=cp_row.parent_checkpoint_id,
                    checkpoint_serde_type=cp_row.checkpoint_serde_type,
                    checkpoint_payload=cp_row.checkpoint_payload,
                    metadata_serde_type=cp_row.metadata_serde_type,
                    metadata_payload=cp_row.metadata_payload,
                )
            )
            self._record_commit(res.commit_sequence)

            full_cp = client.get_checkpoint_by_id(
                GetCheckpointByIdParams(
                    scope=self._scope,
                    thread_id=source_thread_id,
                    checkpoint_ns=cp_row.checkpoint_ns,
                    checkpoint_id=cp_row.checkpoint_id,
                ),
                self._query_options(),
            )
            if full_cp.value.writes:
                write_inputs = [
                    CheckpointWriteInput(
                        scope=self._scope,
                        thread_id=target_thread_id,
                        checkpoint_ns=cp_row.checkpoint_ns,
                        checkpoint_id=cp_row.checkpoint_id,
                        task_id=w.task_id,
                        write_idx=w.write_idx,
                        channel=w.channel,
                        serde_type=w.serde_type,
                        payload=w.payload,
                        task_path=w.task_path,
                        allow_overwrite=True,
                    )
                    for w in full_cp.value.writes
                ]
                for w_chunk in _chunk_writes(write_inputs):
                    w_res = client.put_checkpoint_writes(
                        PutCheckpointWritesInput(
                            request_id=str(uuid4()),
                            scope=self._scope,
                            thread_id=target_thread_id,
                            checkpoint_ns=cp_row.checkpoint_ns,
                            checkpoint_id=cp_row.checkpoint_id,
                            writes=w_chunk,
                        )
                    )
                    self._record_commit(w_res.commit_sequence)

    async def acopy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        if source_thread_id == target_thread_id:
            return
        client = await self._async_app()
        pages = await client.list_checkpoints_by_thread_pages(
            ListCheckpointsByThreadParams(
                scope=self._scope, thread_id=source_thread_id, limit=_LIST_PAGE_SIZE
            ),
            max_pages=_MAX_PAGES,
            options=self._query_options(),
        )
        cp_rows = [cp for page in pages for cp in page.value.checkpoints]
        if not cp_rows:
            return

        _, src_blob_keys, _ = await self._collect_thread_keys_async(client, source_thread_id)
        keys_by_ns: dict[str, list[str]] = {}
        for bk in src_blob_keys:
            keys_by_ns.setdefault(bk.checkpoint_ns, []).append(bk.blob_key)
        for ns, bks in keys_by_ns.items():
            unique_bks = sorted(set(bks))
            target_blobs: list[CheckpointBlob] = []
            for i in range(0, len(unique_bks), _GET_BLOBS_CHUNK_SIZE):
                chunk = tuple(unique_bks[i : i + _GET_BLOBS_CHUNK_SIZE])
                b_res = await client.get_checkpoint_blobs(
                    GetCheckpointBlobsParams(
                        scope=self._scope,
                        thread_id=source_thread_id,
                        checkpoint_ns=ns,
                        blob_keys=chunk,
                    ),
                    self._query_options(),
                )
                for b in b_res.value.blobs:
                    target_blobs.append(
                        CheckpointBlob(
                            scope=self._scope,
                            thread_id=target_thread_id,
                            checkpoint_ns=ns,
                            checkpoint_id="",
                            blob_key=b.blob_key,
                            channel=b.channel,
                            blob_version=b.blob_version,
                            serde_type=b.serde_type,
                            payload=b.payload,
                        )
                    )
            for b_chunk in _chunk_blobs(target_blobs):
                b_put = await client.put_checkpoint_blobs(
                    PutCheckpointBlobsInput(
                        request_id=str(uuid4()),
                        scope=self._scope,
                        thread_id=target_thread_id,
                        checkpoint_ns=ns,
                        blobs=b_chunk,
                    )
                )
                self._record_commit(b_put.commit_sequence)

        for cp_row in reversed(cp_rows):
            res = await client.put_checkpoint(
                PutCheckpointInput(
                    request_id=str(uuid4()),
                    scope=self._scope,
                    thread_id=target_thread_id,
                    checkpoint_ns=cp_row.checkpoint_ns,
                    checkpoint_id=cp_row.checkpoint_id,
                    parent_checkpoint_id=cp_row.parent_checkpoint_id,
                    checkpoint_serde_type=cp_row.checkpoint_serde_type,
                    checkpoint_payload=cp_row.checkpoint_payload,
                    metadata_serde_type=cp_row.metadata_serde_type,
                    metadata_payload=cp_row.metadata_payload,
                )
            )
            self._record_commit(res.commit_sequence)

            full_cp = await client.get_checkpoint_by_id(
                GetCheckpointByIdParams(
                    scope=self._scope,
                    thread_id=source_thread_id,
                    checkpoint_ns=cp_row.checkpoint_ns,
                    checkpoint_id=cp_row.checkpoint_id,
                ),
                self._query_options(),
            )
            if full_cp.value.writes:
                write_inputs = [
                    CheckpointWriteInput(
                        scope=self._scope,
                        thread_id=target_thread_id,
                        checkpoint_ns=cp_row.checkpoint_ns,
                        checkpoint_id=cp_row.checkpoint_id,
                        task_id=w.task_id,
                        write_idx=w.write_idx,
                        channel=w.channel,
                        serde_type=w.serde_type,
                        payload=w.payload,
                        task_path=w.task_path,
                        allow_overwrite=True,
                    )
                    for w in full_cp.value.writes
                ]
                for w_chunk in _chunk_writes(write_inputs):
                    w_res = await client.put_checkpoint_writes(
                        PutCheckpointWritesInput(
                            request_id=str(uuid4()),
                            scope=self._scope,
                            thread_id=target_thread_id,
                            checkpoint_ns=cp_row.checkpoint_ns,
                            checkpoint_id=cp_row.checkpoint_id,
                            writes=w_chunk,
                        )
                    )
                    self._record_commit(w_res.commit_sequence)

    def prune(
        self,
        thread_ids: Sequence[str],
        *,
        strategy: str = "keep_latest",
    ) -> None:
        if strategy not in ("keep_latest", "delete"):
            raise ValueError(f"unsupported prune strategy: {strategy!r}")
        if not thread_ids:
            return
        client = self._sync_app()
        for tid in thread_ids:
            if strategy == "delete":
                self.delete_thread(tid)
                continue
            cp_keys, _, write_keys = self._collect_thread_keys_sync(client, tid)
            if not cp_keys:
                continue
            latest_by_ns: dict[str, str] = {}
            for ck in cp_keys:
                prev = latest_by_ns.get(ck.checkpoint_ns)
                if prev is None or ck.checkpoint_id > prev:
                    latest_by_ns[ck.checkpoint_ns] = ck.checkpoint_id
            delete_cp_keys = [
                ck for ck in cp_keys if ck.checkpoint_id != latest_by_ns.get(ck.checkpoint_ns)
            ]
            delete_write_keys = [
                wk for wk in write_keys if wk.checkpoint_id != latest_by_ns.get(wk.checkpoint_ns)
            ]
            self._delete_keys_sync(client, delete_cp_keys, [], delete_write_keys)

    async def aprune(
        self,
        thread_ids: Sequence[str],
        *,
        strategy: str = "keep_latest",
    ) -> None:
        if strategy not in ("keep_latest", "delete"):
            raise ValueError(f"unsupported prune strategy: {strategy!r}")
        if not thread_ids:
            return
        client = await self._async_app()
        for tid in thread_ids:
            if strategy == "delete":
                await self.adelete_thread(tid)
                continue
            cp_keys, _, write_keys = await self._collect_thread_keys_async(client, tid)
            if not cp_keys:
                continue
            latest_by_ns: dict[str, str] = {}
            for ck in cp_keys:
                prev = latest_by_ns.get(ck.checkpoint_ns)
                if prev is None or ck.checkpoint_id > prev:
                    latest_by_ns[ck.checkpoint_ns] = ck.checkpoint_id
            delete_cp_keys = [
                ck for ck in cp_keys if ck.checkpoint_id != latest_by_ns.get(ck.checkpoint_ns)
            ]
            delete_write_keys = [
                wk for wk in write_keys if wk.checkpoint_id != latest_by_ns.get(wk.checkpoint_ns)
            ]
            await self._delete_keys_async(client, delete_cp_keys, [], delete_write_keys)

    def get_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, DeltaChannelHistory]:
        return super().get_delta_channel_history(config=config, channels=channels)

    async def aget_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, DeltaChannelHistory]:
        return await super().aget_delta_channel_history(config=config, channels=channels)
