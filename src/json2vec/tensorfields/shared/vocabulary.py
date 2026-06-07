from __future__ import annotations

from dataclasses import dataclass
from functools import partialmethod
from multiprocessing import Manager
from multiprocessing.managers import ListProxy, SyncManager
from threading import RLock
from typing import TYPE_CHECKING, Any, Callable

import torch
from lightning.pytorch import Callback, Trainer
from loguru import logger

from json2vec.distributed import all_gather_object, broadcast_object, is_distributed, is_rank_zero
from json2vec.structs.tree import Address

if TYPE_CHECKING:
    from json2vec.architecture.root import Model


class _LocalLock:
    """Pickle-friendly local lock used outside multiprocessing data workers."""

    def __init__(self) -> None:
        self._lock = RLock()

    def __enter__(self):
        return self._lock.__enter__()

    def __exit__(self, *args):
        return self._lock.__exit__(*args)

    def __getstate__(self) -> dict[str, Any]:
        return {}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._lock = RLock()


@dataclass
class _VocabularyStorage:
    master: list[str] | ListProxy[str]
    lock: Any
    proposals: list[str] | ListProxy[str]
    proposal_lock: Any


class VocabularyState:
    def __init__(
        self,
        storage: _VocabularyStorage,
        max_vocab_size: int,
        share: Callable[[], _VocabularyStorage] | None = None,
    ):
        self.storage = storage
        self.max_vocab_size: int = max_vocab_size
        self._share = share
        self.vocab: list[str] = []
        self.index: dict[str, int] = {}
        self.global_rank: int = 0
        self.world_size: int = 1
        self.refresh(force=True)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_share"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def configure_distributed(self, global_rank: int = 0, world_size: int = 1) -> None:
        self.global_rank = global_rank
        self.world_size = world_size

    def share(self) -> None:
        if self._share is None:
            return

        self.storage = self._share()
        self.refresh(force=True)

    def refresh(self, force: bool = False) -> None:
        if not force and len(self.master) == len(self.vocab):
            return

        self.vocab = list(self.master)
        self.index = {word: index for index, word in enumerate(self.vocab)}

    @property
    def master(self) -> list[str] | ListProxy[str]:
        return self.storage.master

    @property
    def lock(self) -> Any:
        return self.storage.lock

    @property
    def proposals(self) -> list[str] | ListProxy[str]:
        return self.storage.proposals

    @property
    def proposal_lock(self) -> Any:
        return self.storage.proposal_lock

    @property
    def unavailable_index(self) -> int:
        return self.max_vocab_size

    def __call__(self, word: str, update: bool) -> int | None:
        if word is None:
            return None

        if word in self.index:
            return self.index[word]

        if not update:
            self.refresh()
            if word in self.index:
                return self.index[word]

            # Validation/test/inference should preserve "field exists" semantics even when
            # the label was never seen during training.
            return self.unavailable_index

        if self.global_rank != 0:
            with self.proposal_lock:
                if word not in self.proposals:
                    self.proposals.append(word)

            return self.unavailable_index

        # OK, it is not known locally... We will lock the global state and update the local vocab
        with self.lock:
            self.refresh(force=True)

            if word in self.index:
                return self.index[word]

            if len(self.vocab) >= self.max_vocab_size:
                # Once the learned vocabulary is full, new labels also fall back to the
                # reserved unavailable bucket instead of evicting existing ids.
                return self.unavailable_index

            if word not in self.vocab:
                self.vocab.append(word)
                self.index[word] = len(self.vocab) - 1
                self.master.append(word)

        return self.index[word]

    def __len__(self) -> int:
        self.refresh()
        return len(self.vocab)


class OnlineVocabularyModel(torch.nn.Module):
    @classmethod
    def from_model(cls, module: Model) -> dict[Address, OnlineVocabularyModel]:
        resources: dict[Address, OnlineVocabularyModel] = {}

        for address, node in module.nodes.items():
            embedder = getattr(node, "embedder", None)
            vocabulary = getattr(embedder, "vocab", None)
            if isinstance(vocabulary, cls):
                resources[address] = vocabulary

        return resources

    def __init__(self, max_vocab_size: int):
        super().__init__()

        self.max_vocab_size: int = max_vocab_size
        self.manager: SyncManager | None = None
        self.master: list[str] | ListProxy[str] = []
        self.lock: Any = _LocalLock()
        self.proposals: list[str] | ListProxy[str] = []
        self.proposal_lock: Any = _LocalLock()
        self._snapshot_cache: list[str] | None = None
        self._snapshot_size: int = -1

    @property
    def storage(self) -> _VocabularyStorage:
        return _VocabularyStorage(
            master=self.master,
            lock=self.lock,
            proposals=self.proposals,
            proposal_lock=self.proposal_lock,
        )

    @property
    def is_shared(self) -> bool:
        return self.manager is not None

    def share(self) -> None:
        """Move vocabulary state into multiprocessing-safe storage."""
        if self.manager is not None:
            return

        master = list(self.master)
        proposals = list(self.proposals)
        self.manager = Manager()
        self.master = self.manager.list(master)
        self.lock = self.manager.Lock()
        self.proposals = self.manager.list(proposals)
        self.proposal_lock = self.manager.Lock()
        self._snapshot_cache = None
        self._snapshot_size = -1

    def freeze(self) -> None:
        """Snapshot vocabulary state back into local, read-optimized storage."""
        if self.manager is None:
            return

        manager = self.manager
        self.master = list(self.master)
        self.lock = _LocalLock()
        self.proposals = []
        self.proposal_lock = _LocalLock()
        self.manager = None
        self._snapshot_cache = None
        self._snapshot_size = -1
        manager.shutdown()

    def _shared_state(self) -> _VocabularyStorage:
        self.share()
        return self.storage

    def _save_to_state_dict(self, state_dict, prefix, keep_vars):  # ty:ignore[invalid-method-override]
        super()._save_to_state_dict(state_dict, prefix, keep_vars)
        state_dict[prefix + "vocabulary"] = list(self.master)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        key = prefix + "vocabulary"
        if key in state_dict:
            vocab: list[str] = state_dict.pop(key)
            self.load_snapshot(vocab)
            self._snapshot_cache = None
            self._snapshot_size = -1
        else:
            missing_keys.append(key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @property
    def state(self) -> VocabularyState:
        return VocabularyState(
            storage=self.storage,
            max_vocab_size=self.max_vocab_size,
            share=self._shared_state,
        )

    def snapshot(self) -> list[str]:
        size = len(self.master)
        if self._snapshot_cache is None or self._snapshot_size != size:
            self._snapshot_cache = list(self.master)
            self._snapshot_size = size

        return self._snapshot_cache

    def drain_proposals(self) -> list[str]:
        with self.proposal_lock:
            proposals = list(self.proposals)
            self.proposals[:] = []

        return proposals

    def extend(self, proposals: list[str]) -> tuple[int, int]:
        accepted = 0
        rejected = 0

        with self.lock:
            vocab = list(self.master)
            index = set(vocab)
            for word in proposals:
                if word in index:
                    continue

                if len(vocab) >= self.max_vocab_size:
                    rejected += 1
                    continue

                vocab.append(word)
                index.add(word)
                self.master.append(word)
                accepted += 1

        if accepted:
            self._snapshot_cache = None
            self._snapshot_size = -1

        return accepted, rejected

    def load_snapshot(self, vocabulary: list[str]) -> None:
        with self.lock:
            self.master[:] = vocabulary[: self.max_vocab_size]

        with self.proposal_lock:
            self.proposals[:] = []

        self._snapshot_cache = None
        self._snapshot_size = -1


def sync(_callback: Callback, trainer: Trainer, pl_module: Model, reason: str) -> None:
    if not is_distributed():
        return

    resources = OnlineVocabularyModel.from_model(pl_module)
    if not resources:
        return

    local_proposals = {address: vocab.drain_proposals() for address, vocab in resources.items()}
    gathered = all_gather_object(local_proposals)

    payload = None
    if is_rank_zero():
        snapshots = {}
        stats = {}
        for address, vocab in resources.items():
            proposals = []
            for rank_proposals in gathered:
                proposals.extend(rank_proposals.get(address, []))

            accepted, rejected = vocab.extend(proposals)
            snapshot = vocab.snapshot()
            snapshots[address] = snapshot
            stats[address] = {
                "proposed": len(proposals),
                "accepted": accepted,
                "rejected_full": rejected,
                "size": len(snapshot),
                "max": vocab.max_vocab_size,
            }

        payload = {"snapshots": snapshots, "stats": stats}

    payload = broadcast_object(payload, src=0)

    for address, snapshot in payload["snapshots"].items():
        resources[address].load_snapshot(snapshot)

    trainer.strategy.barrier(name=f"vocabulary-sync-{reason}")

    if is_rank_zero():
        for address, stats in payload["stats"].items():
            if stats["max"] > 0 and stats["size"] / stats["max"] >= 0.95:
                logger.bind(
                    component="vocabulary",
                    address=address,
                    size=stats["size"],
                    max=stats["max"],
                ).warning("vocabulary is near capacity")


class VocabularySyncCallback(Callback):
    """Synchronize online vocabularies registered by tensorfield extensions."""

    on_fit_start = partialmethod(sync, reason="fit_start")
    on_train_epoch_end = partialmethod(sync, reason="train_epoch_end")

    def on_fit_end(self, trainer: Trainer, pl_module: Model) -> None:  # ty:ignore[invalid-method-override]
        for vocabulary in OnlineVocabularyModel.from_model(pl_module).values():
            vocabulary.freeze()


__all__ = [
    "OnlineVocabularyModel",
    "VocabularyState",
    "VocabularySyncCallback",
]
