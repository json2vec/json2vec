from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lightning.pytorch import Callback, Trainer
from loguru import logger

from json2vec.distributed import all_gather_object, broadcast_object, is_distributed, is_rank_zero

if TYPE_CHECKING:
    from json2vec.architecture.root import JSON2Vec
    from json2vec.structs.tree import Address


# FIXME AI slop code here... This should be moved to the category extension, and/or support abilities to add callbacks to the root module
# FIXME this should really be cleaned up though ...

class VocabularySyncCallback(Callback):
    """Synchronize online categorical vocabularies across distributed ranks.

    Rank zero owns category id assignment. Other ranks propose unseen labels and
    use the reserved unavailable bucket until the next synchronization point.
    """

    def _vocabularies(self, pl_module: JSON2Vec) -> dict[Address, Any]:
        vocabularies: dict[Address, Any] = {}
        for address, node in pl_module.nodes.items():
            embedder = getattr(node, "embedder", None)
            vocab = getattr(embedder, "vocab", None)
            required = ("snapshot", "drain_proposals", "extend", "load_snapshot")
            if vocab is not None and all(hasattr(vocab, name) for name in required):
                vocabularies[address] = vocab

        return vocabularies

    def _sync(self, trainer: Trainer, pl_module: JSON2Vec, reason: str) -> None:
        if not is_distributed():
            return

        vocabularies = self._vocabularies(pl_module=pl_module)
        if not vocabularies:
            return

        local_proposals = {address: vocab.drain_proposals() for address, vocab in vocabularies.items()}
        gathered = all_gather_object(local_proposals)

        payload = None
        if is_rank_zero():
            snapshots = {}
            stats = {}
            for address, vocab in vocabularies.items():
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
            vocabularies[address].load_snapshot(snapshot)

        trainer.strategy.barrier(name=f"vocabulary-sync-{reason}")

        if is_rank_zero():
            for address, stats in payload["stats"].items():
                logger.bind(
                    component="vocabulary",
                    reason=reason,
                    address=address,
                    **stats,
                ).info("synchronized vocabulary")

                if stats["max"] > 0 and stats["size"] / stats["max"] >= 0.95:
                    logger.bind(
                        component="vocabulary",
                        address=address,
                        size=stats["size"],
                        max=stats["max"],
                    ).warning("vocabulary is near capacity")

    def on_fit_start(self, trainer: Trainer, pl_module: JSON2Vec) -> None:
        self._sync(trainer=trainer, pl_module=pl_module, reason="fit_start")

    def on_train_epoch_end(self, trainer: Trainer, pl_module: JSON2Vec) -> None:
        self._sync(trainer=trainer, pl_module=pl_module, reason="train_epoch_end")
