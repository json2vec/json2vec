import datetime
from types import SimpleNamespace

from loguru import logger

from json2vec.logging.throughput import ThroughputLogger
from json2vec.structs.enums import Strata


def test_throughput_logger_logs_once_per_epoch():
    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{message}")
    callback = ThroughputLogger()
    callback.timestamp[Strata.train] = datetime.datetime.now() - datetime.timedelta(seconds=2)
    module = SimpleNamespace(batch_size=10)

    try:
        callback.count(trainer=object(), pl_module=module, outputs=None, batch=None, batch_idx=0, strata=Strata.train)
        callback.count(trainer=object(), pl_module=module, outputs=None, batch=None, batch_idx=1, strata=Strata.train)
        assert messages == []

        callback.end(trainer=object(), pl_module=module, strata=Strata.train)
    finally:
        logger.remove(sink_id)

    assert len(messages) == 1
    assert any("train epoch throughput:" in message for message in messages)
    assert any("observations/s" in message for message in messages)
