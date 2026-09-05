import io
import logging
from logging.handlers import BufferingHandler

import pytest
from rich.console import Console

from relflow.logging.config import CONTEXT, Handler, Logger, configure


def test_logger_preserves_bound_context_on_standard_records():
    core = logging.Logger("relflow-test", level=logging.DEBUG)
    records = BufferingHandler(capacity=10)
    core.addHandler(records)
    logger = Logger(core, {}).bind(component="model", rank=2)

    logger.bind(epoch=4).info("started training")

    record = records.buffer[0]
    assert record.getMessage() == "started training"
    assert getattr(record, CONTEXT) == {"component": "model", "rank": 2, "epoch": 4}


def test_rich_handler_renders_component_message_and_context():
    stream = io.StringIO()
    output = Console(file=stream, force_terminal=False, width=120)
    core = logging.Logger("relflow-rich-test", level=logging.DEBUG)
    core.addHandler(
        Handler(
            console=output,
            show_time=False,
            show_level=True,
            show_path=False,
        )
    )
    logger = Logger(core, {})

    logger.bind(component="tensorfield", address="record/amount", count=12, trainable=True).warning(
        "values exceeded the safe range"
    )

    rendered = stream.getvalue()
    assert "WARNING" in rendered
    assert "TENSORFIELD" in rendered
    assert "values exceeded the safe range" in rendered
    assert "address=record/amount" in rendered
    assert "count=12" in rendered
    assert "trainable=true" in rendered
    assert "•" in rendered


@pytest.mark.parametrize("level", ["verbose", "WARN", "NOTSET", True, object()])
def test_configure_rejects_invalid_levels(level: object):
    with pytest.raises((TypeError, ValueError), match="RelFlow log level"):
        configure(level=level)
