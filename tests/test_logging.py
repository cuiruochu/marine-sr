import logging
from contextlib import redirect_stderr
from io import StringIO

from src.utils.logging import configure_logging


def test_configure_logging_rebinds_stream_handler():
    first_stream = StringIO()
    with redirect_stderr(first_stream):
        configure_logging()
        logging.getLogger(__name__).info("first message")

    second_stream = StringIO()
    with redirect_stderr(second_stream):
        configure_logging()
        logging.getLogger(__name__).info("second message")

    assert "first message" in first_stream.getvalue()
    assert "second message" in second_stream.getvalue()


def test_configure_logging_replaces_existing_root_handlers():
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)

    stale_stream = StringIO()
    stale_handler = logging.StreamHandler(stale_stream)
    root_logger.handlers = [stale_handler]

    try:
        rebound_stream = StringIO()
        with redirect_stderr(rebound_stream):
            configure_logging()
            logging.getLogger(__name__).info("single message")

        assert "single message" not in stale_stream.getvalue()
        assert rebound_stream.getvalue().count("single message") == 1
    finally:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            handler.close()
        root_logger.handlers = original_handlers
