"""日志配置工具。"""

import logging
import sys
from pathlib import Path


def build_formatter() -> logging.Formatter:
    return logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def configure_logging(level: int = logging.INFO, file_path: str | Path | None = None) -> None:
    """配置进程内根 logger 的 handler。"""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(build_formatter())
    stream_handler._marine_handler_kind = "stream"
    root_logger.addHandler(stream_handler)

    if file_path is None:
        return

    resolved_path = str(Path(file_path).resolve())
    file_handler = logging.FileHandler(resolved_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(build_formatter())
    file_handler._marine_handler_kind = "file"
    file_handler.baseFilename = resolved_path
    root_logger.addHandler(file_handler)
