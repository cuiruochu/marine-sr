"""
日志配置模块

提供统一的日志配置，替代 print 语句。
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from src.utils.path import PROJECT_ROOT


# 日志格式
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "marine_sr",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    配置并返回 logger

    Args:
        name: logger 名称
        level: 日志级别
        log_file: 日志文件路径（可选）
        console: 是否输出到控制台

    Returns:
        配置好的 logger
    """
    logger = logging.getLogger(name)

    # 避免重复配置
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    # 控制台输出
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件输出
    if log_file:
        log_path = Path(log_file)
        if not log_path.is_absolute():
            log_path = PROJECT_ROOT / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# 全局 logger
_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    """
    获取全局 logger

    如果未初始化，自动初始化为控制台输出
    """
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger


def init_logger(log_file: Optional[str] = None, level: int = logging.INFO):
    """
    初始化全局 logger

    Args:
        log_file: 日志文件路径
        level: 日志级别
    """
    global _logger
    _logger = setup_logger(log_file=log_file, level=level)


# 便捷函数
def info(msg: str):
    get_logger().info(msg)


def debug(msg: str):
    get_logger().debug(msg)


def warning(msg: str):
    get_logger().warning(msg)


def error(msg: str):
    get_logger().error(msg)