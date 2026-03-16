from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


def resolve_path(path_str: str) -> Path:
    """解析路径，相对路径基于 PROJECT_ROOT"""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path