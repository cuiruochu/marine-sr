"""
项目路径工具
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


def resolve_project_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
