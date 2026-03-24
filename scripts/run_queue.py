"""串行实验队列执行器。"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHELL_OPERATOR_TOKENS = {"|", "||", "&", "&&", ";", "<", ">"}


@dataclass(frozen=True)
class QueueJob:
    index: int
    command: str
    argv: list[str]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的整数")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="顺序执行实验队列。")
    parser.add_argument("--queue", type=Path, required=True, help="任务列表文件，每行一条命令。")
    parser.add_argument("--start-from", type=positive_int, default=1, help="从第几个任务开始执行（1-based）。")
    parser.add_argument("--fail-fast", action="store_true", help="遇到失败立即停止。")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要执行的任务，不真正运行。")
    return parser


def parse_job_argv(command: str) -> list[str]:
    stripped = command.strip()
    if not stripped:
        raise ValueError("空命令")

    if stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON 命令格式错误: {exc.msg}") from exc
        if not isinstance(parsed, list) or not parsed or not all(isinstance(item, str) for item in parsed):
            raise ValueError("JSON 命令必须是非空字符串数组")
        argv = parsed
    else:
        try:
            argv = shlex.split(stripped, posix=True)
        except ValueError as exc:
            raise ValueError(f"命令解析失败: {exc}") from exc

    shell_tokens = [token for token in argv if token in SHELL_OPERATOR_TOKENS]
    if shell_tokens:
        raise ValueError(f"队列命令不支持 shell 操作符: {', '.join(shell_tokens)}")
    return argv


def load_jobs(queue_path: Path) -> list[QueueJob]:
    if not queue_path.exists():
        raise FileNotFoundError(f"队列文件不存在: {queue_path}")

    jobs = []
    for line_no, raw_line in enumerate(queue_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip().lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue
        try:
            argv = parse_job_argv(line)
        except ValueError as exc:
            raise ValueError(f"{queue_path}:{line_no} {exc}") from exc
        jobs.append(QueueJob(index=len(jobs) + 1, command=line, argv=argv))
    return jobs


def _print_status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def run_job(job: QueueJob, *, cwd: Path) -> int:
    completed = subprocess.run(job.argv, cwd=str(cwd), shell=False, check=False)
    return int(completed.returncode)


def execute_jobs(
    jobs: list[QueueJob],
    *,
    start_from: int,
    fail_fast: bool,
    dry_run: bool,
    cwd: Path,
) -> int:
    selected_jobs = [job for job in jobs if job.index >= start_from]
    if not selected_jobs:
        raise ValueError(f"--start-from={start_from} 超出任务数量 ({len(jobs)})")

    any_failed = False

    for job in selected_jobs:
        if dry_run:
            _print_status(f"[DRY-RUN] job_{job.index:03d}: {job.command}")
            continue

        _print_status(f"[RUN] job_{job.index:03d}: {job.command}")
        return_code = run_job(job, cwd=cwd)
        if return_code == 0:
            _print_status(f"[OK] job_{job.index:03d} exit=0")
            continue

        any_failed = True
        _print_status(f"[FAIL] job_{job.index:03d} exit={return_code}")
        if fail_fast:
            break

    return 1 if any_failed else 0


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        jobs = load_jobs(args.queue)
        if not jobs:
            raise ValueError(f"队列文件中没有可执行任务: {args.queue}")
        return execute_jobs(
            jobs,
            start_from=args.start_from,
            fail_fast=args.fail_fast,
            dry_run=args.dry_run,
            cwd=PROJECT_ROOT,
        )
    except KeyboardInterrupt:
        _print_status("[INTERRUPTED] queue execution cancelled by user")
        return 130
    except (FileNotFoundError, ValueError) as exc:
        _print_status(f"[ERROR] {exc}")
        return 1
    except SystemExit as exc:
        return int(exc.code)


if __name__ == "__main__":
    raise SystemExit(main())
