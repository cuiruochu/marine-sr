"""
串行实验队列执行器。

用法示例:
    uv run python scripts/run_queue.py --queue jobs/queue.txt
    uv run python scripts/run_queue.py --queue jobs/queue.txt --dry-run
    uv run python scripts/run_queue.py --queue jobs/queue.txt --start-from 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


PLACEHOLDER_PATTERN = re.compile(r"<[A-Z0-9_]+>")


@dataclass
class QueueJob:
    index: int
    command: str


@dataclass
class QueueResult:
    index: int
    command: str
    status: str
    return_code: int | None
    started_at: str | None
    finished_at: str | None
    duration_seconds: float | None
    log_path: str | None
    note: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="顺序执行实验队列，失败后继续后续任务。")
    parser.add_argument("--queue", type=Path, required=True, help="任务列表文件，每行一条命令。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("queue_runs"),
        help="队列运行输出根目录，默认写入 queue_runs/。",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        help="从第几个任务开始执行（1-based）。默认从第 1 个开始。",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="遇到失败立即停止。默认失败后继续执行后续任务。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要执行的任务，不真正运行。",
    )
    return parser.parse_args()


def load_jobs(queue_path: Path) -> list[QueueJob]:
    if not queue_path.exists():
        raise FileNotFoundError(f"队列文件不存在: {queue_path}")

    jobs: list[QueueJob] = []
    for raw_line in queue_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        jobs.append(QueueJob(index=len(jobs) + 1, command=line))
    return jobs


def ensure_run_dirs(output_root: Path) -> tuple[Path, Path]:
    run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, logs_dir


def has_unresolved_placeholder(command: str) -> list[str]:
    return PLACEHOLDER_PATTERN.findall(command)


def stream_process(command: str, log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8", newline="") as log_file:
        log_file.write(f"$ {command}\n\n")
        log_file.flush()

        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)

        return process.wait()


def dry_run_job(job: QueueJob) -> None:
    print(f"[DRY-RUN] job_{job.index:03d}: {job.command}")


def execute_jobs(
    jobs: Iterable[QueueJob],
    logs_dir: Path,
    dry_run: bool,
    fail_fast: bool,
) -> list[QueueResult]:
    results: list[QueueResult] = []

    for job in jobs:
        placeholders = has_unresolved_placeholder(job.command)
        if placeholders:
            note = f"存在未替换占位符: {', '.join(placeholders)}"
            print(f"[SKIP] job_{job.index:03d} {note}")
            result = QueueResult(
                index=job.index,
                command=job.command,
                status="skipped",
                return_code=None,
                started_at=None,
                finished_at=None,
                duration_seconds=None,
                log_path=None,
                note=note,
            )
            results.append(result)
            if fail_fast:
                break
            continue

        if dry_run:
            dry_run_job(job)
            results.append(
                QueueResult(
                    index=job.index,
                    command=job.command,
                    status="dry_run",
                    return_code=None,
                    started_at=None,
                    finished_at=None,
                    duration_seconds=None,
                    log_path=None,
                )
            )
            continue

        log_path = logs_dir / f"job_{job.index:03d}.log"
        started = datetime.now()
        print(f"[START] job_{job.index:03d}: {job.command}")
        return_code = stream_process(job.command, log_path)
        finished = datetime.now()
        duration = (finished - started).total_seconds()

        status = "success" if return_code == 0 else "failed"
        print(
            f"[{status.upper()}] job_{job.index:03d} exit={return_code} "
            f"duration={duration:.2f}s log={log_path}"
        )

        results.append(
            QueueResult(
                index=job.index,
                command=job.command,
                status=status,
                return_code=return_code,
                started_at=started.isoformat(timespec="seconds"),
                finished_at=finished.isoformat(timespec="seconds"),
                duration_seconds=duration,
                log_path=str(log_path),
            )
        )

        if return_code != 0 and fail_fast:
            break

    return results


def build_summary(results: list[QueueResult], queue_path: Path, dry_run: bool) -> dict:
    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1

    return {
        "queue_path": str(queue_path),
        "dry_run": dry_run,
        "total_jobs": len(results),
        "status_counts": status_counts,
        "results": [asdict(result) for result in results],
    }


def main() -> int:
    args = parse_args()
    jobs = load_jobs(args.queue)
    if not jobs:
        print(f"队列文件中没有可执行任务: {args.queue}")
        return 1

    selected_jobs = [job for job in jobs if job.index >= args.start_from]
    if not selected_jobs:
        print(f"--start-from={args.start_from} 超出任务数量 ({len(jobs)})")
        return 1

    run_dir, logs_dir = ensure_run_dirs(args.output_dir)
    summary_path = run_dir / "summary.json"

    print(f"[QUEUE] queue={args.queue}")
    print(f"[QUEUE] run_dir={run_dir}")
    print(f"[QUEUE] jobs={len(selected_jobs)} / total={len(jobs)}")

    results = execute_jobs(
        jobs=selected_jobs,
        logs_dir=logs_dir,
        dry_run=args.dry_run,
        fail_fast=args.fail_fast,
    )

    summary = build_summary(results, args.queue, args.dry_run)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SUMMARY] {summary_path}")

    any_failed = any(result.status == "failed" for result in results)
    return 1 if any_failed and args.fail_fast else 0


if __name__ == "__main__":
    raise SystemExit(main())
