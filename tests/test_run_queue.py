"""run_queue 队列命令解析测试。"""

from pathlib import Path

import pytest

import scripts.run_queue as run_queue


def test_parse_job_argv_preserves_hydra_override_quotes():
    command = (
        'uv run python scripts/train.py '
        'models=edsr '
        '"dataset.train_lr_root=\'E:/毕业论文代码/dataset/train/wind/x2/lr\'"'
    )

    argv = run_queue.parse_job_argv(command)

    assert argv[:4] == ["uv", "run", "python", "scripts/train.py"]
    assert argv[4] == "models=edsr"
    assert argv[5] == "dataset.train_lr_root='E:/毕业论文代码/dataset/train/wind/x2/lr'"


def test_parse_job_argv_supports_json_array_format():
    command = (
        '["uv", "run", "python", "scripts/train.py", "models=edsr", '
        '"dataset.train_lr_root=\'E:/毕业论文代码/dataset/train/wind/x2/lr\'"]'
    )

    argv = run_queue.parse_job_argv(command)

    assert argv == [
        "uv",
        "run",
        "python",
        "scripts/train.py",
        "models=edsr",
        "dataset.train_lr_root='E:/毕业论文代码/dataset/train/wind/x2/lr'",
    ]


def test_parse_job_argv_rejects_shell_operators():
    with pytest.raises(ValueError, match="不支持 shell 操作符"):
        run_queue.parse_job_argv("uv run python scripts/train.py models=edsr > train.log")


def test_load_jobs_parses_queue_file(monkeypatch):
    queue_path = Path("jobs/queue.txt")
    monkeypatch.setattr(Path, "exists", lambda self: self == queue_path)
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, encoding="utf-8-sig": "\n".join(
            [
                "# comment",
                'uv run python scripts/train.py models=edsr "dataset.name=\'wind\'"',
                '["uv", "run", "python", "scripts/train.py", "models=rcan"]',
            ]
        ),
    )

    jobs = run_queue.load_jobs(queue_path)

    assert len(jobs) == 2
    assert jobs[0].argv[4:] == ["models=edsr", "dataset.name='wind'"]
    assert jobs[1].argv[4:] == ["models=rcan"]
