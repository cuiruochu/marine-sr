from scripts import run_queue


def test_load_jobs_ignores_comments_and_blank_lines(tmp_path):
    queue_path = tmp_path / "queue.txt"
    queue_path.write_text(
        "\n# comment\n[\"uv\", \"run\", \"python\", \"-V\"]\n\nuv run python scripts/train.py\n",
        encoding="utf-8",
    )

    jobs = run_queue.load_jobs(queue_path)

    assert len(jobs) == 2
    assert jobs[0].index == 1
    assert jobs[1].index == 2


def test_main_returns_error_for_missing_queue(tmp_path):
    missing_queue = tmp_path / "missing_queue.txt"

    exit_code = run_queue.main(["--queue", str(missing_queue)])

    assert exit_code == 1


def test_main_dry_run_does_not_execute_jobs(monkeypatch, tmp_path):
    queue_path = tmp_path / "queue.txt"
    queue_path.write_text("[\"uv\", \"run\", \"python\", \"-V\"]\n", encoding="utf-8")

    def fail_run(job, *, cwd):
        raise AssertionError("dry-run 不应执行命令")

    monkeypatch.setattr(run_queue, "run_job", fail_run)

    exit_code = run_queue.main(["--queue", str(queue_path), "--dry-run"])

    assert exit_code == 0


def test_main_continues_after_failure_by_default(monkeypatch, tmp_path):
    queue_path = tmp_path / "queue.txt"
    queue_path.write_text(
        "[\"cmd\", \"job1\"]\n[\"cmd\", \"job2\"]\n[\"cmd\", \"job3\"]\n",
        encoding="utf-8",
    )
    called = []
    return_codes = iter([0, 2, 0])

    def fake_run(job, *, cwd):
        called.append((job.index, job.argv, cwd))
        return next(return_codes)

    monkeypatch.setattr(run_queue, "run_job", fake_run)

    exit_code = run_queue.main(["--queue", str(queue_path)])

    assert exit_code == 1
    assert [index for index, _, _ in called] == [1, 2, 3]


def test_main_fail_fast_stops_after_first_failure(monkeypatch, tmp_path):
    queue_path = tmp_path / "queue.txt"
    queue_path.write_text(
        "[\"cmd\", \"job1\"]\n[\"cmd\", \"job2\"]\n[\"cmd\", \"job3\"]\n",
        encoding="utf-8",
    )
    called = []
    return_codes = iter([0, 5, 0])

    def fake_run(job, *, cwd):
        called.append(job.index)
        return next(return_codes)

    monkeypatch.setattr(run_queue, "run_job", fake_run)

    exit_code = run_queue.main(["--queue", str(queue_path), "--fail-fast"])

    assert exit_code == 1
    assert called == [1, 2]


def test_main_start_from_skips_previous_jobs(monkeypatch, tmp_path):
    queue_path = tmp_path / "queue.txt"
    queue_path.write_text(
        "[\"cmd\", \"job1\"]\n[\"cmd\", \"job2\"]\n[\"cmd\", \"job3\"]\n",
        encoding="utf-8",
    )
    called = []

    def fake_run(job, *, cwd):
        called.append(job.index)
        return 0

    monkeypatch.setattr(run_queue, "run_job", fake_run)

    exit_code = run_queue.main(["--queue", str(queue_path), "--start-from", "2"])

    assert exit_code == 0
    assert called == [2, 3]


def test_main_returns_error_for_invalid_line(tmp_path):
    queue_path = tmp_path / "queue.txt"
    queue_path.write_text('["uv", 1]\n', encoding="utf-8")

    exit_code = run_queue.main(["--queue", str(queue_path)])

    assert exit_code == 1


def test_main_returns_130_on_keyboard_interrupt(monkeypatch, tmp_path):
    queue_path = tmp_path / "queue.txt"
    queue_path.write_text("[\"cmd\", \"job1\"]\n", encoding="utf-8")

    def interrupt(job, *, cwd):
        raise KeyboardInterrupt()

    monkeypatch.setattr(run_queue, "run_job", interrupt)

    exit_code = run_queue.main(["--queue", str(queue_path)])

    assert exit_code == 130


def test_main_help_returns_zero():
    exit_code = run_queue.main(["--help"])

    assert exit_code == 0
