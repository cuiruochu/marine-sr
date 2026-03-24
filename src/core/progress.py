"""批次进度日志辅助。"""


def build_progress_points(total_batches: int) -> set[int]:
    """按总批次数生成固定间隔的进度打印节点。"""
    if total_batches <= 0:
        return set()

    step = max(1, total_batches // 10)
    points = set(range(step, total_batches + 1, step))
    points.add(total_batches)
    return points
