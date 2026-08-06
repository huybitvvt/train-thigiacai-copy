from __future__ import annotations

import threading
import time

import pytest

from roll_qr_scale.inference_queue import InferenceCoordinator, InferenceQueueFull


def test_inference_coordinator_is_fifo_and_has_exactly_one_worker() -> None:
    coordinator = InferenceCoordinator(max_queue_size=8)
    active = 0
    maximum_active = 0
    order: list[int] = []
    lock = threading.Lock()

    def work(value: int) -> int:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
            order.append(value)
        time.sleep(0.01)
        with lock:
            active -= 1
        return value * 2

    futures = [coordinator.submit(work, value) for value in range(6)]
    assert [future.result(timeout=2) for future in futures] == [0, 2, 4, 6, 8, 10]
    status = coordinator.status()
    coordinator.close()

    assert order == list(range(6))
    assert maximum_active == 1
    assert status.worker_count == 1
    assert status.completed == 6


def test_inference_coordinator_rejects_work_beyond_bounded_queue() -> None:
    coordinator = InferenceCoordinator(max_queue_size=1)
    started = threading.Event()
    release = threading.Event()

    def blocked() -> None:
        started.set()
        assert release.wait(2)

    first = coordinator.submit(blocked)
    assert started.wait(1)
    second = coordinator.submit(lambda: None)
    with pytest.raises(InferenceQueueFull):
        coordinator.submit(lambda: None)
    release.set()
    first.result(timeout=2)
    second.result(timeout=2)
    coordinator.close()
